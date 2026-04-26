import os
import re
import cv2
import time
import ssl
import argparse
import pandas as pd
from ultralytics import YOLO
import easyocr
import certifi


def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(1, min(xmax, w))
    ymax = max(1, min(ymax, h))
    if xmax <= xmin:
        xmax = min(w, xmin + 1)
    if ymax <= ymin:
        ymax = min(h, ymin + 1)
    return xmin, ymin, xmax, ymax


def normalize_plate_text(text):
    return "".join(ch for ch in text.upper() if ch.isalnum())


def normalize_plate_candidate(text, min_chars=3):
    txt = normalize_plate_text(text)
    if len(txt) < min_chars:
        return ""
    return re.sub(r"[^A-Z0-9]", "", txt)


def read_plate_text(reader, crop_bgr, min_conf=0.15):
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape[:2]
    if h < 40 or w < 120:
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    best_text = ""
    best_conf = 0.0
    candidates = [
        gray,
        cv2.GaussianBlur(gray, (3, 3), 0),
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        ),
    ]

    for candidate in candidates:
        results = reader.readtext(
            candidate,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            paragraph=False,
            detail=1,
        )
        for _, txt, conf in results:
            txt_norm = normalize_plate_candidate(txt, min_chars=3)
            if conf > best_conf and txt_norm:
                best_text = txt_norm
                best_conf = float(conf)

    if best_conf < min_conf:
        return "", best_conf
    return best_text, best_conf


def configure_ssl_certifi():
    # Force urllib/ssl to use certifi CA bundle (helps on some conda-based macOS Python setups).
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    os.environ.setdefault("CURL_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=cafile)


def run_inference(
    model_path,
    video_path,
    output_video_path,
    output_csv_path,
    evidence_dir,
    conf_thres=0.22,
    iou_thres=0.5,
    ocr_lang="en",
    ocr_gpu=True,
    max_evidence=50,
    save_every_n=30,
):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)
    configure_ssl_certifi()

    detector = YOLO(model_path)
    reader = easyocr.Reader([ocr_lang], gpu=ocr_gpu)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {width}x{height}, FPS={fps:.2f}, frames={num_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    logs = []
    frame_idx = 0
    saved_evidence = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)[0]
        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                cls_id = int(clss[i])
                cls_name = detector.names.get(cls_id, str(cls_id))

                bw = max(1, int(x2 - x1))
                bh = max(1, int(y2 - y1))
                pad_x = int(0.12 * bw)
                pad_y = int(0.22 * bh)
                x1, y1, x2, y2 = clamp_bbox(
                    int(x1) - pad_x,
                    int(y1) - pad_y,
                    int(x2) + pad_x,
                    int(y2) + pad_y,
                    width,
                    height,
                )
                crop = frame[y1:y2, x1:x2].copy()
                plate_text, ocr_conf = read_plate_text(reader, crop)
                t_sec = frame_idx / max(fps, 1e-6)

                label = f"{cls_name} {conf:.2f}"
                if plate_text:
                    label += f" | {plate_text} ({ocr_conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

                logs.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(t_sec, 3),
                        "class_name": cls_name,
                        "det_conf": round(conf, 4),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "plate_text": plate_text,
                        "ocr_conf": round(float(ocr_conf), 4),
                    }
                )

                if plate_text and (frame_idx % save_every_n == 0) and (saved_evidence < max_evidence):
                    ev_path = os.path.join(evidence_dir, f"frame_{frame_idx:06d}_{plate_text}.jpg")
                    cv2.imwrite(ev_path, frame)
                    saved_evidence += 1

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"Time: {frame_idx / max(fps, 1e-6):.2f}s",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - start
    print(f"Inference selesai dalam {elapsed:.2f}s")
    print("Output video:", output_video_path)

    df = pd.DataFrame(logs)
    df.to_csv(output_csv_path, index=False)
    print("Output CSV:", output_csv_path)
    print("Total deteksi:", len(df))


def main():
    parser = argparse.ArgumentParser(description="CCTV License Plate Detection + OCR")
    parser.add_argument("--model", required=True, help="Path model best.pt")
    parser.add_argument("--video", required=True, help="Path video input")
    parser.add_argument("--out-video", default="artifacts/cctv_plate_annotated.mp4")
    parser.add_argument("--out-csv", default="artifacts/cctv_plate_ocr_log.csv")
    parser.add_argument("--evidence-dir", default="artifacts/evidence_frames")
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--ocr-lang", default="en")
    parser.add_argument("--ocr-gpu", action="store_true")
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        video_path=args.video,
        output_video_path=args.out_video,
        output_csv_path=args.out_csv,
        evidence_dir=args.evidence_dir,
        conf_thres=args.conf,
        iou_thres=args.iou,
        ocr_lang=args.ocr_lang,
        ocr_gpu=args.ocr_gpu,
    )


if __name__ == "__main__":
    main()
