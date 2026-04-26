import atexit
import base64
import os
import re
import ssl
import threading
import time
from collections import deque
from datetime import datetime

import certifi
import cv2
import easyocr
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO


DEFAULT_STREAM_URL = "https://pplterpadu.kedirikota.go.id:8888/tosaren/stream.m3u8"


def configure_ssl_certifi():
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    os.environ.setdefault("CURL_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=cafile)


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
    # Indonesian plates are alphanumeric; keep compact representation.
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


class StreamDetector:
    def __init__(
        self,
        model_path,
        stream_url,
        conf_thres=0.22,
        iou_thres=0.5,
        ocr_lang="en",
        ocr_gpu=False,
        history_size=30,
        fallback_video_path="",
        detect_every_n=2,
        process_width=960,
    ):
        configure_ssl_certifi()

        self.model = YOLO(model_path)
        self.reader = easyocr.Reader([ocr_lang], gpu=ocr_gpu)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.stream_url = stream_url
        self.fallback_video_path = fallback_video_path.strip()
        self.history = deque(maxlen=history_size)
        self.detect_every_n = max(1, int(detect_every_n))
        self.process_width = max(320, int(process_width))

        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.latest_jpeg = None
        self.frame_idx = 0
        self.connected = False
        self.last_error = ""
        self.fps = 0.0
        self.start_time = time.time()
        self.last_update = None
        self.raw_det_count = 0
        self.ocr_success_count = 0
        self.cached_boxes = []
        self.all_detections = deque(maxlen=400)
        self.next_detection_id = 1

    def set_stream_url(self, stream_url):
        with self.lock:
            self.stream_url = stream_url
            self.connected = False
            self.last_error = "Menyambungkan ulang stream..."

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _worker_loop(self):
        last_frame_time = time.time()
        current_url = ""
        cap = None
        using_fallback = False

        while self.running:
            with self.lock:
                target_url = self.stream_url

            if cap is None or target_url != current_url:
                if cap is not None:
                    cap.release()

                current_url = target_url
                cap = cv2.VideoCapture(current_url)
                using_fallback = False

                if not cap.isOpened():
                    cap.release()
                    cap = None
                    if self.fallback_video_path and os.path.exists(self.fallback_video_path):
                        cap = cv2.VideoCapture(self.fallback_video_path)
                        using_fallback = cap.isOpened()

                    if cap is None or not cap.isOpened():
                        with self.lock:
                            self.connected = False
                            self.last_error = f"Gagal membuka stream: {current_url}"
                        time.sleep(2.0)
                        continue

                with self.lock:
                    self.connected = True
                    self.last_error = (
                        ""
                        if not using_fallback
                        else f"Stream utama gagal, memakai fallback: {self.fallback_video_path}"
                    )

            ok, frame = cap.read()
            if not ok or frame is None:
                if using_fallback:
                    cap.release()
                    cap = cv2.VideoCapture(self.fallback_video_path)
                    time.sleep(0.03)
                    continue
                with self.lock:
                    self.connected = False
                    self.last_error = "Frame stream putus, mencoba reconnect..."
                cap.release()
                cap = None
                time.sleep(1.0)
                continue

            self.frame_idx += 1
            run_detect = (self.frame_idx % self.detect_every_n) == 0
            annotated, detections, raw_det_count, ocr_ok_count = self._detect_and_annotate(
                frame,
                run_detect=run_detect,
            )
            ret, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            now = time.time()
            dt = max(now - last_frame_time, 1e-6)
            instant_fps = 1.0 / dt
            last_frame_time = now

            if ret:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()
                    self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
                    self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.raw_det_count += raw_det_count
                    self.ocr_success_count += ocr_ok_count
                    if detections:
                        for item in detections:
                            item["id"] = self.next_detection_id
                            self.next_detection_id += 1
                            self.history.appendleft(item)
                            self.all_detections.appendleft(item)

        if cap is not None:
            cap.release()

    def _encode_crop_data_url(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        h, w = crop_bgr.shape[:2]
        if w > 360:
            scale = 360.0 / float(w)
            crop_bgr = cv2.resize(crop_bgr, (360, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _detect_and_annotate(self, frame, run_detect=True):
        h, w = frame.shape[:2]
        detections_for_ui = []
        raw_det_count = 0
        ocr_ok_count = 0

        if run_detect:
            scale = 1.0
            infer_frame = frame
            if w > self.process_width:
                scale = self.process_width / float(w)
                infer_frame = cv2.resize(frame, (self.process_width, int(h * scale)), interpolation=cv2.INTER_AREA)

            results = self.model.predict(
                infer_frame,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False,
            )[0]
            boxes = results.boxes
            self.cached_boxes = []

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    det_conf = float(confs[i])
                    cls_id = int(clss[i])
                    cls_name = self.model.names.get(cls_id, str(cls_id))

                    if scale != 1.0:
                        inv = 1.0 / scale
                        x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv

                    bw = max(1, int(x2 - x1))
                    bh = max(1, int(y2 - y1))
                    pad_x = int(0.12 * bw)
                    pad_y = int(0.22 * bh)
                    x1, y1, x2, y2 = clamp_bbox(
                        int(x1) - pad_x,
                        int(y1) - pad_y,
                        int(x2) + pad_x,
                        int(y2) + pad_y,
                        w,
                        h,
                    )
                    crop = frame[y1:y2, x1:x2].copy()
                    plate_text, ocr_conf = read_plate_text(self.reader, crop)
                    crop_image = self._encode_crop_data_url(crop)
                    raw_det_count += 1
                    if plate_text:
                        ocr_ok_count += 1

                    self.cached_boxes.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "cls_name": cls_name,
                            "det_conf": det_conf,
                            "plate_text": plate_text,
                            "ocr_conf": float(ocr_conf),
                            "crop_image": crop_image,
                        }
                    )

        for item in self.cached_boxes:
            x1 = item["x1"]
            y1 = item["y1"]
            x2 = item["x2"]
            y2 = item["y2"]
            cls_name = item["cls_name"]
            det_conf = item["det_conf"]
            plate_text = item["plate_text"]
            ocr_conf = item["ocr_conf"]

            det_label = f"{cls_name} {det_conf:.2f}"
            if plate_text:
                det_label += f" | {plate_text} ({ocr_conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 153), 2)
            cv2.putText(
                frame,
                det_label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (15, 28, 40),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                det_label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 255, 240),
                1,
                cv2.LINE_AA,
            )

            if run_detect:
                detections_for_ui.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "class_name": cls_name,
                        "plate": plate_text if plate_text else "-",
                        "det_conf": round(float(det_conf), 3),
                        "ocr_conf": round(float(ocr_conf), 3),
                        "frame_idx": self.frame_idx,
                        "crop_image": item.get("crop_image", ""),
                    }
                )

        cv2.rectangle(frame, (0, 0), (w, 64), (15, 21, 32), -1)
        cv2.putText(
            frame,
            "Live Plate Detection",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Frame: {self.frame_idx} | FPS: {self.fps:.1f} | OCR OK: {self.ocr_success_count}",
            (14, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (179, 203, 255),
            2,
            cv2.LINE_AA,
        )

        return frame, detections_for_ui, raw_det_count, ocr_ok_count

    def get_status(self):
        with self.lock:
            recent = []
            for item in list(self.history)[:20]:
                recent.append(
                    {
                        "id": item.get("id"),
                        "time": item.get("time"),
                        "frame_idx": item.get("frame_idx"),
                        "class_name": item.get("class_name"),
                        "plate": item.get("plate"),
                        "det_conf": item.get("det_conf"),
                        "ocr_conf": item.get("ocr_conf"),
                    }
                )
            return {
                "connected": self.connected,
                "stream_url": self.stream_url,
                "frame_idx": self.frame_idx,
                "fps": round(self.fps, 2),
                "last_error": self.last_error,
                "uptime_sec": int(time.time() - self.start_time),
                "last_update": self.last_update,
                "raw_det_count": self.raw_det_count,
                "ocr_success_count": self.ocr_success_count,
                "detections": recent,
            }

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def list_all_detections(self, limit=200):
        with self.lock:
            items = list(self.all_detections)[: max(1, min(int(limit), 400))]
        out = []
        for item in items:
            out.append(
                {
                    "id": item.get("id"),
                    "time": item.get("time"),
                    "frame_idx": item.get("frame_idx"),
                    "class_name": item.get("class_name"),
                    "plate": item.get("plate"),
                    "det_conf": item.get("det_conf"),
                    "ocr_conf": item.get("ocr_conf"),
                    "has_crop_image": bool(item.get("crop_image")),
                }
            )
        return out

    def get_detection_detail(self, det_id):
        with self.lock:
            for item in self.all_detections:
                if item.get("id") == det_id:
                    return {
                        "id": item.get("id"),
                        "time": item.get("time"),
                        "frame_idx": item.get("frame_idx"),
                        "class_name": item.get("class_name"),
                        "plate": item.get("plate"),
                        "det_conf": item.get("det_conf"),
                        "ocr_conf": item.get("ocr_conf"),
                        "crop_image": item.get("crop_image", ""),
                    }
        return None


app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
STREAM_URL = os.getenv("STREAM_URL", DEFAULT_STREAM_URL)
CONF_THRES = float(os.getenv("CONF_THRES", "0.22"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.5"))
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_GPU = os.getenv("OCR_GPU", "false").lower() == "true"
FALLBACK_VIDEO_PATH = os.getenv("FALLBACK_VIDEO_PATH", "cctv.mp4")
DETECT_EVERY_N = int(os.getenv("DETECT_EVERY_N", "2"))
PROCESS_WIDTH = int(os.getenv("PROCESS_WIDTH", "960"))

engine = StreamDetector(
    model_path=MODEL_PATH,
    stream_url=STREAM_URL,
    conf_thres=CONF_THRES,
    iou_thres=IOU_THRES,
    ocr_lang=OCR_LANG,
    ocr_gpu=OCR_GPU,
    fallback_video_path=FALLBACK_VIDEO_PATH,
    detect_every_n=DETECT_EVERY_N,
    process_width=PROCESS_WIDTH,
)
engine.start()


@app.route("/")
def index():
    return render_template("index.html", stream_url=engine.get_status()["stream_url"])


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = engine.get_latest_jpeg()
            if frame is None:
                time.sleep(0.1)
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(engine.get_status())


@app.route("/api/stream", methods=["POST"])
def api_stream():
    payload = request.get_json(force=True, silent=True) or {}
    stream_url = (payload.get("stream_url") or "").strip()
    if not stream_url:
        return jsonify({"ok": False, "error": "stream_url wajib diisi"}), 400

    engine.set_stream_url(stream_url)
    return jsonify({"ok": True, "stream_url": stream_url})


@app.route("/api/detections")
def api_detections():
    limit = request.args.get("limit", default=200, type=int)
    items = engine.list_all_detections(limit=limit)
    return jsonify({"items": items, "count": len(items)})


@app.route("/api/detections/<int:det_id>")
def api_detection_detail(det_id):
    item = engine.get_detection_detail(det_id)
    if not item:
        return jsonify({"ok": False, "error": "Deteksi tidak ditemukan"}), 404
    return jsonify({"ok": True, "item": item})


@atexit.register
def _shutdown_engine():
    engine.stop()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=False, threaded=True)
