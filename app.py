import atexit
import base64
import os
import ssl
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import certifi
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

from plate_enhance import PlateEnhancer, crop_quality
from plate_ocr import PlateOCR, format_plate, is_valid_plate, plate_distance


# Kamera ANPR (Automatic Number Plate Recognition) Kota Yogyakarta: 1280x720, dipasang rendah dan
# menghadap lajur, sehingga plat terlihat ~40-120 px. Kamera ATCS Kediri hanya 704x576 dengan sudut
# lebar perempatan (plat ~13-20 px), terlalu kecil untuk dibaca model apa pun.
CAMERA_PRESETS = [
    {"name": "Jogja ANPR - Jl. Wardhani", "url": "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-Wardhani.stream/playlist.m3u8"},
    {"name": "Jogja ANPR - Jl. Ahmad Jazuli", "url": "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-Ahmad-Jazuli.stream/playlist.m3u8"},
    {"name": "Jogja ANPR - Jl. Yos Sudarso", "url": "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-Yos-Sudarso.stream/playlist.m3u8"},
    {"name": "Jogja ANPR - Jl. FM Noto (McDonalds)", "url": "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-FM-Noto-McDonalds.stream/playlist.m3u8"},
    {"name": "Jogja ANPR - Jl. FM Noto (Raminten)", "url": "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-FM-Noto-Raminten.stream/playlist.m3u8"},
    {"name": "Jogja ANPR - Simpang Gramedia", "url": "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Simpang-Gramedia-V-Timur_Ex-Jl-Prau.stream/playlist.m3u8"},
    {"name": "Kediri ATCS - Simpang Baruna (704x576)", "url": "https://pplterpadu.kedirikota.go.id:8888/baruna/stream.m3u8"},
    {"name": "Kediri ATCS - Simpang Tosaren (704x576)", "url": "https://pplterpadu.kedirikota.go.id:8888/tosaren/stream.m3u8"},
]
DEFAULT_STREAM_URL = CAMERA_PRESETS[0]["url"]


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


def padded_crop(frame, box, pad_x, pad_y):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1, x2, y2 = clamp_bbox(
        int(x1 - pad_x * bw), int(y1 - pad_y * bh), int(x2 + pad_x * bw), int(y2 + pad_y * bh), w, h
    )
    return frame[y1:y2, x1:x2].copy()


def encode_data_url(img_bgr, max_width=None, quality=92):
    if img_bgr is None or img_bgr.size == 0:
        return ""
    h, w = img_bgr.shape[:2]
    if max_width and w > max_width:
        img_bgr = cv2.resize(img_bgr, (max_width, max(1, int(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class FrameGrabber:
    """Membaca stream di thread sendiri dan hanya menyimpan frame terbaru.

    Decoding dijalankan di proses ffmpeg terpisah (imageio-ffmpeg) dan dibatasi GRAB_FPS, sehingga
    tidak berebut CPU/GIL dengan YOLO dan video tidak tertinggal dari live. Kalau imageio-ffmpeg
    tidak tersedia, dipakai cv2.VideoCapture. File video (fallback) diputar real-time dan diulang.
    """

    def __init__(self, stream_url, fallback_video_path="", grab_fps=10):
        self.stream_url = stream_url
        self.fallback_video_path = fallback_video_path.strip()
        self.grab_fps = max(1, int(grab_fps))
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.connected = False
        self.using_fallback = False
        self.error = ""
        self.resolution = ""
        self.running = False
        self.thread = None
        try:
            import imageio_ffmpeg

            self._ffmpeg = imageio_ffmpeg
        except ImportError:
            self._ffmpeg = None

    def set_url(self, url):
        with self.cond:
            self.stream_url = url
            self.connected = False
            self.error = "Menyambungkan ulang stream..."

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def latest(self, after_seq, timeout=1.0):
        with self.cond:
            if self.seq <= after_seq:
                self.cond.wait(timeout)
            return self.seq, self.frame

    def _ffmpeg_frames(self, source, is_file):
        # -re: baca sesuai kecepatan asli. Server HLS mengirim segmen secara bergelombang; tanpa -re
        # frame datang berdempetan lalu kosong beberapa detik, sehingga banyak frame terbuang.
        input_params = ["-re", "-threads", "1"]
        if is_file:
            input_params = ["-stream_loop", "-1"] + input_params
        else:
            input_params = ["-rw_timeout", "15000000"] + input_params
        gen = self._ffmpeg.read_frames(
            source,
            pix_fmt="bgr24",
            input_params=input_params,
            output_params=["-an", "-vf", f"fps={self.grab_fps}"],
        )
        meta = next(gen)  # gagal di sini = stream tidak bisa dibuka
        w, h = meta["size"]

        def frames():
            try:
                for buf in gen:
                    yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            finally:
                gen.close()

        return frames()

    def _cv2_frames(self, source, is_file):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("cv2.VideoCapture gagal membuka sumber")
        step = 1.0 / self.grab_fps
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        def frames():
            next_t, src_t = time.time(), 0.0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if is_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        return
                    if is_file:
                        # Lewati frame agar sesuai GRAB_FPS, dan putar real-time.
                        src_t += 1.0 / src_fps
                        if src_t < step:
                            continue
                        src_t -= step
                        next_t += step
                        delay = next_t - time.time()
                        if delay > 0:
                            time.sleep(delay)
                        else:
                            next_t = time.time()
                    yield frame
            finally:
                cap.release()

        return frames()

    def _open_source(self, source, is_file):
        if self._ffmpeg is not None:
            return self._ffmpeg_frames(source, is_file)
        return self._cv2_frames(source, is_file)

    def _open(self, url):
        try:
            return self._open_source(url, os.path.isfile(url)), False
        except Exception:
            pass
        if self.fallback_video_path and os.path.exists(self.fallback_video_path):
            try:
                return self._open_source(self.fallback_video_path, True), True
            except Exception:
                pass
        return None, False

    def _loop(self):
        while self.running:
            with self.cond:
                current_url = self.stream_url
            frames, fallback = self._open(current_url)
            with self.cond:
                if frames is None:
                    self.connected = False
                    self.error = f"Gagal membuka stream: {current_url}"
                else:
                    self.connected = True
                    self.using_fallback = fallback
                    self.error = (
                        f"Stream utama gagal, memakai video fallback: {self.fallback_video_path}" if fallback else ""
                    )
            if frames is None:
                time.sleep(2.0)
                continue

            try:
                for frame in frames:
                    with self.cond:
                        if not self.running or self.stream_url != current_url:
                            break
                        self.frame = frame
                        self.seq += 1
                        self.connected = True
                        self.resolution = f"{frame.shape[1]}x{frame.shape[0]}"
                        self.cond.notify_all()
                else:
                    with self.cond:
                        self.connected = False
                        self.error = "Frame stream putus, mencoba reconnect..."
                    time.sleep(1.0)
            except Exception as exc:
                with self.cond:
                    self.connected = False
                    self.error = f"Stream error: {exc}; mencoba reconnect..."
                time.sleep(1.0)
            finally:
                frames.close()


class StreamDetector:
    def __init__(
        self,
        model_path,
        stream_url,
        conf_thres=0.3,
        iou_thres=0.5,
        imgsz=960,
        ocr_engine="fastplate",
        ocr_model="cct-s-v2-global-model",
        ocr_lang="en",
        ocr_gpu=False,
        enhance=True,
        min_ocr_conf=0.6,
        fallback_video_path="",
        detect_every_n=1,
        track_timeout=1.5,
        jpeg_quality=90,
        grab_fps=10,
    ):
        configure_ssl_certifi()

        self.model = YOLO(model_path)
        self.ocr = PlateOCR(engine=ocr_engine, model_name=ocr_model, lang=ocr_lang, gpu=ocr_gpu)
        self.enhancer = PlateEnhancer(enabled=enhance)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = int(imgsz)
        self.min_ocr_conf = min_ocr_conf
        self.detect_every_n = max(1, int(detect_every_n))
        self.track_timeout = track_timeout
        self.jpeg_quality = int(jpeg_quality)

        self.grabber = FrameGrabber(stream_url, fallback_video_path, grab_fps=grab_fps)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.latest_jpeg = None
        self.frame_idx = 0
        self.fps = 0.0
        self.start_time = time.time()
        self.last_update = None
        self.raw_det_count = 0
        self.ocr_success_count = 0

        self.tracks = {}
        self.next_track_id = 1
        self.cached_boxes = []
        self.all_detections = deque(maxlen=400)
        self.detections_by_track = {}
        self.next_detection_id = 1

    # ------------------------------------------------------------------ lifecycle
    def set_stream_url(self, stream_url):
        self.grabber.set_url(stream_url)
        with self.lock:
            self.tracks.clear()
            self.cached_boxes = []

    def start(self):
        if self.running:
            return
        self.running = True
        self.grabber.start()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.grabber.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _worker_loop(self):
        last_seq = 0
        last_frame_time = time.time()
        while self.running:
            seq, frame = self.grabber.latest(last_seq)
            if frame is None or seq == last_seq:
                continue
            last_seq = seq
            self.frame_idx += 1
            run_detect = (self.frame_idx % self.detect_every_n) == 0
            annotated = self._process(frame.copy(), run_detect)
            ok, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

            now = time.time()
            instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
            last_frame_time = now
            if ok:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()
                    self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
                    self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------ pipeline
    def _detect(self, frame):
        results = self.model.predict(
            frame, conf=self.conf_thres, iou=self.iou_thres, imgsz=self.imgsz, verbose=False
        )[0]
        out = []
        if results.boxes is None:
            return out
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            bw, bh = x2 - x1, y2 - y1
            # Plat selalu lebih lebar dari tingginya; buang kotak dengan rasio tidak wajar.
            if bw < 8 or bh < 4 or bw < bh * 1.2 or bw > bh * 7:
                continue
            out.append(
                {
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                    "det_conf": float(conf),
                    "cls_name": self.model.names.get(int(cls_id), str(cls_id)),
                }
            )
        return out

    def _match_tracks(self, dets, now):
        """Asosiasi sederhana berdasarkan jarak pusat kotak (greedy), cukup untuk plat yang bergerak pelan."""
        pairs = []
        for t_id, tr in self.tracks.items():
            tx1, ty1, tx2, ty2 = tr["box"]
            tcx, tcy, tw = (tx1 + tx2) / 2, (ty1 + ty2) / 2, tx2 - tx1
            for d_idx, det in enumerate(dets):
                x1, y1, x2, y2 = det["box"]
                cx, cy, w = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1
                ratio = w / max(tw, 1.0)
                dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                if 0.5 < ratio < 2.0 and dist < 2.5 * max(w, tw):
                    pairs.append((dist / max(w, tw), t_id, d_idx))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, t_id, d_idx in pairs:
            if t_id in used_t or d_idx in used_d:
                continue
            used_t.add(t_id)
            used_d.add(d_idx)
            dets[d_idx]["track_id"] = t_id
        for det in dets:
            if "track_id" not in det:
                det["track_id"] = self.next_track_id
                self.tracks[self.next_track_id] = {
                    "votes": defaultdict(float),
                    "best_score": 0.0,
                    "first_seen": now,
                    "hits": 0,
                    "reads": 0,
                }
                self.next_track_id += 1
            tr = self.tracks[det["track_id"]]
            tr["box"] = det["box"]
            tr["last_seen"] = now
            tr["hits"] += 1
        for t_id in [t for t, tr in self.tracks.items() if now - tr["last_seen"] > self.track_timeout]:
            del self.tracks[t_id]

    def _process(self, frame, run_detect):
        now = time.time()
        if run_detect:
            dets = self._detect(frame)
            with self.lock:
                self._match_tracks(dets, now)
            boxes = []
            for det in dets:
                ocr_crop = padded_crop(frame, det["box"], 0.04, 0.08)
                text, ocr_conf = self.ocr.read(ocr_crop)
                valid = is_valid_plate(text) and ocr_conf >= self.min_ocr_conf
                self._update_track(det, frame, ocr_crop, text if valid else "", ocr_conf)
                tr = self.tracks.get(det["track_id"], {})
                boxes.append({**det, "plate": tr.get("plate", ""), "ocr_conf": tr.get("plate_conf", 0.0)})
                with self.lock:
                    self.raw_det_count += 1
                    if valid:
                        self.ocr_success_count += 1
            self.cached_boxes = boxes

        for item in self.cached_boxes:
            self._draw_box(frame, item)
        self._draw_header(frame)
        return frame

    def _update_track(self, det, frame, ocr_crop, text, ocr_conf):
        tr = self.tracks.get(det["track_id"])
        if tr is None:
            return
        if text:
            tr["votes"][text] += ocr_conf * det["det_conf"]
            tr["reads"] += 1
        if not tr["votes"]:
            return

        # Hasil akhir = teks dengan bobot voting terbesar di seluruh frame track ini.
        plate, weight = max(tr["votes"].items(), key=lambda kv: kv[1])
        reads = sum(tr["votes"].values())
        tr["plate"] = plate
        tr["plate_conf"] = weight / max(reads, 1e-6)

        score = crop_quality(ocr_crop, det["det_conf"]) * (1.0 + ocr_conf if text == plate else 1.0)
        is_better = score > tr["best_score"] * 1.1
        if is_better:
            tr["best_score"] = score
            display_crop = padded_crop(frame, det["box"], 0.10, 0.25)
            enhanced = self.enhancer.enhance(display_crop)
            context = padded_crop(frame, det["box"], 2.5, 6.0)
            tr["images"] = {
                "crop_image": encode_data_url(enhanced, max_width=720),
                "raw_image": encode_data_url(display_crop),
                "context_image": encode_data_url(context, max_width=640, quality=85),
                "thumb": encode_data_url(enhanced, max_width=200, quality=85),
                "det_conf": det["det_conf"],
                "frame_idx": self.frame_idx,
                "plate_px": int(det["box"][2] - det["box"][0]),
            }
        self._publish_track(det["track_id"], tr, det["cls_name"])

    def _publish_track(self, track_id, tr, cls_name):
        """Satu baris riwayat per kendaraan, diperbarui saat voting/gambar membaik."""
        images = tr.get("images", {})
        with self.lock:
            item = self.detections_by_track.get(track_id)
            if item is None:
                # Kendaraan yang sama kadang muncul lagi sebagai track baru (terhalang / salah asosiasi);
                # gabungkan jika platnya sama atau hanya beda 1 karakter dan baru saja terlihat.
                for prev in list(self.all_detections)[:20]:
                    if time.time() - prev["_updated"] < 10 and plate_distance(prev["plate_raw"], tr["plate"]) <= 1:
                        item = prev
                        tr["votes"].update({k: v for k, v in prev["_votes"].items() if k not in tr["votes"]})
                        break
            if item is None:
                item = {"id": self.next_detection_id, "time": datetime.now().strftime("%H:%M:%S"), "_best": 0.0}
                self.next_detection_id += 1
                self.all_detections.appendleft(item)
            self.detections_by_track[track_id] = item
            item["_votes"] = tr["votes"]
            plate, weight = max(tr["votes"].items(), key=lambda kv: kv[1])
            tr["plate"], tr["plate_conf"] = plate, weight / max(sum(tr["votes"].values()), 1e-6)
            item.update(
                {
                    "class_name": cls_name,
                    "plate_raw": tr["plate"],
                    "plate": format_plate(tr["plate"]),
                    "ocr_conf": round(tr["plate_conf"], 3),
                    "reads": tr["reads"],
                    "hits": tr["hits"],
                    "_updated": time.time(),
                }
            )
            if images and tr["best_score"] > item["_best"]:
                item["_best"] = tr["best_score"]
                item.update(images)
                item["det_conf"] = round(images["det_conf"], 3)
            if len(self.detections_by_track) > 500:
                for old in list(self.detections_by_track)[:-200]:
                    del self.detections_by_track[old]

    # ------------------------------------------------------------------ drawing
    def _draw_box(self, frame, item):
        x1, y1, x2, y2 = (int(v) for v in item["box"])
        has_plate = bool(item["plate"])
        color = (36, 255, 153) if has_plate else (0, 190, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = format_plate(item["plate"]) if has_plate else f"{item['det_conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 8
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), (15, 28, 40), -1)
        cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _draw_header(self, frame):
        w = frame.shape[1]
        strip = frame[:34]
        cv2.addWeighted(strip, 0.35, strip * 0 + (15, 21, 32), 0.65, 0, strip, dtype=cv2.CV_8U)
        cv2.putText(
            frame,
            f"Live Plate Detection | Frame {self.frame_idx} | FPS {self.fps:.1f} | Plat terbaca {self.ocr_success_count}",
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 235, 255),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------ API
    @staticmethod
    def _public(item, with_images=False):
        out = {
            "id": item.get("id"),
            "time": item.get("time"),
            "frame_idx": item.get("frame_idx"),
            "class_name": item.get("class_name"),
            "plate": item.get("plate") or "-",
            "det_conf": item.get("det_conf"),
            "ocr_conf": item.get("ocr_conf"),
            "hits": item.get("hits"),
            "reads": item.get("reads"),
            "plate_px": item.get("plate_px"),
            "thumb": item.get("thumb", ""),
        }
        if with_images:
            out["crop_image"] = item.get("crop_image", "")
            out["raw_image"] = item.get("raw_image", "")
            out["context_image"] = item.get("context_image", "")
        return out

    def get_status(self):
        g = self.grabber
        with g.cond:
            connected, stream_url, err, res = g.connected, g.stream_url, g.error, g.resolution
        with self.lock:
            recent = [self._public(item) for item in list(self.all_detections)[:12]]
            return {
                "connected": connected,
                "stream_url": stream_url,
                "resolution": res,
                "frame_idx": self.frame_idx,
                "fps": round(self.fps, 2),
                "last_error": err or self.enhancer.error,
                "uptime_sec": int(time.time() - self.start_time),
                "last_update": self.last_update,
                "raw_det_count": self.raw_det_count,
                "ocr_success_count": self.ocr_success_count,
                "vehicle_count": len(self.all_detections),
                "ocr_engine": self.ocr.name,
                "enhancer": self.enhancer.mode,
                "detections": recent,
            }

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def list_all_detections(self, limit=200):
        with self.lock:
            items = list(self.all_detections)[: max(1, min(int(limit), 400))]
            return [self._public(item) for item in items]

    def get_detection_detail(self, det_id):
        with self.lock:
            for item in self.all_detections:
                if item.get("id") == det_id:
                    return self._public(item, with_images=True)
        return None


app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
STREAM_URL = os.getenv("STREAM_URL", DEFAULT_STREAM_URL)
CONF_THRES = float(os.getenv("CONF_THRES", "0.3"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.5"))
# PROCESS_WIDTH = ukuran input YOLO (imgsz). 960 menangkap plat kecil lebih baik dari 640 bawaan.
PROCESS_WIDTH = int(os.getenv("PROCESS_WIDTH", "960"))
OCR_ENGINE = os.getenv("OCR_ENGINE", "fastplate")
OCR_MODEL = os.getenv("OCR_MODEL", "cct-s-v2-global-model")
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_GPU = os.getenv("OCR_GPU", "false").lower() == "true"
MIN_OCR_CONF = float(os.getenv("MIN_OCR_CONF", "0.6"))
ENHANCE = os.getenv("ENHANCE", "true").lower() == "true"
FALLBACK_VIDEO_PATH = os.getenv("FALLBACK_VIDEO_PATH", "cctv.mp4")
DETECT_EVERY_N = int(os.getenv("DETECT_EVERY_N", "1"))
# Jumlah frame per detik yang diambil dari kamera (decoding di proses ffmpeg terpisah).
GRAB_FPS = int(os.getenv("GRAB_FPS", "10"))

engine = StreamDetector(
    model_path=MODEL_PATH,
    stream_url=STREAM_URL,
    conf_thres=CONF_THRES,
    iou_thres=IOU_THRES,
    imgsz=PROCESS_WIDTH,
    ocr_engine=OCR_ENGINE,
    ocr_model=OCR_MODEL,
    ocr_lang=OCR_LANG,
    ocr_gpu=OCR_GPU,
    enhance=ENHANCE,
    min_ocr_conf=MIN_OCR_CONF,
    fallback_video_path=FALLBACK_VIDEO_PATH,
    detect_every_n=DETECT_EVERY_N,
    grab_fps=GRAB_FPS,
)
engine.start()


@app.route("/")
def index():
    return render_template("index.html", stream_url=engine.get_status()["stream_url"], cameras=CAMERA_PRESETS)


@app.route("/video_feed")
def video_feed():
    def generate():
        last = None
        while True:
            frame = engine.get_latest_jpeg()
            if frame is None or frame is last:
                time.sleep(0.02)
                continue
            last = frame
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(engine.get_status())


@app.route("/api/cameras")
def api_cameras():
    return jsonify({"items": CAMERA_PRESETS})


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
