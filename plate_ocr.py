"""OCR plat nomor.

Default memakai fast-plate-ocr (model CCT khusus plat, dilatih pada plat dari 60+ negara
termasuk Indonesia, berjalan via ONNX Runtime di CPU dalam hitungan milidetik).
EasyOCR tetap tersedia sebagai cadangan lewat OCR_ENGINE=easyocr.
"""

import re

import cv2
import numpy as np


# Format plat Indonesia: 1-2 huruf kode wilayah, 1-4 angka, 0-3 huruf seri. Contoh: AB 1234 CD, B 9168 PXE.
PLATE_RE = re.compile(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$")
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def normalize_plate_text(text):
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


def is_valid_plate(text):
    return bool(PLATE_RE.match(text or ""))


def plate_distance(a, b):
    """Jarak Levenshtein, untuk menggabungkan bacaan plat yang hanya beda 1 karakter."""
    a, b = a or "", b or ""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def format_plate(text):
    """'AB1234CD' -> 'AB 1234 CD' (teks yang tidak sesuai format dikembalikan apa adanya)."""
    m = PLATE_RE.match(text or "")
    if not m:
        return text or ""
    return " ".join(part for part in m.groups() if part)


class PlateOCR:
    def __init__(self, engine="fastplate", model_name="cct-s-v2-global-model", lang="en", gpu=False):
        self.engine = engine
        self.model_name = model_name
        if engine == "fastplate":
            from fast_plate_ocr import LicensePlateRecognizer

            self.model = LicensePlateRecognizer(model_name, device="cuda" if gpu else "cpu")
        elif engine == "easyocr":
            import easyocr

            self.model = easyocr.Reader([lang], gpu=gpu, verbose=False)
        else:
            raise ValueError(f"OCR_ENGINE tidak dikenal: {engine}")

    @property
    def name(self):
        return f"fast-plate-ocr ({self.model_name})" if self.engine == "fastplate" else "EasyOCR"

    def read(self, crop_bgr):
        """Kembalikan (teks_ternormalisasi, confidence 0..1)."""
        if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 4:
            return "", 0.0
        if self.engine == "fastplate":
            return self._read_fastplate(crop_bgr)
        return self._read_easyocr(crop_bgr)

    def _read_fastplate(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pred = self.model.run(rgb, return_confidence=True)[0]
        text = normalize_plate_text(pred.plate)
        if not text:
            return "", 0.0
        probs = np.asarray(pred.char_probs if pred.char_probs is not None else [], dtype=np.float32).ravel()
        conf = float(probs[: len(text)].mean()) if probs.size else 0.0
        return text, conf

    def _read_easyocr(self, crop_bgr):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape[:2]
        if h < 40 or w < 120:
            gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        best_text, best_conf = "", 0.0
        for candidate in (gray, cv2.GaussianBlur(gray, (3, 3), 0)):
            for _, txt, conf in self.model.readtext(candidate, allowlist=ALLOWLIST, paragraph=False, detail=1):
                txt = normalize_plate_text(txt)
                if len(txt) >= 3 and conf > best_conf:
                    best_text, best_conf = txt, float(conf)
        return best_text, best_conf
