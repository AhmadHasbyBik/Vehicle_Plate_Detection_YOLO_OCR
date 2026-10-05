"""Peningkatan kualitas crop plat nomor dari CCTV resolusi rendah.

- Super-resolution x4 memakai Real-ESRGAN `realesr-general-x4v3` (SRVGGNetCompact, ~5 MB)
  yang dilatih untuk foto dunia nyata yang buram / terkompresi (cocok untuk CCTV H.264).
- Fallback ke upscale Lanczos + unsharp mask kalau bobot model tidak tersedia.
- Skor kualitas crop (ukuran x ketajaman x confidence) untuk memilih frame terbaik per kendaraan.
"""

import os
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SR_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "realesr-general-x4v3.pth")
SR_WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"


class SRVGGNetCompact(nn.Module):
    """Arsitektur kompak Real-ESRGAN (identik dengan basicsr/realesrgan, tanpa dependensi tambahan)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def _download_weights(path):
    import urllib.request

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(SR_WEIGHTS_URL, tmp)
    os.replace(tmp, path)


def unsharp(img, amount=0.8, sigma=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def classic_upscale(crop_bgr, scale=4):
    up = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return unsharp(up, amount=0.6, sigma=1.2)


class PlateEnhancer:
    def __init__(self, weights_path=DEFAULT_SR_WEIGHTS, device="auto", enabled=True, max_input_width=200):
        self.max_input_width = max_input_width
        self.model = None
        self.device = "cpu"
        self.lock = threading.Lock()
        self.error = ""
        if not enabled:
            return
        try:
            if not os.path.exists(weights_path):
                _download_weights(weights_path)
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            state = torch.load(weights_path, map_location="cpu")
            state = state.get("params_ema", state.get("params", state))
            model = SRVGGNetCompact()
            model.load_state_dict(state, strict=True)
            self.model = model.eval().to(device)
            self.device = device
        except Exception as exc:  # model SR opsional: tetap jalan dengan fallback klasik
            self.model = None
            self.error = f"Super-resolution nonaktif ({exc}); memakai upscale klasik."

    @property
    def mode(self):
        return f"Real-ESRGAN x4 ({self.device})" if self.model is not None else "Lanczos x4 + sharpen"

    def enhance(self, crop_bgr):
        """Kembalikan crop yang sudah diperbesar & dipertajam (BGR uint8)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        h, w = crop_bgr.shape[:2]
        if w > self.max_input_width:
            s = self.max_input_width / float(w)
            crop_bgr = cv2.resize(crop_bgr, (self.max_input_width, max(1, int(h * s))), interpolation=cv2.INTER_AREA)

        if self.model is None:
            return classic_upscale(crop_bgr)

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self.lock, torch.inference_mode():
            out = self.model(tensor)
        out = out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0 + 0.5).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def sharpness(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_quality(crop_bgr, det_conf):
    """Skor untuk memilih frame terbaik: plat lebih besar, lebih tajam, dan lebih yakin = lebih baik."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    sharp = min(sharpness(crop_bgr), 2000.0)
    return float(w * h) ** 0.5 * (1.0 + np.log1p(sharp)) * float(det_conf)
