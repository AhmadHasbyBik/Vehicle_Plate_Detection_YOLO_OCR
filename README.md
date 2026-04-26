# Vehicle Plate Detection YOLO OCR

Aplikasi ini sekarang mendukung:
- Inference file video (`inference_cctv.py`)
- Live streaming `m3u8` + deteksi + dashboard modern di localhost (`app.py`)

## Jalankan Live Dashboard

```bash
source .venv/bin/activate
pip install -r requirements.txt
DETECT_EVERY_N=2 PROCESS_WIDTH=960 CONF_THRES=0.22 python app.py
```

Buka browser:

`http://localhost:5000`

Default stream:

`https://pplterpadu.kedirikota.go.id:8888/tosaren/stream.m3u8`

Jika stream utama gagal, aplikasi otomatis fallback ke `cctv.mp4` (bisa diubah via env).

## Opsi Environment Variable

- `MODEL_PATH` (default: `best.pt`)
- `STREAM_URL` (default: URL Kediri)
- `CONF_THRES` (default: `0.22`)
- `IOU_THRES` (default: `0.5`)
- `OCR_LANG` (default: `en`)
- `OCR_GPU` (`true` / `false`, default: `false`)
- `FALLBACK_VIDEO_PATH` (default: `cctv.mp4`)
- `DETECT_EVERY_N` (default: `2`, lebih besar = lebih ringan/smooth)
- `PROCESS_WIDTH` (default: `960`, lebih kecil = lebih ringan)
- `HOST` (default: `127.0.0.1`)
- `PORT` (default: `5000`)

## Jalankan Inference Video File

```bash
source .venv/bin/activate
python inference_cctv.py \
  --model best.pt \
  --video "cctv.mp4" \
  --out-video artifacts/cctv_plate_annotated.mp4 \
  --out-csv artifacts/cctv_plate_ocr_log.csv \
  --evidence-dir artifacts/evidence_frames \
  --conf 0.22 --iou 0.5
```
