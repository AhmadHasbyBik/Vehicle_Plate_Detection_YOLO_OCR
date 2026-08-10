# Vehicle Plate Detection YOLO OCR

Aplikasi ini sekarang mendukung:
- Inference file video (`inference_cctv.py`)
- Live streaming `m3u8` + deteksi + dashboard modern di localhost (`app.py`)

## Jalankan Live Dashboard

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
DETECT_EVERY_N=2 PROCESS_WIDTH=960 CONF_THRES=0.22 python app.py
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:DETECT_EVERY_N="2"; $env:PROCESS_WIDTH="960"; $env:CONF_THRES="0.22"
python app.py
```

Jika PowerShell menolak menjalankan script aktivasi, sekali saja:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Windows (Command Prompt / cmd)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

set DETECT_EVERY_N=2
set PROCESS_WIDTH=960
set CONF_THRES=0.22
python app.py
```

Catatan Windows:
- Env var **tidak bisa** ditulis di depan perintah (`VAR=1 python app.py` hanya jalan di bash/zsh) — pakai `set` / `$env:` seperti di atas.
- `pip install -r requirements.txt` otomatis menarik PyTorch versi CPU. Untuk GPU NVIDIA, install torch CUDA dulu, baru requirements:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```
  lalu jalankan dengan `$env:OCR_GPU="true"`.

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

### macOS / Linux

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

### Windows (PowerShell)

Pemisah baris pakai backtick `` ` ``, bukan `\`:

```powershell
.\.venv\Scripts\Activate.ps1
python inference_cctv.py `
  --model best.pt `
  --video "cctv.mp4" `
  --out-video artifacts\cctv_plate_annotated.mp4 `
  --out-csv artifacts\cctv_plate_ocr_log.csv `
  --evidence-dir artifacts\evidence_frames `
  --conf 0.22 --iou 0.5
```

Di cmd, ganti backtick dengan `^`, atau tulis satu baris penuh.
