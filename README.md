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

### Windows - cara tercepat (venv otomatis)

Klik dua kali `run_windows.bat`, atau dari terminal:

```bat
run_windows.bat
```

Script ini membuat `.venv` kalau belum ada, install dependency ke dalamnya, lalu menjalankan
`app.py` memakai `.venv\Scripts\python.exe`. Tidak butuh aktivasi, jadi **tidak kena error
ExecutionPolicy**. Ubah `DETECT_EVERY_N` / `PROCESS_WIDTH` / `CONF_THRES` langsung di dalam file `.bat`.

### Windows - manual (Command Prompt / cmd)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt

set DETECT_EVERY_N=2
set PROCESS_WIDTH=960
set CONF_THRES=0.22
python app.py
```

### Windows - manual (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:DETECT_EVERY_N="2"; $env:PROCESS_WIDTH="960"; $env:CONF_THRES="0.22"
python app.py
```

Kalau muncul `Activate.ps1 cannot be loaded because running scripts is disabled on this system`,
jalankan salah satu:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # permanen untuk user ini
Set-ExecutionPolicy -Scope Process Bypass             # hanya untuk jendela terminal ini
```

Atau lewati aktivasi dan panggil python venv langsung:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Catatan Windows:
- Pastikan venv benar-benar aktif sebelum `pip install`. Kalau aktivasi gagal, `pip` akan memasang
  paket ke Python global, bukan ke `.venv`. Cek dengan `python -c "import sys; print(sys.prefix)"` —
  hasilnya harus menunjuk ke folder `.venv` proyek ini.
- Env var **tidak bisa** ditulis di depan perintah (`VAR=1 python app.py` hanya jalan di bash/zsh) — pakai `set` / `$env:` seperti di atas.
- `pip install -r requirements.txt` otomatis menarik PyTorch versi CPU (unduhan ±2 GB saat venv baru).
  Untuk GPU NVIDIA, install torch CUDA dulu, baru requirements:
  ```powershell
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  python -m pip install -r requirements.txt
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

Pemisah baris pakai backtick `` ` ``, bukan `\`. Tanpa aktivasi, ganti `python` dengan
`.\.venv\Scripts\python.exe`:

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
