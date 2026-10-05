# Vehicle Plate Detection YOLO OCR

Aplikasi ini sekarang mendukung:
- Inference file video (`inference_cctv.py`)
- Live streaming `m3u8` + deteksi + dashboard modern di localhost (`app.py`)

## Pipeline Live Dashboard

1. **Kamera**: default memakai kamera **ANPR Kota Yogyakarta** (1280x720, dipasang menghadap lajur).
   Kamera ATCS Kediri hanya 704x576 dengan sudut lebar perempatan, sehingga plat cuma ~13-20 px
   (huruf ~3-4 px) dan tidak bisa dibaca model apa pun. Di kamera ANPR Jogja plat ~40-120 px.
   Kamera bisa diganti dari dropdown di dashboard; preset Kediri tetap ada untuk perbandingan.
2. **Pembacaan stream**: decoding di proses ffmpeg terpisah (`imageio-ffmpeg`), dibatasi `GRAB_FPS`,
   supaya tidak berebut CPU dengan YOLO dan video tidak tertinggal dari live.
3. **Deteksi**: YOLO `best.pt` dengan input `PROCESS_WIDTH` (960) — sebelumnya efektif hanya 640.
4. **Tracking + frame terbaik**: setiap plat dilacak antar-frame; untuk tiap kendaraan disimpan crop
   terbaik (paling besar, tajam, dan yakin), bukan satu baris per frame.
5. **OCR**: `fast-plate-ocr` (model `cct-s-v2-global-model`, khusus plat, data latih termasuk
   Indonesia). Hasil divalidasi dengan format plat Indonesia (`AB 1234 CD`) lalu di-voting dari
   banyak frame. Pada uji crop CCTV Jogja, EasyOCR kebanyakan salah, sedangkan fast-plate-ocr
   membaca plat dengan benar (confidence ~0.99).
6. **Enhancement gambar**: crop terbaik diperbesar x4 dengan Real-ESRGAN `realesr-general-x4v3`
   (`models/realesr-general-x4v3.pth`, ~5 MB, otomatis diunduh bila tidak ada). Di modal Detail
   ditampilkan hasil enhancement, crop asli kamera, dan konteks kendaraan. OCR tetap dijalankan
   pada crop asli supaya hasilnya tidak terpengaruh detail yang "dikarang" oleh super-resolution.

## Jalankan Live Dashboard

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
PROCESS_WIDTH=960 CONF_THRES=0.3 python app.py
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

set PROCESS_WIDTH=960
set CONF_THRES=0.3
python app.py
```

### Windows - manual (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:PROCESS_WIDTH="960"; $env:CONF_THRES="0.3"
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

`https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-Wardhani.stream/playlist.m3u8`

Jika stream utama gagal, aplikasi otomatis fallback ke `cctv.mp4` (bisa diubah via env).

## Opsi Environment Variable

- `MODEL_PATH` (default: `best.pt`)
- `STREAM_URL` (default: kamera ANPR Jogja Jl. Wardhani)
- `CONF_THRES` (default: `0.3`)
- `IOU_THRES` (default: `0.5`)
- `OCR_ENGINE` (`fastplate` / `easyocr`, default: `fastplate`)
- `OCR_MODEL` (default: `cct-s-v2-global-model`; versi lebih ringan: `cct-xs-v2-global-model`)
- `MIN_OCR_CONF` (default: `0.6`, batas confidence OCR per frame sebelum ikut voting)
- `ENHANCE` (`true` / `false`, default: `true`, super-resolution Real-ESRGAN untuk gambar plat)
- `OCR_LANG` (default: `en`, hanya untuk EasyOCR)
- `OCR_GPU` (`true` / `false`, default: `false`)
- `FALLBACK_VIDEO_PATH` (default: `cctv.mp4`)
- `GRAB_FPS` (default: `10`, frame per detik yang diambil dari kamera)
- `DETECT_EVERY_N` (default: `1`, lebih besar = lebih ringan/smooth)
- `PROCESS_WIDTH` (default: `960`, ukuran input YOLO; lebih kecil = lebih ringan, plat jauh terlewat)
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
