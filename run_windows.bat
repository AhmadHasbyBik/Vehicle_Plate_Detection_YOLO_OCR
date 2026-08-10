@echo off
REM ============================================================
REM  Live Dashboard - Windows launcher (pakai virtual environment)
REM  Klik dua kali file ini, atau jalankan: run_windows.bat
REM ============================================================
setlocal
cd /d "%~dp0"

set VENV_PY=.venv\Scripts\python.exe

REM 1) Buat venv kalau belum ada
if not exist "%VENV_PY%" (
    echo [1/3] Membuat virtual environment di .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo GAGAL membuat venv. Pastikan Python terpasang dan ada di PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment sudah ada, lanjut.
)

REM 2) Install dependency ke dalam venv
echo [2/3] Install dependency ke dalam .venv ...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo GAGAL install dependency.
    pause
    exit /b 1
)

REM 3) Konfigurasi + jalankan (ubah nilai di bawah sesuai kebutuhan)
set DETECT_EVERY_N=2
set PROCESS_WIDTH=960
set CONF_THRES=0.22
set OCR_GPU=false

echo [3/3] Menjalankan dashboard di http://localhost:5000
echo Tekan CTRL+C untuk berhenti.
"%VENV_PY%" app.py

pause
endlocal
