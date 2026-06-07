@echo off
echo Starting Vein Auto-Annotation ML Backend...
echo.
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..

REM Use SAM2 small model by default
set SAM2_CHECKPOINT=%PROJECT_ROOT%\models\sam2\sam2.1_hiera_small.pt
set PORT=9090
set DEVICE=auto
set CONF_THRESH=0.35

python "%SCRIPT_DIR%ml_backend.py"
pause
