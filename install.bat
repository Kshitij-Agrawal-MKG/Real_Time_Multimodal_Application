@echo off
chcp 65001 >nul 2>&1
echo.
echo ============================================================
echo   Voice Assistant -- Windows 11 Setup
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Download: https://www.python.org/downloads/
    echo         Check "Add Python to PATH" during install.
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PV=%%v
echo [OK] Python %PV%

echo [1/5] Upgrading pip...
python -m pip install --upgrade pip -q

echo [2/5] Installing PyAudio...
python -m pip install pyaudio -q 2>nul
if errorlevel 1 (
    python -m pip install pipwin -q && python -m pipwin install pyaudio
    if errorlevel 1 (
        echo [ERROR] PyAudio install failed.
        echo         Try: pip install pipwin ^&^& pipwin install pyaudio
        pause & exit /b 1
    )
)
echo [OK] PyAudio installed

echo [3/5] Installing dependencies...
python -m pip install vosk websockets google-genai boto3 python-dotenv -q
if errorlevel 1 ( echo [ERROR] Failed. & pause & exit /b 1 )
echo [OK] Dependencies installed

echo [4/5] Creating .env...
if not exist .env (
    copy .env.example .env >nul
    echo [OK] Created .env -- edit it and add your API keys
) else (
    echo [OK] .env already exists
)

echo [5/5] Vosk model check...
if exist vosk_model\conf (
    echo [OK] Vosk model found
) else (
    echo [WARN] Vosk model not found. Download it:
    echo        PowerShell: Invoke-WebRequest https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -OutFile model.zip
    echo        Then:       Expand-Archive model.zip . ^&^& Rename-Item vosk-model-small-en-us-0.15 vosk_model
)

echo.
echo ============================================================
echo   Running tests...
echo ============================================================
python tests.py
echo.
echo ============================================================
echo   Done! Next steps:
echo     1. Edit .env with your Gemini and AWS keys
echo     2. Download vosk_model\ if not done (see above)
echo     3. python check_setup.py   (verify everything)
echo     4. python main.py          (start the assistant)
echo     5. Open dashboard.html     (live metrics)
echo ============================================================
echo.
pause
