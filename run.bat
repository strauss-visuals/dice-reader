@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
    if errorlevel 1 (
        echo Unable to create a Python virtual environment. Install Python and try again.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. Check the error above and try again.
    pause
    exit /b 1
)

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
    taskkill /PID %%p /F >nul 2>nul
)

python main.py
endlocal
