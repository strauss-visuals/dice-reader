@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    powershell -ExecutionPolicy Bypass -File ".\scripts\create_standalone_package.ps1"
) else (
    powershell -ExecutionPolicy Bypass -File ".\scripts\create_standalone_package.ps1" -Version "%~1"
)

if errorlevel 1 (
    echo Standalone packaging failed.
    exit /b 1
)

echo Standalone package created successfully.
endlocal
