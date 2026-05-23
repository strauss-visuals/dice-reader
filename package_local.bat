@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    powershell -ExecutionPolicy Bypass -File ".\scripts\create_local_package.ps1"
) else (
    powershell -ExecutionPolicy Bypass -File ".\scripts\create_local_package.ps1" -Version "%~1"
)

if errorlevel 1 (
    echo Packaging failed.
    exit /b 1
)

echo Packaging complete.
endlocal
