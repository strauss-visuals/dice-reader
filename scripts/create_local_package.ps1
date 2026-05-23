param(
    [string]$Version = "v1.1.0-local"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $repoRoot "dist"
$packageName = "dice-reader-$Version-private-local"
$stageRoot = Join-Path $distRoot $packageName
$zipPath = Join-Path $distRoot "$packageName.zip"

if (Test-Path $stageRoot) {
    Remove-Item -Recurse -Force $stageRoot
}
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

New-Item -ItemType Directory -Path $stageRoot | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageRoot "templates") | Out-Null

$copyFiles = @(
    "main.py",
    "vision.py",
    "requirements.txt",
    "run.bat",
    "config.json",
    "OPERATOR_GUIDE.md",
    "templates\index.html",
    "templates\app.js"
)

foreach ($relativePath in $copyFiles) {
    $source = Join-Path $repoRoot $relativePath
    if (-not (Test-Path $source)) {
        throw "Missing required package file: $relativePath"
    }
    $destination = Join-Path $stageRoot $relativePath
    $destinationParent = Split-Path -Parent $destination
    if (-not (Test-Path $destinationParent)) {
        New-Item -ItemType Directory -Path $destinationParent | Out-Null
    }
    Copy-Item -Path $source -Destination $destination -Force
}

$manifest = [ordered]@{
    package_name = $packageName
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    version = $Version
    startup_command = ".\run.bat"
    host = "127.0.0.1"
    port = 8000
    includes = $copyFiles
}

$manifestJson = $manifest | ConvertTo-Json -Depth 5
Set-Content -Path (Join-Path $stageRoot "PACKAGE_MANIFEST.json") -Value $manifestJson -Encoding UTF8

$readme = @"
# Private Local Package

This package is meant for local/private use only.

## Run
1. Double-click `run.bat`.
2. Open `http://127.0.0.1:8000` in a browser.

## Contents
- FastAPI application and vision pipeline
- Troubleshooting dashboard UI
- Operator guide
"@
Set-Content -Path (Join-Path $stageRoot "PACKAGE_README.md") -Value $readme -Encoding UTF8

Compress-Archive -Path "$stageRoot\*" -DestinationPath $zipPath -Force

Write-Host "Package created:"
Write-Host "  Stage: $stageRoot"
Write-Host "  Zip:   $zipPath"
