param(
    [string]$Version = "v1.1.0-standalone"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $repoRoot "dist"
$packageName = "dice-reader-$Version"
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
    "STANDALONE.md",
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
    mode = "standalone"
    startup_command = ".\run.bat"
    host = "127.0.0.1"
    port = 8000
    includes = $copyFiles
}

$manifestJson = $manifest | ConvertTo-Json -Depth 5
Set-Content -Path (Join-Path $stageRoot "PACKAGE_MANIFEST.json") -Value $manifestJson -Encoding UTF8

Compress-Archive -Path "$stageRoot\*" -DestinationPath $zipPath -Force

Write-Host "Standalone package created:"
Write-Host "  Stage: $stageRoot"
Write-Host "  Zip:   $zipPath"
