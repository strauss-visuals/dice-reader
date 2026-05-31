# Dice Reader Standalone Split

This repository now supports a clean standalone export that is independent from MythosGrid.

## Create standalone package

From this folder:

```bat
split_standalone.bat
```

Optional custom version tag:

```bat
split_standalone.bat v1.2.0
```

## Output

The script creates:

- `dist\dice-reader-<version>\` (staging folder)
- `dist\dice-reader-<version>.zip` (portable package)

## Run package

1. Unzip package.
2. Double-click `run.bat`.
3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Notes

- The standalone package includes only Dice Reader runtime files and UI templates.
- No MythosGrid runtime files are required.
