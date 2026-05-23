# Optical Fate Dice Reader: Operator Guide

## What This Tool Does
The Optical Fate Dice Reader watches your physical Fate/Fudge dice tray with a webcam and sends digital roll results to your game app.

You do not need to code to use it.

---

## 1. Prerequisites

You need:
- A Windows PC
- A USB or built-in webcam pointed at your dice tray
- Python installed (3.10+ recommended)
- This project folder on your PC

Before first use:
- Place all dice fully inside the tray area
- Keep tray lighting steady (avoid flickering lights)
- Make sure the camera is not being used by another app (Zoom, Teams, OBS, etc.)

---

## 2. Starting the Software (run.bat)

1. Open the project folder in File Explorer.
2. Double-click `run.bat`.
3. The script will:
- Create a virtual environment if needed
- Install/update required packages from `requirements.txt`
- Start the server locally

When running, keep the terminal window open.

To stop:
- Press `Ctrl + C` in the terminal window

---

## 3. Opening the Dashboard

1. Start the app with `run.bat`.
2. Open a browser and go to:
- `http://127.0.0.1:8000`

The dashboard includes:
- Live camera feed
- System Status panel (`IDLE`, `WATCHING`, `CALCULATING`, `ERROR`)
- Vision Calibration controls
- Recent Rolls history
- Export Diagnostics button

---

## 4. Physical Setup Checklist

Do this before calibration:

1. Mount the camera above the tray so the tray stays in frame.
2. Keep camera and tray fixed (no movement during gameplay).
3. Use even light across the tray.
4. Use high-contrast dice symbols (clean, visible `+`, `-`, blank faces).
5. Make sure only the tray area is relevant in view.

---

## 5. Calibrating ROI and Vision

## 5.1 Set the tray ROI (Region of Interest)
ROI means "only this rectangle matters".

If your build includes ROI controls:
1. Use the ROI controls on the dashboard to set X, Y, Width, Height around the tray.
2. Confirm the magenta ROI border tightly wraps the tray.
3. Keep hands and other objects outside ROI when possible.

If ROI controls are not exposed in your current UI build:
1. Use the app's ROI endpoint/config workflow provided by your installer.
2. Restart and confirm the magenta ROI box appears correctly on the feed.

## 5.2 Use debug views while tuning
Use the `Feed View` dropdown:
- `Raw`: normal camera image
- `Motion Mask`: shows what the system treats as movement
- `Edges`: shows contour outlines used for die detection
- `Thresholded`: shows symbol thresholding used for `+`, `-`, `blank`

Switch between these while adjusting sliders.

## 5.3 Tune sliders
Use these controls in the `Vision Calibration` panel:

1. `Motion Threshold`
- Higher value: less sensitive to tiny motion/noise
- Lower value: more sensitive

2. `Contour Min Area`
- Increase if small noise blobs are counted as dice
- Decrease if real dice are missed

3. `Contour Max Area`
- Decrease if large tray/background shapes get treated as dice
- Increase if real dice boxes are getting filtered out

4. `Symbol Threshold`
- Adjust to improve `+` / `-` / blank separation in `Thresholded` view

Changes apply immediately and are saved.

---

## 6. Normal Roll Flow

1. Game app sends `REQUEST_ROLL`.
2. Status changes:
- `IDLE` -> `WATCHING` -> `CALCULATING` -> `IDLE`
3. Result is sent back to the game app as `ROLL_COMPLETE`.
4. Roll appears in `Recent Rolls`.

If physical reading fails, you can use:
- `Generate Random Roll` for manual fallback

---

## 7. Troubleshooting

## The camera won't connect
Symptoms:
- Status shows `ERROR`
- Feed says camera unavailable

Fix:
1. Close other camera apps (Zoom/Teams/OBS/Camera app).
2. Replug USB camera.
3. Restart `run.bat`.
4. Confirm correct camera is selected in config (`camera_index`).

## It keeps reading `+` as `-`
Fix sequence:
1. Open `Thresholded` view.
2. Adjust `Symbol Threshold` slowly until symbols are clearly separated.
3. Ensure lighting is even (reduce shadows/glare).
4. Clean dice faces if markings are faint.

## Dice count mismatch errors
Fix:
1. Ensure all expected dice are inside ROI.
2. Check `Edges` view for extra contours from tray textures.
3. Raise `Contour Min Area` if noise is counted.
4. Lower `Contour Max Area` if large background shapes are counted.

## Frequent false motion triggers
Fix:
1. Improve room lighting stability.
2. Raise `Motion Threshold`.
3. Tighten ROI so only tray area is included.

## Roll requests rejected as busy
Reason:
- System is already processing a request.

Fix:
1. Wait for current request to finish.
2. Check status panel and message.
3. If stuck, use fallback or restart app.

---

## 8. Diagnostics Export (for support)

If you need help:
1. Click `Export Diagnostics`.
2. A `diagnostics.zip` file is downloaded.
3. Share this zip with support/developer.

Zip contents include:
- `config.json`
- `app.log`
- `state_snapshot.json`

---

## 9. Safe Shutdown

Always stop the app with:
- `Ctrl + C` in the terminal

This allows clean camera release and prevents device lock issues.

---

## 10. Quick Start (Short Version)

1. Run `run.bat`.
2. Open `http://127.0.0.1:8000`.
3. Confirm status is `IDLE`.
4. Calibrate ROI and sliders using debug views.
5. Start gameplay.
6. Use fallback and diagnostics export when needed.

