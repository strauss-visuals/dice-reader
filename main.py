from __future__ import annotations

import atexit
import asyncio
from collections import deque
from datetime import datetime, timezone
import io
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import secrets
import uuid
import signal
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from vision import VisionConfig, VisionProcessor

logger = logging.getLogger("dice_reader")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler("app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


class RollConfig(BaseModel):
    expected_dice_count: int = Field(default=3, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)


class DieResult(BaseModel):
    value: Literal["+", "-", "blank"]
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(min_length=4, max_length=4)


class RollPayload(BaseModel):
    total_score: int
    dice: list[DieResult]
    is_fallback: bool
    roll_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback_reason: str | None = None


class SystemState(BaseModel):
    status: Literal["IDLE", "WATCHING", "CALCULATING", "ERROR"]
    message: str | None = None
    active_dice_count: int = 0


class GameCommand(BaseModel):
    action: str
    request_id: str | None = None
    config: RollConfig | None = None


class RuntimeState(BaseModel):
    system: SystemState
    expected_dice_count: int = 3


class VisionTuningConfig(BaseModel):
    motion_threshold: int = Field(default=1200, ge=1)
    contour_min_area: int = Field(default=500, ge=1)
    contour_max_area: int = Field(default=30000, ge=1)
    symbol_threshold_value: int = Field(default=127, ge=0, le=255)


class CalibrationProfile(BaseModel):
    roi: list[int] | None = Field(default=None, min_length=4, max_length=4)
    vision: VisionTuningConfig = Field(default_factory=VisionTuningConfig)


class AppConfig(BaseModel):
    camera_index: int = Field(default=0, ge=0)
    roi: list[int] | None = Field(default=None, min_length=4, max_length=4)
    vision: VisionTuningConfig = Field(default_factory=VisionTuningConfig)
    calibration_profiles: dict[str, CalibrationProfile] = Field(default_factory=dict)


class CameraOption(BaseModel):
    index: int
    available: bool


class CameraSelectRequest(BaseModel):
    camera_index: int = Field(ge=0)


class CalibrationProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9 _-]+$")


class RoiConfig(BaseModel):
    roi: list[int] | None = Field(default=None, min_length=4, max_length=4)


class CalibrationQuality(BaseModel):
    status: Literal["GOOD", "CHECK", "POOR"]
    score: int = Field(ge=0, le=100)
    message: str
    motion_pixels: int
    recent_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RollHistoryItem(BaseModel):
    timestamp_utc: str
    request_id: str | None = None
    total_score: int
    is_fallback: bool
    dice_count: int
    dice: list[DieResult]
    roll_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback_reason: str | None = None
    snapshot_id: str | None = None


runtime_state = RuntimeState(
    system=SystemState(status="IDLE", message="Waiting for roll request", active_dice_count=0),
    expected_dice_count=3,
)
app_config = AppConfig()
vision = VisionProcessor(
    VisionConfig(
        motion_threshold=app_config.vision.motion_threshold,
        contour_min_area=app_config.vision.contour_min_area,
        contour_max_area=app_config.vision.contour_max_area,
        symbol_threshold_value=app_config.vision.symbol_threshold_value,
        roi=tuple(app_config.roi) if app_config.roi else None,
    )
)
game_bridge_socket: WebSocket | None = None
watch_timeout_task: asyncio.Task | None = None
error_reset_task: asyncio.Task | None = None
heartbeat_task: asyncio.Task | None = None
main_event_loop: asyncio.AbstractEventLoop | None = None
active_request_id: str | None = None
last_heartbeat_ts: float = 0.0

cap: cv2.VideoCapture | None = None
camera_lock = threading.Lock()
camera_index = 0
camera_cleanup_done = False
camera_thread: threading.Thread | None = None
camera_stop_event = threading.Event()
last_camera_reconnect_attempt: float = 0.0
camera_reconnect_interval_seconds = 2.0

latest_frame_lock = threading.Lock()
latest_jpeg_frame: bytes | None = None
latest_raw_frame = None
debug_overlay_lock = threading.Lock()
debug_dice_results: list[dict] = []
latest_motion_pixels: int = 0
config_path = Path("config.json")
history = deque(maxlen=50)
roll_snapshots: dict[str, bytes] = {}


def append_history(payload: RollPayload, request_id: str | None, snapshot_bytes: bytes | None = None) -> None:
    snapshot_id: str | None = None
    if snapshot_bytes is not None:
        snapshot_id = str(uuid.uuid4())
        roll_snapshots[snapshot_id] = snapshot_bytes

    item = RollHistoryItem(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
        total_score=payload.total_score,
        is_fallback=payload.is_fallback,
        dice_count=len(payload.dice),
        dice=payload.dice,
        roll_confidence=payload.roll_confidence,
        fallback_reason=payload.fallback_reason,
        snapshot_id=snapshot_id,
    )
    history.append(item)

    # Keep snapshot memory aligned with capped history window.
    active_snapshot_ids = {entry.snapshot_id for entry in history if entry.snapshot_id}
    stale_ids = [key for key in roll_snapshots.keys() if key not in active_snapshot_ids]
    for stale_id in stale_ids:
        del roll_snapshots[stale_id]


def log_event(level: int, message: str, request_id: str | None = None) -> None:
    suffix = f" | request_id={request_id}" if request_id else ""
    logger.log(level, f"{message}{suffix}")


def calculate_calibration_quality() -> CalibrationQuality:
    if runtime_state.system.status == "ERROR":
        return CalibrationQuality(
            status="POOR",
            score=0,
            message=runtime_state.system.message or "Resolve the system error before calibration.",
            motion_pixels=latest_motion_pixels,
            recent_confidence=None,
        )

    with latest_frame_lock:
        has_camera_frame = latest_raw_frame is not None
    if not has_camera_frame:
        return CalibrationQuality(
            status="CHECK",
            score=0,
            message="Waiting for camera frames.",
            motion_pixels=latest_motion_pixels,
            recent_confidence=None,
        )

    vision_rolls = [item for item in history if not item.is_fallback and item.roll_confidence is not None]
    recent_confidence = vision_rolls[-1].roll_confidence if vision_rolls else None
    motion_limit = max(1, vision.config.motion_threshold)
    motion_ratio = min(1.0, latest_motion_pixels / float(motion_limit))
    score = 100 - int(motion_ratio * 50)

    messages: list[str] = []
    if latest_motion_pixels >= motion_limit:
        messages.append("Tray is moving or lighting is unstable.")
    if recent_confidence is not None:
        score -= int((1.0 - recent_confidence) * 50)
        if recent_confidence < 0.7:
            messages.append("Recent symbol confidence is low.")

    score = max(0, min(100, score))
    if score >= 80:
        return CalibrationQuality(
            status="GOOD",
            score=score,
            message="Setup is stable for testing." if not messages else " ".join(messages),
            motion_pixels=latest_motion_pixels,
            recent_confidence=recent_confidence,
        )
    if score >= 55:
        status: Literal["GOOD", "CHECK", "POOR"] = "CHECK"
    else:
        status = "POOR"
    return CalibrationQuality(
        status=status,
        score=score,
        message=" ".join(messages) or "Review ROI and threshold settings.",
        motion_pixels=latest_motion_pixels,
        recent_confidence=recent_confidence,
    )


def set_state(status: Literal["IDLE", "WATCHING", "CALCULATING", "ERROR"], message: str, active_dice_count: int) -> None:
    previous = runtime_state.system.status
    runtime_state.system.status = status
    runtime_state.system.message = message
    runtime_state.system.active_dice_count = active_dice_count
    if previous != status:
        log_event(logging.INFO, f"State transition {previous} -> {status}: {message}", request_id=active_request_id)


def cancel_task(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        if main_event_loop is not None and threading.current_thread() is not threading.main_thread():
            main_event_loop.call_soon_threadsafe(task.cancel)
        else:
            task.cancel()


def schedule_coroutine(coro) -> None:
    if main_event_loop is None:
        return
    asyncio.run_coroutine_threadsafe(coro, main_event_loop)


def write_config() -> None:
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(app_config.model_dump(), file, indent=2)


def apply_config_to_runtime() -> None:
    global camera_index
    camera_index = app_config.camera_index
    vision.config.motion_threshold = app_config.vision.motion_threshold
    vision.config.contour_min_area = app_config.vision.contour_min_area
    vision.config.contour_max_area = app_config.vision.contour_max_area
    vision.config.symbol_threshold_value = app_config.vision.symbol_threshold_value
    vision.config.roi = tuple(app_config.roi) if app_config.roi else None


def load_or_create_config() -> None:
    global app_config
    if not config_path.exists():
        app_config = AppConfig()
        write_config()
        apply_config_to_runtime()
        logger.info("Created default config.json")
        return

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)
    app_config = AppConfig.model_validate(raw_config)
    apply_config_to_runtime()


def update_debug_overlay(results: list[dict] | None = None, motion_pixels: int | None = None) -> None:
    global debug_dice_results, latest_motion_pixels
    with debug_overlay_lock:
        if results is not None:
            debug_dice_results = list(results)
        if motion_pixels is not None:
            latest_motion_pixels = motion_pixels


def draw_debug_overlay(frame):
    with debug_overlay_lock:
        current_results = list(debug_dice_results)
        motion_value = latest_motion_pixels

    cv2.putText(
        frame,
        f"State: {runtime_state.system.status} | Motion: {motion_value}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        runtime_state.system.message or "",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    if vision.config.roi is not None:
        roi_x, roi_y, roi_w, roi_h = vision.config.roi
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 255), 2)
        cv2.putText(frame, "ROI", (roi_x, max(20, roi_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    for item in current_results:
        x, y, w, h = item["bbox"]
        label = f'{item["value"]} ({item["confidence"]:.2f})'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return frame


async def broadcast_event(event_name: str, payload: dict, request_id: str | None = None) -> None:
    if game_bridge_socket is None:
        return

    try:
        message = {"event": event_name, "data": payload}
        if request_id is not None:
            message["request_id"] = request_id
        await game_bridge_socket.send_json(message)
    except Exception as error:  # pragma: no cover - best effort socket send
        log_event(logging.WARNING, f"Broadcast failed for {event_name}: {error}", request_id=request_id)


async def heartbeat_loop(interval_seconds: int = 10) -> None:
    global last_heartbeat_ts
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            if game_bridge_socket is not None:
                await broadcast_event("PING", {"ts": int(time.time())})
                if last_heartbeat_ts > 0 and (time.time() - last_heartbeat_ts) > (interval_seconds * 3):
                    logger.warning("Heartbeat timeout, closing stale game bridge socket.")
                    await game_bridge_socket.close()
    except asyncio.CancelledError:
        logger.info("Heartbeat task cancelled.")
        raise


def run_startup_diagnostics() -> bool:
    config_ok = True
    camera_ok = True

    if not config_path.exists():
        config_ok = False
        logger.error("Startup diagnostic: config.json is missing.")
        set_state(status="ERROR", message="Config Not Readable", active_dice_count=0)
    else:
        try:
            with config_path.open("r", encoding="utf-8") as file:
                json.load(file)
            logger.info("Startup diagnostic: config.json is readable.")
        except (OSError, json.JSONDecodeError) as error:
            config_ok = False
            logger.error("Startup diagnostic: config.json read failed: %s", error)
            set_state(status="ERROR", message="Config Not Readable", active_dice_count=0)

    with camera_lock:
        current_cap = cap
        if current_cap is None or not current_cap.isOpened():
            camera_ok = False

    if not camera_ok:
        logger.critical("Startup diagnostic: camera open failed.")
        set_state(status="ERROR", message="Camera Not Found", active_dice_count=0)
    elif config_ok:
        set_state(status="IDLE", message="Waiting for roll request", active_dice_count=0)

    return config_ok and camera_ok


def generate_fallback_roll(expected_dice_count: int) -> RollPayload:
    symbol_choices = ["+", "-", "blank"]
    symbol_to_score = {"+": 1, "-": -1, "blank": 0}
    dice: list[DieResult] = []
    total_score = 0

    for _ in range(expected_dice_count):
        symbol = secrets.choice(symbol_choices)
        total_score += symbol_to_score[symbol]
        dice.append(DieResult(value=symbol, confidence=1.0, bbox=[0, 0, 0, 0]))

    return RollPayload(total_score=total_score, dice=dice, is_fallback=True)


async def reset_to_idle_after_delay(delay_seconds: float) -> None:
    await asyncio.sleep(delay_seconds)
    if runtime_state.system.status == "ERROR":
        set_state(status="IDLE", message="Waiting for roll request", active_dice_count=0)
        update_debug_overlay(results=[], motion_pixels=0)
        logger.info("State auto-reset to IDLE after timeout error.")


async def handle_roll_timeout(timeout_seconds: int) -> None:
    global error_reset_task
    try:
        await asyncio.sleep(timeout_seconds)
        if runtime_state.system.status == "WATCHING":
            set_state(
                status="ERROR",
                message="Timeout reached: no settlement detected.",
                active_dice_count=runtime_state.expected_dice_count,
            )
            log_event(logging.WARNING, f"Roll request timed out after {timeout_seconds} seconds", request_id=active_request_id)
            await broadcast_event(
                "ROLL_ERROR",
                {"reason": "TIMEOUT_REACHED_NO_SETTLEMENT"},
                request_id=active_request_id,
            )
            cancel_task(error_reset_task)
            error_reset_task = asyncio.create_task(reset_to_idle_after_delay(2.0))
    except asyncio.CancelledError:
        logger.info("Roll timeout task cancelled.")
        raise


def process_settled_frame(frame) -> None:
    global active_request_id, watch_timeout_task

    set_state(
        status="CALCULATING",
        message="Dice settled. Calculating roll.",
        active_dice_count=runtime_state.expected_dice_count,
    )

    parsed_results = vision.calculate_roll(frame)
    update_debug_overlay(results=parsed_results)
    expected_count = runtime_state.expected_dice_count

    if len(parsed_results) != expected_count:
        set_state(
            status="ERROR",
            message=f"Detected {len(parsed_results)} dice, expected {expected_count}.",
            active_dice_count=len(parsed_results),
        )
        schedule_coroutine(
            broadcast_event("ROLL_ERROR", {"reason": "DICE_COUNT_MISMATCH"}, request_id=active_request_id)
        )
        schedule_coroutine(reset_to_idle_after_delay(2.0))
        vision.reset_motion_history()
        return

    dice = [DieResult.model_validate(item) for item in parsed_results]
    symbol_to_score = {"+": 1, "-": -1, "blank": 0}
    total_score = sum(symbol_to_score[die.value] for die in dice)

    roll_confidence = round(sum(die.confidence for die in dice) / max(1, len(dice)), 3)
    payload = RollPayload(
        total_score=total_score,
        dice=dice,
        is_fallback=False,
        roll_confidence=roll_confidence,
        fallback_reason=None,
    )
    schedule_coroutine(broadcast_event("ROLL_COMPLETE", payload.model_dump(), request_id=active_request_id))
    snapshot_bytes = None
    ok, encoded = cv2.imencode(".jpg", draw_debug_overlay(frame.copy()))
    if ok:
        snapshot_bytes = encoded.tobytes()
    append_history(payload, active_request_id, snapshot_bytes=snapshot_bytes)
    log_event(logging.INFO, f"Roll complete with score={payload.total_score}", request_id=active_request_id)

    cancel_task(watch_timeout_task)
    watch_timeout_task = None
    cancel_task(error_reset_task)
    error_reset_task = None

    set_state(status="IDLE", message="Roll complete", active_dice_count=0)
    active_request_id = None
    vision.reset_motion_history()


def camera_worker_loop() -> None:
    global last_camera_reconnect_attempt, latest_jpeg_frame, latest_raw_frame
    frame_interval = 1.0 / vision.config.processing_fps
    frame_failure_started: float | None = None

    while not camera_stop_event.is_set():
        with camera_lock:
            current_cap = cap
            if current_cap is None or not current_cap.isOpened():
                frame = None
            else:
                ok, frame = current_cap.read()
                if not ok:
                    frame = None

        if frame is None:
            now = time.monotonic()
            if frame_failure_started is None:
                frame_failure_started = now
            if now - frame_failure_started >= camera_reconnect_interval_seconds:
                if runtime_state.system.status not in {"WATCHING", "CALCULATING"}:
                    set_state(status="ERROR", message="Camera Not Providing Frames", active_dice_count=0)
            if now - last_camera_reconnect_attempt >= camera_reconnect_interval_seconds:
                last_camera_reconnect_attempt = now
                if reconnect_camera():
                    log_event(logging.INFO, f"Camera reconnected on index {camera_index}")
                elif runtime_state.system.status != "ERROR":
                    set_state(status="ERROR", message="Camera Not Found", active_dice_count=0)
            time.sleep(0.05)
            continue

        frame_failure_started = None
        if runtime_state.system.status == "ERROR" and runtime_state.system.message in {
            "Camera Not Found",
            "Camera Not Providing Frames",
        }:
            set_state(status="IDLE", message="Camera reconnected", active_dice_count=0)

        with latest_frame_lock:
            latest_raw_frame = frame.copy()

        if runtime_state.system.status == "WATCHING":
            motion_pixels = vision.detect_motion(frame)
            update_debug_overlay(motion_pixels=motion_pixels)
            has_settled = vision.update_settlement(motion_pixels)
            if has_settled:
                process_settled_frame(frame)
        else:
            vision.reset_motion_history()
            update_debug_overlay(motion_pixels=0)

        display_frame = draw_debug_overlay(frame.copy())
        success, encoded = cv2.imencode(".jpg", display_frame)
        if success:
            with latest_frame_lock:
                latest_jpeg_frame = encoded.tobytes()

        time.sleep(frame_interval)


def cleanup_camera() -> None:
    global cap, camera_cleanup_done
    with camera_lock:
        if camera_cleanup_done:
            return
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("Camera released.")
        cap = None
        camera_cleanup_done = True


def open_camera_device(index: int) -> cv2.VideoCapture | None:
    candidate = cv2.VideoCapture(index)
    if candidate is None or not candidate.isOpened():
        if candidate is not None:
            candidate.release()
        return None
    return candidate


def switch_camera(index: int) -> bool:
    global cap
    with camera_lock:
        previous = cap
        replacement = open_camera_device(index)
        if replacement is None:
            return False

        cap = replacement
        if previous is not None and previous.isOpened():
            previous.release()
    return True


def reconnect_camera() -> bool:
    global cap
    with camera_lock:
        previous = cap
        if previous is not None:
            previous.release()
        cap = open_camera_device(camera_index)
        return cap is not None


def handle_exit_signal(signum: int, _frame: object) -> None:
    logger.info("Received shutdown signal %s. Releasing camera.", signum)
    camera_stop_event.set()
    cleanup_camera()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global camera_cleanup_done, camera_thread, cap, last_camera_reconnect_attempt, main_event_loop

    main_event_loop = asyncio.get_running_loop()
    camera_cleanup_done = False
    camera_stop_event.clear()
    last_camera_reconnect_attempt = 0.0
    config_ok = True

    try:
        load_or_create_config()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        logger.error("Config load failed during startup: %s", error)
        set_state(status="ERROR", message="Config Not Readable", active_dice_count=0)
        config_ok = False

    with camera_lock:
        cap = open_camera_device(camera_index)
        if cap is None:
            logger.error("Could not open camera at index %s.", camera_index)
        else:
            logger.info("Camera initialized at index %s.", camera_index)

    run_startup_diagnostics()

    if config_ok:
        camera_thread = threading.Thread(target=camera_worker_loop, name="camera-worker", daemon=True)
        camera_thread.start()
        log_event(logging.INFO, f"Camera worker started on index {camera_index}")

    try:
        yield
    finally:
        camera_stop_event.set()
        if camera_thread is not None and camera_thread.is_alive():
            camera_thread.join(timeout=2.0)
        cleanup_camera()


app = FastAPI(title="Optical Fate Dice Reader Module", lifespan=lifespan)
atexit.register(cleanup_camera)
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)


def status_frame_bytes(message: str) -> bytes:
    frame = np.full((360, 640, 3), (25, 31, 38), dtype=np.uint8)
    cv2.putText(frame, "Camera Feed Unavailable", (42, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
    cv2.putText(frame, message, (42, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 190, 255), 2)
    success, encoded = cv2.imencode(".jpg", frame)
    return encoded.tobytes() if success else b""


def mjpeg_frame_generator(stage: Literal["raw", "motion_mask", "edges", "thresholded"]):
    previous_raw = None
    while True:
        with latest_frame_lock:
            frame_bytes = latest_jpeg_frame
            raw = None if latest_raw_frame is None else latest_raw_frame.copy()

        if raw is None:
            if frame_bytes is None:
                placeholder_message = (
                    runtime_state.system.message if runtime_state.system.status == "ERROR" else "Waiting for camera frames"
                )
                placeholder = status_frame_bytes(placeholder_message)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
                )
            else:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.1)
            continue

        if stage == "raw":
            display = raw
        elif stage == "motion_mask":
            if previous_raw is None:
                mask = vision.motion_mask_from_pair(raw, raw)
            else:
                mask = vision.motion_mask_from_pair(previous_raw, raw)
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif stage == "edges":
            edges = vision.edges_view(raw)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            display = raw.copy()
            display[:, :] = 0
            if vision.config.roi is not None:
                roi_x, roi_y, roi_w, roi_h = vision.config.roi
                display[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = edges_bgr
            else:
                display = edges_bgr
        elif stage == "thresholded":
            thresholded = vision.thresholded_view(raw)
            thresholded_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
            display = raw.copy()
            display[:, :] = 0
            if vision.config.roi is not None:
                roi_x, roi_y, roi_w, roi_h = vision.config.roi
                display[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = thresholded_bgr
            else:
                display = thresholded_bgr
        else:
            display = raw

        previous_raw = raw
        success, encoded = cv2.imencode(".jpg", display)
        if not success:
            time.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        )
        time.sleep(0.03)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open("templates/index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


@app.get("/app.js")
async def app_js() -> FileResponse:
    return FileResponse("templates/app.js", media_type="application/javascript")


@app.get("/api/status", response_model=SystemState)
async def get_status() -> SystemState:
    return runtime_state.system


@app.get("/api/calibration_quality", response_model=CalibrationQuality)
async def get_calibration_quality() -> CalibrationQuality:
    return calculate_calibration_quality()


@app.get("/api/config", response_model=AppConfig)
async def get_config() -> AppConfig:
    return app_config


@app.get("/api/cameras", response_model=list[CameraOption])
def list_cameras() -> list[CameraOption]:
    options: list[CameraOption] = []
    for index in range(5):
        probe = open_camera_device(index)
        available = probe is not None
        if probe is not None:
            probe.release()
        options.append(CameraOption(index=index, available=available))
    return options


@app.post("/api/camera", response_model=AppConfig)
def update_camera(selection: CameraSelectRequest) -> AppConfig:
    selected_index = selection.camera_index
    if not switch_camera(selected_index):
        log_event(logging.WARNING, f"Requested camera index {selected_index} is unavailable")
        raise HTTPException(status_code=400, detail=f"Camera index {selected_index} is unavailable.")

    app_config.camera_index = selected_index
    apply_config_to_runtime()
    write_config()
    set_state(status="IDLE", message=f"Camera switched to index {selected_index}", active_dice_count=0)
    log_event(logging.INFO, f"Camera switched to index {selected_index}")
    return app_config


@app.get("/api/roi", response_model=RoiConfig)
async def get_roi() -> RoiConfig:
    return RoiConfig(roi=app_config.roi)


@app.post("/api/roi", response_model=RoiConfig)
async def update_roi(updated: RoiConfig) -> RoiConfig:
    app_config.roi = updated.roi
    apply_config_to_runtime()
    write_config()
    return RoiConfig(roi=app_config.roi)


@app.get("/api/calibration_profiles", response_model=dict[str, CalibrationProfile])
async def get_calibration_profiles() -> dict[str, CalibrationProfile]:
    return app_config.calibration_profiles


@app.post("/api/calibration_profiles", response_model=CalibrationProfile)
async def save_calibration_profile(request: CalibrationProfileRequest) -> CalibrationProfile:
    profile = CalibrationProfile(roi=app_config.roi, vision=app_config.vision.model_copy(deep=True))
    app_config.calibration_profiles[request.name] = profile
    write_config()
    log_event(logging.INFO, f"Calibration profile saved: {request.name}")
    return profile


@app.post("/api/calibration_profiles/apply", response_model=AppConfig)
async def apply_calibration_profile(request: CalibrationProfileRequest) -> AppConfig:
    profile = app_config.calibration_profiles.get(request.name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Calibration profile '{request.name}' not found.")

    app_config.roi = profile.roi
    app_config.vision = profile.vision.model_copy(deep=True)
    apply_config_to_runtime()
    write_config()
    log_event(logging.INFO, f"Calibration profile applied: {request.name}")
    return app_config


@app.get("/api/history", response_model=list[RollHistoryItem])
async def get_history() -> list[RollHistoryItem]:
    return list(history)


@app.get("/api/history/{snapshot_id}/snapshot.jpg")
async def get_history_snapshot(snapshot_id: str) -> StreamingResponse:
    snapshot = roll_snapshots.get(snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return StreamingResponse(io.BytesIO(snapshot), media_type="image/jpeg")


@app.get("/api/export_diagnostics")
async def export_diagnostics() -> StreamingResponse:
    snapshot = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": runtime_state.system.model_dump(),
        "active_request_id": active_request_id,
        "history_count": len(history),
    }

    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        if config_path.exists():
            zip_file.write(config_path, arcname="config.json")
        else:
            zip_file.writestr("config.json", "{}")

        log_path = Path("app.log")
        if log_path.exists():
            zip_file.write(log_path, arcname="app.log")
        else:
            zip_file.writestr("app.log", "")

        zip_file.writestr("state_snapshot.json", json.dumps(snapshot, indent=2))

    memory_zip.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="diagnostics.zip"'}
    return StreamingResponse(memory_zip, media_type="application/zip", headers=headers)


@app.post("/api/config", response_model=AppConfig)
async def update_config(updated: AppConfig) -> AppConfig:
    global app_config
    app_config = updated
    apply_config_to_runtime()
    write_config()
    return app_config


@app.post("/api/config/vision", response_model=VisionTuningConfig)
async def update_vision_config(updated: VisionTuningConfig) -> VisionTuningConfig:
    app_config.vision = updated
    apply_config_to_runtime()
    write_config()
    return app_config.vision


@app.get("/video_feed")
async def video_feed(
    stage: Literal["raw", "motion_mask", "edges", "thresholded"] = Query(default="raw"),
) -> StreamingResponse:
    return StreamingResponse(
        mjpeg_frame_generator(stage),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/fallback_roll")
async def fallback_roll() -> dict:
    global active_request_id, error_reset_task, watch_timeout_task
    if game_bridge_socket is None:
        raise HTTPException(status_code=409, detail="No active game bridge WebSocket connection.")

    cancel_task(watch_timeout_task)
    watch_timeout_task = None
    cancel_task(error_reset_task)
    error_reset_task = None

    set_state(
        status="CALCULATING",
        message="Generating fallback roll",
        active_dice_count=runtime_state.expected_dice_count,
    )
    fallback_payload = generate_fallback_roll(runtime_state.expected_dice_count)
    fallback_payload.fallback_reason = "MANUAL_FALLBACK_TRIGGERED"

    await broadcast_event("ROLL_COMPLETE", fallback_payload.model_dump(), request_id=active_request_id)
    append_history(fallback_payload, active_request_id, snapshot_bytes=None)
    log_event(logging.INFO, f"Fallback roll complete with score={fallback_payload.total_score}", request_id=active_request_id)

    set_state(status="IDLE", message="Fallback roll complete", active_dice_count=0)
    active_request_id = None
    update_debug_overlay(results=fallback_payload.model_dump()["dice"], motion_pixels=0)
    vision.reset_motion_history()
    return {"ok": True, "event": "ROLL_COMPLETE", "data": fallback_payload.model_dump()}


@app.websocket("/ws/game-bridge")
async def game_bridge(websocket: WebSocket) -> None:
    global active_request_id, error_reset_task, game_bridge_socket, heartbeat_task, last_heartbeat_ts, watch_timeout_task
    await websocket.accept()
    game_bridge_socket = websocket
    last_heartbeat_ts = time.time()
    cancel_task(heartbeat_task)
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    log_event(logging.INFO, f"Game bridge connected: {websocket.client}")

    try:
        while True:
            message = await websocket.receive_text()
            try:
                parsed = GameCommand.model_validate(json.loads(message))
                log_event(logging.INFO, "Message received on /ws/game-bridge", request_id=parsed.request_id)
            except (json.JSONDecodeError, ValueError) as error:
                log_event(logging.WARNING, f"Invalid game command: {error}")
                await websocket.send_json({"event": "ROLL_ERROR", "data": {"reason": "INVALID_COMMAND_PAYLOAD"}})
                continue

            if parsed.action == "REQUEST_ROLL":
                if not parsed.request_id:
                    await websocket.send_json(
                        {
                            "event": "ROLL_ERROR",
                            "request_id": None,
                            "data": {"reason": "MISSING_REQUEST_ID"},
                        }
                    )
                    continue

                if runtime_state.system.status in {"WATCHING", "CALCULATING"}:
                    # Safer behavior: reject overlap and preserve deterministic handling of the active request.
                    await websocket.send_json(
                        {
                            "event": "ROLL_ERROR",
                            "request_id": parsed.request_id,
                            "data": {
                                "reason": "REQUEST_REJECTED_BUSY",
                                "active_request_id": active_request_id,
                            },
                        }
                    )
                    continue

                requested_count = parsed.config.expected_dice_count if parsed.config else 3
                timeout_seconds = parsed.config.timeout_seconds if parsed.config else 30
                active_request_id = parsed.request_id
                runtime_state.expected_dice_count = requested_count

                cancel_task(watch_timeout_task)
                watch_timeout_task = None
                cancel_task(error_reset_task)
                error_reset_task = None

                set_state(
                    status="WATCHING",
                    message="Roll requested. Waiting for settlement.",
                    active_dice_count=requested_count,
                )
                update_debug_overlay(results=[], motion_pixels=0)
                vision.reset_motion_history()
                watch_timeout_task = asyncio.create_task(handle_roll_timeout(timeout_seconds))

                await websocket.send_json(
                    {
                        "event": "ROLL_ACK",
                        "request_id": parsed.request_id,
                        "data": {
                            "status": runtime_state.system.status,
                            "message": runtime_state.system.message,
                            "expected_dice_count": requested_count,
                            "timeout_seconds": timeout_seconds,
                        },
                    }
                )
            elif parsed.action == "PING":
                last_heartbeat_ts = time.time()
                await websocket.send_json({"event": "PONG", "data": {"ts": int(time.time())}})
            elif parsed.action == "PONG":
                last_heartbeat_ts = time.time()
            else:
                await websocket.send_json(
                    {
                        "event": "ROLL_ERROR",
                        "request_id": parsed.request_id,
                        "data": {"reason": f"UNSUPPORTED_ACTION_{parsed.action}"},
                    }
                )
    except WebSocketDisconnect:
        cancel_task(heartbeat_task)
        heartbeat_task = None
        cancel_task(watch_timeout_task)
        watch_timeout_task = None
        cancel_task(error_reset_task)
        error_reset_task = None
        active_request_id = None
        set_state(status="IDLE", message="Waiting for roll request", active_dice_count=0)
        update_debug_overlay(results=[], motion_pixels=0)
        vision.reset_motion_history()
        game_bridge_socket = None
        log_event(logging.INFO, f"Game bridge disconnected: {websocket.client}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
