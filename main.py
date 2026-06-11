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
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from vision import VisionConfig, VisionProcessor

try:
    import NDIlib as ndi
except ImportError:
    ndi = None

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


class BridgeCommand(BaseModel):
    type: str | None = None
    action: str | None = None
    request_id: str | None = Field(default=None, max_length=120)
    client_name: str | None = Field(default=None, max_length=80)
    config: RollConfig | None = None


class BridgeErrorPayload(BaseModel):
    reason: str
    message: str | None = None
    active_request_id: str | None = None


class RuntimeState(BaseModel):
    system: SystemState
    expected_dice_count: int = 3


class VisionTuningConfig(BaseModel):
    motion_threshold: int = Field(default=2000, ge=1)
    motion_diff_threshold: int = Field(default=45, ge=1, le=255)
    settlement_seconds: float = Field(default=1.0, ge=0.2, le=10.0)
    contour_min_area: int = Field(default=650, ge=1)
    contour_max_area: int = Field(default=28100, ge=1)
    symbol_threshold_value: int = Field(default=211, ge=0, le=255)


class CalibrationProfile(BaseModel):
    roi: list[int] | None = Field(default=None, min_length=4, max_length=4)
    vision: VisionTuningConfig = Field(default_factory=VisionTuningConfig)


class AppConfig(BaseModel):
    camera_index: int = Field(default=0, ge=0)
    camera_source_type: Literal["opencv", "ndi"] = "opencv"
    ndi_source_name: str | None = None
    force_no_signal_camera: bool = False
    roi: list[int] | None = Field(default=None, min_length=4, max_length=4)
    vision: VisionTuningConfig = Field(default_factory=VisionTuningConfig)
    calibration_profiles: dict[str, CalibrationProfile] = Field(default_factory=dict)


class CameraOption(BaseModel):
    index: int
    available: bool
    has_signal: bool = False
    mean_intensity: float | None = None
    contrast: float | None = None


class CameraSelectRequest(BaseModel):
    camera_index: int = Field(ge=0)
    force_no_signal: bool = False


class NdiSourceOption(BaseModel):
    name: str
    url_address: str | None = None


class NdiSourceSelectRequest(BaseModel):
    source_name: str = Field(min_length=1)


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


class NdiSourceNotFoundError(RuntimeError):
    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        super().__init__(f"NDI source '{source_name}' was not found.")


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


class DiceSuitabilityItem(BaseModel):
    index: int
    value: Literal["+", "-", "blank"]
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(min_length=4, max_length=4)
    color_label: str
    status: Literal["PASS", "CHECK", "REMOVE"]
    reason: str


class DiceSuitabilityReport(BaseModel):
    ok: bool
    generated_at: str
    dice_count: int
    items: list[DiceSuitabilityItem]


runtime_state = RuntimeState(
    system=SystemState(status="IDLE", message="Waiting for roll request", active_dice_count=0),
    expected_dice_count=3,
)
app_config = AppConfig()
vision = VisionProcessor(
    VisionConfig(
        motion_threshold=app_config.vision.motion_threshold,
        motion_diff_threshold=app_config.vision.motion_diff_threshold,
        settlement_seconds=app_config.vision.settlement_seconds,
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
ndi_receiver: "NdiReceiver | None" = None
camera_lock = threading.Lock()
camera_index = 0
camera_cleanup_done = False
camera_thread: threading.Thread | None = None
camera_stop_event = threading.Event()
last_camera_reconnect_attempt: float = 0.0
camera_reconnect_interval_seconds = 2.0
ndi_reconnect_pause_until: float = 0.0
ndi_reconnect_pause_seconds = 30.0

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


def ndi_recovery_message(source_name: str | None) -> str:
    if source_name:
        return f"NDI source '{source_name}' not found. Refresh sources or switch to Camera."
    return "NDI source not found. Refresh sources or switch to Camera."


def pause_ndi_reconnects(now: float | None = None) -> None:
    global ndi_reconnect_pause_until
    ndi_reconnect_pause_until = (now if now is not None else time.monotonic()) + ndi_reconnect_pause_seconds


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
    vision.config.motion_diff_threshold = app_config.vision.motion_diff_threshold
    vision.config.settlement_seconds = app_config.vision.settlement_seconds
    vision.config.contour_min_area = app_config.vision.contour_min_area
    vision.config.contour_max_area = app_config.vision.contour_max_area
    vision.config.symbol_threshold_value = app_config.vision.symbol_threshold_value
    if app_config.roi and app_config.roi[2] >= 50 and app_config.roi[3] >= 50:
        vision.config.roi = tuple(app_config.roi)
    else:
        vision.config.roi = None


class NdiReceiver:
    def __init__(self, source_name: str) -> None:
        if ndi is None:
            raise RuntimeError("NDI Python binding is not installed.")
        if not ndi.initialize():
            raise RuntimeError("NDI runtime failed to initialize.")

        source = self._find_source(source_name)
        if source is None:
            raise NdiSourceNotFoundError(source_name)

        settings = ndi.RecvCreateV3()
        settings.source_to_connect_to = source
        settings.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        settings.bandwidth = ndi.RECV_BANDWIDTH_HIGHEST
        settings.allow_video_fields = False
        self.receiver = ndi.recv_create_v3(settings)
        if self.receiver is None:
            raise RuntimeError(f"Could not create NDI receiver for '{source_name}'.")
        self.source_name = source_name

    @staticmethod
    def _find_source(source_name: str):
        finder = ndi.find_create_v2()
        if finder is None:
            return None
        try:
            ndi.find_wait_for_sources(finder, 2000)
            for source in ndi.find_get_current_sources(finder) or []:
                if source.ndi_name == source_name:
                    return source
        finally:
            ndi.find_destroy(finder)
        return None

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame_type, video_frame, _audio_frame, _metadata_frame = ndi.recv_capture_v2(
            self.receiver,
            1000,
            True,
            False,
            False,
        )
        if frame_type != ndi.FRAME_TYPE_VIDEO:
            return False, None

        try:
            frame_data = np.asarray(video_frame.data, dtype=np.uint8)
            if frame_data.size == 0 or video_frame.xres <= 0 or video_frame.yres <= 0:
                return False, None

            row_stride = int(video_frame.line_stride_in_bytes)
            width_bytes = int(video_frame.xres) * 4
            if frame_data.ndim == 1:
                bgra = frame_data.reshape((int(video_frame.yres), row_stride))[:, :width_bytes]
                bgra = bgra.reshape((int(video_frame.yres), int(video_frame.xres), 4))
            else:
                bgra = frame_data.reshape((int(video_frame.yres), int(video_frame.xres), 4))
            return True, cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        finally:
            ndi.recv_free_video_v2(self.receiver, video_frame)

    def release(self) -> None:
        if self.receiver is not None:
            ndi.recv_destroy(self.receiver)
            self.receiver = None


def list_ndi_sources(timeout_ms: int = 2000) -> list[NdiSourceOption]:
    if ndi is None or not ndi.initialize():
        return []
    finder = ndi.find_create_v2()
    if finder is None:
        return []
    try:
        ndi.find_wait_for_sources(finder, timeout_ms)
        return [
            NdiSourceOption(name=source.ndi_name, url_address=source.url_address or None)
            for source in (ndi.find_get_current_sources(finder) or [])
        ]
    finally:
        ndi.find_destroy(finder)


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


def draw_debug_overlay(frame, results: list[dict] | None = None):
    with debug_overlay_lock:
        current_results = list(debug_dice_results) if results is None else list(results)
        motion_value = latest_motion_pixels

    cv2.putText(
        frame,
        f"State: {runtime_state.system.status} | Motion: {motion_value} | Stable: {vision.settlement_progress_seconds:.1f}/{vision.config.settlement_seconds:.1f}s",
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


def estimate_die_color(frame: np.ndarray, bbox: list[int]) -> str:
    x, y, w, h = bbox
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = float(np.median(hsv[:, :, 0]))
    saturation = float(np.median(hsv[:, :, 1]))
    value = float(np.median(hsv[:, :, 2]))
    if value < 45:
        return "black"
    if saturation < 45:
        return "white/gray"
    if hue < 10 or hue >= 170:
        return "red/pink"
    if hue < 25:
        return "orange"
    if hue < 40:
        return "yellow/gold"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue/teal"
    return "purple"


def estimate_die_color_strength(frame: np.ndarray, bbox: list[int]) -> float:
    x, y, w, h = bbox
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0 or w <= 0 or h <= 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturated_visible = cv2.bitwise_and(cv2.inRange(saturation, 90, 255), cv2.inRange(value, 45, 255))
    return float(cv2.countNonZero(saturated_visible)) / float(w * h)


def build_suitability_report(frame: np.ndarray) -> DiceSuitabilityReport:
    results = vision.calculate_roll(frame)
    areas = [item["bbox"][2] * item["bbox"][3] for item in results]
    median_area = float(np.median(areas)) if areas else 0.0
    report_items: list[DiceSuitabilityItem] = []

    for index, item in enumerate(results, start=1):
        bbox = [int(value) for value in item["bbox"]]
        confidence = float(item["confidence"])
        color_label = estimate_die_color(frame, bbox)
        color_strength = estimate_die_color_strength(frame, bbox)
        area = bbox[2] * bbox[3]
        reasons: list[str] = []
        status: Literal["PASS", "CHECK", "REMOVE"] = "PASS"

        color_visible_blank = (
            item["value"] == "blank"
            and color_label in {"red/pink", "orange", "yellow/gold", "green", "blue/teal", "purple"}
            and color_strength >= 0.40
            and confidence >= 0.55
        )

        if confidence < 0.58 and not color_visible_blank:
            status = "CHECK"
            reasons.append("low confidence")
        if color_visible_blank:
            reasons.append("colored blank face is clearly separated from tray")
        if median_area > 0 and area > median_area * 2.2:
            status = "CHECK"
            reasons.append("oversized box, likely overlap")
        if median_area > 0 and area < median_area * 0.45:
            status = "CHECK"
            reasons.append("small box, symbol may be weak")

        if confidence < 0.45:
            status = "REMOVE"
            reasons.append("below usable confidence")

        report_items.append(
            DiceSuitabilityItem(
                index=index,
                value=item["value"],
                confidence=confidence,
                bbox=bbox,
                color_label=color_label,
                status=status,
                reason=", ".join(reasons) if reasons else "usable",
            )
        )

    return DiceSuitabilityReport(
        ok=True,
        generated_at=datetime.now(timezone.utc).isoformat(),
        dice_count=len(report_items),
        items=report_items,
    )


async def broadcast_event(event_name: str, payload: dict, request_id: str | None = None) -> None:
    if game_bridge_socket is None:
        return

    try:
        message_type = {
            "PING": "heartbeat",
            "ROLL_ACK": "roll.ack",
            "ROLL_COMPLETE": "roll.result",
            "ROLL_ERROR": "error",
        }.get(event_name, event_name.lower())
        message = {"type": message_type, "event": event_name, "data": payload}
        if request_id is not None:
            message["request_id"] = request_id
        await game_bridge_socket.send_json(message)
    except Exception as error:  # pragma: no cover - best effort socket send
        log_event(logging.WARNING, f"Broadcast failed for {event_name}: {error}", request_id=request_id)


async def send_bridge_message(
    websocket: WebSocket,
    message_type: str,
    payload: dict,
    request_id: str | None = None,
    event_name: str | None = None,
) -> None:
    message = {"type": message_type, "data": payload}
    if event_name is not None:
        message["event"] = event_name
    if request_id is not None:
        message["request_id"] = request_id
    await websocket.send_json(message)


async def send_bridge_error(
    websocket: WebSocket,
    reason: str,
    request_id: str | None = None,
    message: str | None = None,
    active_id: str | None = None,
) -> None:
    payload = BridgeErrorPayload(reason=reason, message=message, active_request_id=active_id)
    await send_bridge_message(
        websocket,
        "error",
        payload.model_dump(exclude_none=True),
        request_id=request_id,
        event_name="ROLL_ERROR",
    )


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
        if app_config.camera_source_type == "ndi":
            camera_ok = ndi_receiver is not None
        else:
            current_cap = cap
            if current_cap is None or not current_cap.isOpened():
                camera_ok = False

    if not camera_ok:
        logger.critical("Startup diagnostic: video source open failed.")
        message = (
            ndi_recovery_message(app_config.ndi_source_name)
            if app_config.camera_source_type == "ndi"
            else "Camera Not Found"
        )
        set_state(status="ERROR", message=message, active_dice_count=0)
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


def decode_image_bytes(image_bytes: bytes):
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")
    return frame


def build_roll_payload_from_dice(parsed_results: list[dict]) -> RollPayload:
    dice = [DieResult.model_validate(item) for item in parsed_results]
    symbol_to_score = {"+": 1, "-": -1, "blank": 0}
    total_score = sum(symbol_to_score[die.value] for die in dice)
    roll_confidence = None
    if dice:
        roll_confidence = round(sum(die.confidence for die in dice) / len(dice), 3)

    return RollPayload(
        total_score=total_score,
        dice=dice,
        is_fallback=False,
        roll_confidence=roll_confidence,
        fallback_reason=None,
    )


def clear_active_roll_request() -> None:
    global active_request_id, error_reset_task, watch_timeout_task
    cancel_task(watch_timeout_task)
    watch_timeout_task = None
    cancel_task(error_reset_task)
    error_reset_task = None
    active_request_id = None


async def reset_to_idle_after_delay(delay_seconds: float) -> None:
    await asyncio.sleep(delay_seconds)
    if runtime_state.system.status == "ERROR":
        set_state(status="IDLE", message="Waiting for roll request", active_dice_count=0)
        update_debug_overlay(results=[], motion_pixels=0)
        logger.info("State auto-reset to IDLE after timeout error.")


async def handle_roll_timeout(timeout_seconds: int) -> None:
    global active_request_id, error_reset_task, watch_timeout_task
    try:
        await asyncio.sleep(timeout_seconds)
        if runtime_state.system.status == "WATCHING":
            set_state(
                status="ERROR",
                message="Timeout reached: no settlement detected.",
                active_dice_count=runtime_state.expected_dice_count,
            )
            request_id = active_request_id
            log_event(logging.WARNING, f"Roll request timed out after {timeout_seconds} seconds", request_id=request_id)
            await broadcast_event(
                "ROLL_ERROR",
                {"reason": "TIMEOUT_REACHED_NO_SETTLEMENT"},
                request_id=request_id,
            )
            active_request_id = None
            watch_timeout_task = None
            cancel_task(error_reset_task)
            error_reset_task = asyncio.create_task(reset_to_idle_after_delay(2.0))
    except asyncio.CancelledError:
        logger.info("Roll timeout task cancelled.")
        raise


def process_settled_frame(frame) -> None:
    global active_request_id

    set_state(
        status="CALCULATING",
        message="Dice settled. Calculating roll.",
        active_dice_count=runtime_state.expected_dice_count,
    )

    parsed_results = vision.calculate_roll(frame)
    update_debug_overlay(results=parsed_results)
    expected_count = runtime_state.expected_dice_count

    if len(parsed_results) != expected_count:
        request_id = active_request_id
        set_state(
            status="ERROR",
            message=f"Detected {len(parsed_results)} dice, expected {expected_count}.",
            active_dice_count=len(parsed_results),
        )
        schedule_coroutine(
            broadcast_event("ROLL_ERROR", {"reason": "DICE_COUNT_MISMATCH"}, request_id=request_id)
        )
        schedule_coroutine(reset_to_idle_after_delay(2.0))
        clear_active_roll_request()
        vision.reset_motion_history()
        return

    payload = build_roll_payload_from_dice(parsed_results)
    request_id = active_request_id
    schedule_coroutine(broadcast_event("ROLL_COMPLETE", payload.model_dump(), request_id=request_id))
    snapshot_bytes = None
    ok, encoded = cv2.imencode(".jpg", draw_debug_overlay(frame.copy(), parsed_results))
    if ok:
        snapshot_bytes = encoded.tobytes()
    append_history(payload, request_id, snapshot_bytes=snapshot_bytes)
    log_event(logging.INFO, f"Roll complete with score={payload.total_score}", request_id=request_id)

    clear_active_roll_request()
    set_state(status="IDLE", message="Roll complete", active_dice_count=0)
    vision.reset_motion_history()


def camera_worker_loop() -> None:
    global last_camera_reconnect_attempt, latest_jpeg_frame, latest_raw_frame
    frame_interval = 1.0 / vision.config.processing_fps
    frame_failure_started: float | None = None

    while not camera_stop_event.is_set():
        with camera_lock:
            if app_config.camera_source_type == "ndi":
                current_ndi_receiver = ndi_receiver
                if current_ndi_receiver is None:
                    frame = None
                else:
                    ok, frame = current_ndi_receiver.read()
                    if not ok:
                        frame = None
            else:
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
                    message = (
                        ndi_recovery_message(app_config.ndi_source_name)
                        if app_config.camera_source_type == "ndi"
                        else "Camera Not Providing Frames"
                    )
                    set_state(status="ERROR", message=message, active_dice_count=0)
            ndi_retry_paused = app_config.camera_source_type == "ndi" and ndi_reconnect_pause_until > now
            if not ndi_retry_paused and now - last_camera_reconnect_attempt >= camera_reconnect_interval_seconds:
                last_camera_reconnect_attempt = now
                if reconnect_camera():
                    source_label = (
                        f"NDI source {app_config.ndi_source_name}"
                        if app_config.camera_source_type == "ndi"
                        else f"camera index {camera_index}"
                    )
                    log_event(logging.INFO, f"Camera reconnected on {source_label}")
                elif runtime_state.system.status != "ERROR":
                    message = (
                        ndi_recovery_message(app_config.ndi_source_name)
                        if app_config.camera_source_type == "ndi"
                        else "Camera Not Found"
                    )
                    set_state(status="ERROR", message=message, active_dice_count=0)
            time.sleep(0.05)
            continue

        frame_failure_started = None
        if runtime_state.system.status == "ERROR" and (
            runtime_state.system.message in {
                "Camera Not Found",
                "Camera Not Providing Frames",
            }
            or (
                app_config.camera_source_type == "ndi"
                and (runtime_state.system.message or "").startswith("NDI source ")
            )
        ):
            set_state(status="IDLE", message="Camera reconnected", active_dice_count=0)

        with latest_frame_lock:
            latest_raw_frame = frame.copy()

        if runtime_state.system.status == "WATCHING":
            motion_pixels = vision.detect_motion(frame)
            update_debug_overlay(motion_pixels=motion_pixels)
            has_settled = vision.update_settlement(motion_pixels, time.monotonic())
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
    global cap, ndi_receiver, camera_cleanup_done
    with camera_lock:
        if camera_cleanup_done:
            return
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("Camera released.")
        cap = None
        if ndi_receiver is not None:
            ndi_receiver.release()
            ndi_receiver = None
            logger.info("NDI receiver released.")
        camera_cleanup_done = True


def frame_signal_metrics(frame: np.ndarray | None) -> tuple[bool, float | None, float | None]:
    if frame is None or frame.size == 0:
        return False, None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    contrast = float(np.std(gray))
    return mean_intensity > 8.0 and contrast > 3.0, mean_intensity, contrast


def open_camera_device(index: int, require_signal: bool = True) -> cv2.VideoCapture | None:
    backend_preferences = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backend_preferences:
        candidate = cv2.VideoCapture(index, backend)
        if candidate is None or not candidate.isOpened():
            if candidate is not None:
                candidate.release()
            continue

        ok, first_frame = candidate.read()
        has_signal, _mean_intensity, _contrast = frame_signal_metrics(first_frame if ok else None)
        if has_signal or (ok and not require_signal):
            return candidate

        candidate.release()
    return None


def probe_camera_option(index: int) -> CameraOption:
    candidate = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if candidate is None or not candidate.isOpened():
        if candidate is not None:
            candidate.release()
        return CameraOption(index=index, available=False)

    ok, frame = candidate.read()
    candidate.release()
    has_signal, mean_intensity, contrast = frame_signal_metrics(frame if ok else None)
    return CameraOption(
        index=index,
        available=bool(ok),
        has_signal=has_signal,
        mean_intensity=mean_intensity,
        contrast=contrast,
    )


def switch_camera(index: int, require_signal: bool = True) -> bool:
    global cap, ndi_receiver
    with camera_lock:
        previous = cap
        replacement = open_camera_device(index, require_signal=require_signal)
        if replacement is None:
            return False

        cap = replacement
        if previous is not None and previous.isOpened():
            previous.release()
        if ndi_receiver is not None:
            ndi_receiver.release()
            ndi_receiver = None
    return True


def switch_ndi_source(source_name: str) -> bool:
    global cap, ndi_receiver, ndi_reconnect_pause_until
    with camera_lock:
        try:
            replacement = NdiReceiver(source_name)
        except RuntimeError as error:
            log_event(logging.WARNING, f"NDI source switch failed: {error}")
            return False

        previous_ndi_receiver = ndi_receiver
        previous_cap = cap
        ndi_receiver = replacement
        ndi_reconnect_pause_until = 0.0
        cap = None
        if previous_ndi_receiver is not None:
            previous_ndi_receiver.release()
        if previous_cap is not None and previous_cap.isOpened():
            previous_cap.release()
    return True


def reconnect_camera() -> bool:
    global cap, ndi_receiver, ndi_reconnect_pause_until
    with camera_lock:
        if app_config.camera_source_type == "ndi":
            if not app_config.ndi_source_name:
                return False
            if ndi_receiver is not None:
                ndi_receiver.release()
            try:
                ndi_receiver = NdiReceiver(app_config.ndi_source_name)
                ndi_reconnect_pause_until = 0.0
            except NdiSourceNotFoundError as error:
                log_event(logging.WARNING, f"NDI reconnect paused: {error}")
                ndi_receiver = None
                pause_ndi_reconnects()
                return False
            except RuntimeError as error:
                log_event(logging.WARNING, f"NDI reconnect failed: {error}")
                ndi_receiver = None
                return False
            return True

        previous = cap
        if previous is not None:
            previous.release()
        cap = open_camera_device(camera_index, require_signal=not app_config.force_no_signal_camera)
        return cap is not None


def handle_exit_signal(signum: int, _frame: object) -> None:
    logger.info("Received shutdown signal %s. Releasing camera.", signum)
    camera_stop_event.set()
    cleanup_camera()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global camera_cleanup_done, camera_thread, cap, ndi_receiver, last_camera_reconnect_attempt, main_event_loop, ndi_reconnect_pause_until

    main_event_loop = asyncio.get_running_loop()
    camera_cleanup_done = False
    camera_stop_event.clear()
    last_camera_reconnect_attempt = 0.0
    ndi_reconnect_pause_until = 0.0
    config_ok = True

    try:
        load_or_create_config()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        logger.error("Config load failed during startup: %s", error)
        set_state(status="ERROR", message="Config Not Readable", active_dice_count=0)
        config_ok = False

    with camera_lock:
        if app_config.camera_source_type == "ndi" and app_config.ndi_source_name:
            try:
                ndi_receiver = NdiReceiver(app_config.ndi_source_name)
                logger.info("NDI source initialized: %s.", app_config.ndi_source_name)
            except RuntimeError as error:
                ndi_receiver = None
                logger.error("Could not open NDI source '%s': %s", app_config.ndi_source_name, error)
        else:
            cap = open_camera_device(camera_index, require_signal=not app_config.force_no_signal_camera)
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


def mjpeg_frame_generator(stage: Literal["raw", "motion_mask", "edges", "contours", "thresholded"]):
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
        elif stage == "contours":
            display = vision.contour_debug_view(raw)
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


@app.get("/api/dice_suitability", response_model=DiceSuitabilityReport)
async def get_dice_suitability() -> DiceSuitabilityReport:
    with latest_frame_lock:
        frame = None if latest_raw_frame is None else latest_raw_frame.copy()
    if frame is None:
        raise HTTPException(status_code=503, detail="No camera frame is available.")
    return build_suitability_report(frame)


@app.get("/api/config", response_model=AppConfig)
async def get_config() -> AppConfig:
    return app_config


@app.get("/api/cameras", response_model=list[CameraOption])
def list_cameras() -> list[CameraOption]:
    options: list[CameraOption] = []
    for index in range(5):
        options.append(probe_camera_option(index))
    return options


@app.post("/api/camera", response_model=AppConfig)
def update_camera(selection: CameraSelectRequest) -> AppConfig:
    selected_index = selection.camera_index
    if not switch_camera(selected_index, require_signal=not selection.force_no_signal):
        log_event(logging.WARNING, f"Requested camera index {selected_index} is unavailable")
        raise HTTPException(status_code=400, detail=f"Camera index {selected_index} is unavailable.")

    app_config.camera_index = selected_index
    app_config.camera_source_type = "opencv"
    app_config.ndi_source_name = None
    app_config.force_no_signal_camera = selection.force_no_signal
    apply_config_to_runtime()
    write_config()
    message = f"Camera switched to index {selected_index}"
    if selection.force_no_signal:
        message = f"{message} with no-signal override"
    set_state(status="IDLE", message=message, active_dice_count=0)
    log_event(logging.INFO, message)
    return app_config


@app.get("/api/ndi/sources", response_model=list[NdiSourceOption])
def get_ndi_sources() -> list[NdiSourceOption]:
    return list_ndi_sources()


@app.post("/api/ndi/source", response_model=AppConfig)
def update_ndi_source(selection: NdiSourceSelectRequest) -> AppConfig:
    if not switch_ndi_source(selection.source_name):
        raise HTTPException(status_code=400, detail=f"NDI source '{selection.source_name}' is unavailable.")

    app_config.camera_source_type = "ndi"
    app_config.ndi_source_name = selection.source_name
    apply_config_to_runtime()
    write_config()
    set_state(status="IDLE", message=f"NDI source selected: {selection.source_name}", active_dice_count=0)
    log_event(logging.INFO, f"NDI source selected: {selection.source_name}")
    return app_config


@app.get("/api/roi", response_model=RoiConfig)
async def get_roi() -> RoiConfig:
    return RoiConfig(roi=app_config.roi)


@app.post("/api/roi", response_model=RoiConfig)
async def update_roi(updated: RoiConfig) -> RoiConfig:
    if updated.roi is not None and (updated.roi[2] < 50 or updated.roi[3] < 50):
        raise HTTPException(status_code=400, detail="ROI must be at least 50x50 pixels.")
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


@app.post("/api/still_image_roll", response_model=RollPayload)
async def still_image_roll(request: Request) -> RollPayload:
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload one image file.")

    frame = decode_image_bytes(await request.body())
    parsed_results = vision.calculate_roll_from_still_image(frame)
    payload = build_roll_payload_from_dice(parsed_results)

    snapshot_bytes = None
    ok, encoded = cv2.imencode(".jpg", draw_debug_overlay(frame.copy(), parsed_results))
    if ok:
        snapshot_bytes = encoded.tobytes()
    append_history(payload, request_id="still-image-upload", snapshot_bytes=snapshot_bytes)
    log_event(logging.INFO, f"Still image roll complete with score={payload.total_score}")
    return payload


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
    stage: Literal["raw", "motion_mask", "edges", "contours", "thresholded"] = Query(default="raw"),
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
    if active_request_id is None or runtime_state.system.status not in {"WATCHING", "CALCULATING"}:
        raise HTTPException(status_code=409, detail="No active roll request is waiting for fallback.")

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
    if game_bridge_socket is not None:
        log_event(logging.WARNING, f"Rejected duplicate game bridge connection: {websocket.client}")
        await send_bridge_error(websocket, "BRIDGE_ALREADY_CONNECTED", message="Only one game bridge WebSocket connection is allowed.")
        await websocket.close(code=1008)
        return

    game_bridge_socket = websocket
    last_heartbeat_ts = time.time()
    cancel_task(heartbeat_task)
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    log_event(logging.INFO, f"Game bridge connected: {websocket.client}")
    await send_bridge_message(
        websocket,
        "connect.ok",
        {
            "protocol_version": 1,
            "message_types": [
                "connect",
                "ping",
                "pong",
                "config.update",
                "roll.request",
                "roll.ack",
                "roll.result",
                "heartbeat",
                "error",
            ],
            "status": runtime_state.system.status,
            "expected_dice_count": runtime_state.expected_dice_count,
        },
    )

    try:
        while True:
            message = await websocket.receive_text()
            try:
                parsed = BridgeCommand.model_validate(json.loads(message))
                log_event(logging.INFO, "Message received on /ws/game-bridge", request_id=parsed.request_id)
            except (json.JSONDecodeError, ValueError) as error:
                log_event(logging.WARNING, f"Invalid game command: {error}")
                await send_bridge_error(websocket, "INVALID_COMMAND_PAYLOAD", message=str(error))
                continue

            message_type = parsed.type or parsed.action
            legacy_action_map = {
                "REQUEST_ROLL": "roll.request",
                "PING": "ping",
                "PONG": "pong",
            }
            message_type = legacy_action_map.get(message_type or "", message_type)

            if message_type == "connect":
                last_heartbeat_ts = time.time()
                await send_bridge_message(
                    websocket,
                    "connect.ok",
                    {
                        "protocol_version": 1,
                        "client_name": parsed.client_name,
                        "status": runtime_state.system.status,
                        "expected_dice_count": runtime_state.expected_dice_count,
                    },
                    request_id=parsed.request_id,
                )
            elif message_type == "config.update":
                config = parsed.config or RollConfig()
                runtime_state.expected_dice_count = config.expected_dice_count
                await send_bridge_message(
                    websocket,
                    "config.updated",
                    {
                        "expected_dice_count": config.expected_dice_count,
                        "timeout_seconds": config.timeout_seconds,
                    },
                    request_id=parsed.request_id,
                )
            elif message_type == "roll.request":
                if not parsed.request_id:
                    await send_bridge_error(websocket, "MISSING_REQUEST_ID")
                    continue

                if runtime_state.system.status in {"WATCHING", "CALCULATING"}:
                    # Safer behavior: reject overlap and preserve deterministic handling of the active request.
                    await send_bridge_error(
                        websocket,
                        "REQUEST_REJECTED_BUSY",
                        request_id=parsed.request_id,
                        active_id=active_request_id,
                    )
                    continue

                config = parsed.config or RollConfig(expected_dice_count=runtime_state.expected_dice_count)
                requested_count = config.expected_dice_count
                timeout_seconds = config.timeout_seconds
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

                await send_bridge_message(
                    websocket,
                    "roll.ack",
                    {
                        "status": runtime_state.system.status,
                        "message": runtime_state.system.message,
                        "expected_dice_count": requested_count,
                        "timeout_seconds": timeout_seconds,
                    },
                    request_id=parsed.request_id,
                    event_name="ROLL_ACK",
                )
            elif message_type == "ping":
                last_heartbeat_ts = time.time()
                await send_bridge_message(
                    websocket,
                    "pong",
                    {"ts": int(time.time())},
                    request_id=parsed.request_id,
                    event_name="PONG",
                )
            elif message_type == "pong":
                last_heartbeat_ts = time.time()
            else:
                await send_bridge_error(
                    websocket,
                    f"UNSUPPORTED_MESSAGE_TYPE_{message_type}",
                    request_id=parsed.request_id,
                )
    except WebSocketDisconnect:
        if websocket is game_bridge_socket:
            cancel_task(heartbeat_task)
            heartbeat_task = None
            clear_active_roll_request()
            set_state(status="IDLE", message="Waiting for roll request", active_dice_count=0)
            update_debug_overlay(results=[], motion_pixels=0)
            vision.reset_motion_history()
            game_bridge_socket = None
        log_event(logging.INFO, f"Game bridge disconnected: {websocket.client}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
