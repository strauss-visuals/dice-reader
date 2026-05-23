from __future__ import annotations

import atexit
import asyncio
import json
import logging
import secrets
import signal
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from vision import VisionConfig, VisionProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dice_reader")


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


class SystemState(BaseModel):
    status: Literal["IDLE", "WATCHING", "CALCULATING", "ERROR"]
    message: str | None = None
    active_dice_count: int = 0


class GameCommand(BaseModel):
    action: str
    config: RollConfig | None = None


class RuntimeState(BaseModel):
    system: SystemState
    expected_dice_count: int = 3


runtime_state = RuntimeState(
    system=SystemState(status="IDLE", message="Waiting for roll request", active_dice_count=0),
    expected_dice_count=3,
)
vision = VisionProcessor(VisionConfig())
game_bridge_socket: WebSocket | None = None
watch_timeout_task: asyncio.Task | None = None
error_reset_task: asyncio.Task | None = None
main_event_loop: asyncio.AbstractEventLoop | None = None

cap: cv2.VideoCapture | None = None
camera_lock = threading.Lock()
camera_index = 0
camera_cleanup_done = False
camera_thread: threading.Thread | None = None
camera_stop_event = threading.Event()

latest_frame_lock = threading.Lock()
latest_jpeg_frame: bytes | None = None
debug_overlay_lock = threading.Lock()
debug_dice_results: list[dict] = []
latest_motion_pixels: int = 0


def set_state(status: Literal["IDLE", "WATCHING", "CALCULATING", "ERROR"], message: str, active_dice_count: int) -> None:
    runtime_state.system.status = status
    runtime_state.system.message = message
    runtime_state.system.active_dice_count = active_dice_count


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

    for item in current_results:
        x, y, w, h = item["bbox"]
        label = f'{item["value"]} ({item["confidence"]:.2f})'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return frame


async def broadcast_event(event_name: str, payload: dict) -> None:
    if game_bridge_socket is None:
        return

    try:
        await game_bridge_socket.send_json({"event": event_name, "data": payload})
    except Exception as error:  # pragma: no cover - best effort socket send
        logger.warning("Failed to broadcast %s: %s", event_name, error)


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
            logger.warning("Roll request timed out after %s seconds.", timeout_seconds)
            await broadcast_event("ROLL_ERROR", {"reason": "TIMEOUT_REACHED_NO_SETTLEMENT"})
            cancel_task(error_reset_task)
            error_reset_task = asyncio.create_task(reset_to_idle_after_delay(2.0))
    except asyncio.CancelledError:
        logger.info("Roll timeout task cancelled.")
        raise


def process_settled_frame(frame) -> None:
    global watch_timeout_task

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
        schedule_coroutine(broadcast_event("ROLL_ERROR", {"reason": "DICE_COUNT_MISMATCH"}))
        schedule_coroutine(reset_to_idle_after_delay(2.0))
        vision.reset_motion_history()
        return

    dice = [DieResult.model_validate(item) for item in parsed_results]
    symbol_to_score = {"+": 1, "-": -1, "blank": 0}
    total_score = sum(symbol_to_score[die.value] for die in dice)

    payload = RollPayload(total_score=total_score, dice=dice, is_fallback=False)
    schedule_coroutine(broadcast_event("ROLL_COMPLETE", payload.model_dump()))

    cancel_task(watch_timeout_task)
    watch_timeout_task = None
    cancel_task(error_reset_task)
    error_reset_task = None

    set_state(status="IDLE", message="Roll complete", active_dice_count=0)
    vision.reset_motion_history()


def camera_worker_loop() -> None:
    frame_interval = 1.0 / vision.config.processing_fps

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
            time.sleep(0.05)
            continue

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
                global latest_jpeg_frame
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


def handle_exit_signal(signum: int, _frame: object) -> None:
    logger.info("Received shutdown signal %s. Releasing camera.", signum)
    camera_stop_event.set()
    cleanup_camera()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global camera_cleanup_done, camera_thread, cap, main_event_loop

    main_event_loop = asyncio.get_running_loop()
    camera_cleanup_done = False
    camera_stop_event.clear()

    with camera_lock:
        cap = cv2.VideoCapture(camera_index)
        if cap is None or not cap.isOpened():
            logger.error("Could not open camera at index %s.", camera_index)
            cap = None
        else:
            logger.info("Camera initialized at index %s.", camera_index)

    if cap is not None:
        camera_thread = threading.Thread(target=camera_worker_loop, name="camera-worker", daemon=True)
        camera_thread.start()

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


def mjpeg_frame_generator():
    while True:
        with latest_frame_lock:
            frame_bytes = latest_jpeg_frame

        if frame_bytes is None:
            yield (
                b"--frame\r\n"
                b"Content-Type: text/plain\r\n\r\n"
                b"Camera unavailable. Waiting for frames.\r\n"
            )
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open("templates/index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


@app.get("/api/status", response_model=SystemState)
async def get_status() -> SystemState:
    return runtime_state.system


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/fallback_roll")
async def fallback_roll() -> dict:
    global error_reset_task, watch_timeout_task
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

    await broadcast_event("ROLL_COMPLETE", fallback_payload.model_dump())

    set_state(status="IDLE", message="Fallback roll complete", active_dice_count=0)
    update_debug_overlay(results=fallback_payload.model_dump()["dice"], motion_pixels=0)
    vision.reset_motion_history()
    return {"ok": True, "event": "ROLL_COMPLETE", "data": fallback_payload.model_dump()}


@app.websocket("/ws/game-bridge")
async def game_bridge(websocket: WebSocket) -> None:
    global error_reset_task, game_bridge_socket, watch_timeout_task
    await websocket.accept()
    game_bridge_socket = websocket
    logger.info("Game bridge connected: %s", websocket.client)

    try:
        while True:
            message = await websocket.receive_text()
            logger.info("Message received on /ws/game-bridge: %s", message)
            try:
                parsed = GameCommand.model_validate(json.loads(message))
            except (json.JSONDecodeError, ValueError) as error:
                logger.warning("Invalid game command: %s", error)
                await websocket.send_json({"event": "ROLL_ERROR", "data": {"reason": "INVALID_COMMAND_PAYLOAD"}})
                continue

            if parsed.action == "REQUEST_ROLL":
                requested_count = parsed.config.expected_dice_count if parsed.config else 3
                timeout_seconds = parsed.config.timeout_seconds if parsed.config else 30
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
                        "data": {
                            "status": runtime_state.system.status,
                            "message": runtime_state.system.message,
                            "expected_dice_count": requested_count,
                            "timeout_seconds": timeout_seconds,
                        },
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "event": "ROLL_ERROR",
                        "data": {"reason": f"UNSUPPORTED_ACTION_{parsed.action}"},
                    }
                )
    except WebSocketDisconnect:
        cancel_task(watch_timeout_task)
        watch_timeout_task = None
        cancel_task(error_reset_task)
        error_reset_task = None
        set_state(status="IDLE", message="Waiting for roll request", active_dice_count=0)
        update_debug_overlay(results=[], motion_pixels=0)
        vision.reset_motion_history()
        game_bridge_socket = None
        logger.info("Game bridge disconnected: %s", websocket.client)
