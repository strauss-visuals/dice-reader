from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

import main


@pytest.fixture(autouse=True)
def isolated_config_file(tmp_path, monkeypatch):
    class FakeCapture:
        def __init__(self) -> None:
            self.open = True

        def isOpened(self) -> bool:
            return self.open

        def read(self):
            return False, None

        def release(self) -> None:
            self.open = False

    original_config = main.app_config.model_copy(deep=True)
    temporary_config = tmp_path / "config.json"
    temporary_config.write_text(json.dumps(original_config.model_dump()), encoding="utf-8")
    monkeypatch.setattr(main, "config_path", temporary_config)
    monkeypatch.setattr(
        main,
        "open_camera_device",
        lambda index, require_signal=True: FakeCapture() if index == 0 else None,
    )
    main.runtime_state.expected_dice_count = 3
    main.history.clear()
    main.roll_snapshots.clear()
    main.ndi_reconnect_pause_until = 0.0
    yield
    main.cap = None
    main.app_config = original_config
    main.runtime_state.expected_dice_count = 3
    main.apply_config_to_runtime()
    main.history.clear()
    main.roll_snapshots.clear()
    main.ndi_reconnect_pause_until = 0.0


def test_die_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        main.DieResult(value="+", confidence=1.5, bbox=[10, 20, 30, 40])


def test_die_result_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        main.DieResult(value="x", confidence=0.9, bbox=[10, 20, 30, 40])


def test_system_state_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        main.SystemState(status="RUNNING", message="bad", active_dice_count=0)


def test_status_endpoint_returns_system_state() -> None:
    with TestClient(main.app) as client:
        response = client.get("/api/status")
        assert response.status_code == 200
        payload = response.json()
        assert set(payload.keys()) == {"status", "message", "active_dice_count"}
        assert payload["status"] in {"IDLE", "WATCHING", "CALCULATING", "ERROR"}


def test_roi_endpoint_round_trip() -> None:
    with TestClient(main.app) as client:
        updated = {"roi": [10, 20, 300, 220]}
        post_response = client.post("/api/roi", json=updated)
        assert post_response.status_code == 200
        assert post_response.json() == updated

        get_response = client.get("/api/roi")
        assert get_response.status_code == 200
        assert get_response.json() == updated


def test_cameras_endpoint_returns_probe_list() -> None:
    with TestClient(main.app) as client:
        response = client.get("/api/cameras")
        assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 5
    for item in payload:
        assert set(item.keys()) == {"index", "available", "has_signal", "mean_intensity", "contrast"}


def test_camera_update_rejects_unavailable_index() -> None:
    with TestClient(main.app) as client:
        response = client.post("/api/camera", json={"camera_index": 999})
        assert response.status_code == 400
        assert "unavailable" in response.json()["detail"]
        assert client.get("/api/status").json()["status"] == "IDLE"


def test_roi_endpoint_rejects_invalid_shape() -> None:
    with TestClient(main.app) as client:
        response = client.post("/api/roi", json={"roi": [1, 2, 3]})
        assert response.status_code == 422


def test_vision_config_rejects_inverted_contour_area_range() -> None:
    with pytest.raises(ValidationError):
        main.VisionTuningConfig(contour_min_area=2000, contour_max_area=1000)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/config/vision",
            json={
                "motion_threshold": 2000,
                "motion_diff_threshold": 45,
                "settlement_seconds": 1.0,
                "contour_min_area": 2000,
                "contour_max_area": 1000,
                "symbol_threshold_value": 211,
            },
        )
        assert response.status_code == 422


def test_calibration_quality_endpoint_returns_operator_feedback() -> None:
    with TestClient(main.app) as client:
        response = client.get("/api/calibration_quality")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in {"GOOD", "CHECK", "POOR"}
        assert 0 <= payload["score"] <= 100
        assert "message" in payload


def test_calibration_quality_reports_system_error(monkeypatch) -> None:
    monkeypatch.setattr(
        main.runtime_state,
        "system",
        main.SystemState(status="ERROR", message="Camera Not Providing Frames", active_dice_count=0),
    )
    quality = main.calculate_calibration_quality()
    assert quality.status == "POOR"
    assert quality.score == 0
    assert quality.message == "Camera Not Providing Frames"


def test_calibration_quality_waits_for_first_camera_frame(monkeypatch) -> None:
    monkeypatch.setattr(
        main.runtime_state,
        "system",
        main.SystemState(status="IDLE", message="Waiting for roll request", active_dice_count=0),
    )
    monkeypatch.setattr(main, "latest_raw_frame", None)
    quality = main.calculate_calibration_quality()
    assert quality.status == "CHECK"
    assert quality.message == "Waiting for camera frames."


def test_status_frame_bytes_encodes_jpeg_placeholder() -> None:
    frame = main.status_frame_bytes("Waiting for frames")
    assert frame.startswith(b"\xff\xd8")


def test_calibration_profile_can_restore_roi_and_vision_tuning() -> None:
    with TestClient(main.app) as client:
        client.post("/api/roi", json={"roi": [20, 30, 240, 180]})
        client.post(
            "/api/config/vision",
            json={
                "motion_threshold": 1700,
                "contour_min_area": 550,
                "contour_max_area": 32000,
                "symbol_threshold_value": 135,
            },
        )
        saved = client.post("/api/calibration_profiles", json={"name": "table-a"})
        assert saved.status_code == 200

        client.post("/api/roi", json={"roi": [1, 2, 3, 4]})
        applied = client.post("/api/calibration_profiles/apply", json={"name": "table-a"})
        assert applied.status_code == 200
        assert applied.json()["roi"] == [20, 30, 240, 180]
        assert applied.json()["vision"]["motion_threshold"] == 1700


def test_history_snapshot_endpoint_returns_stored_jpeg() -> None:
    payload = main.RollPayload(total_score=0, dice=[], is_fallback=False, roll_confidence=0.8)
    main.append_history(payload, request_id="snapshot-test", snapshot_bytes=b"jpeg-data")
    snapshot_id = main.history[-1].snapshot_id

    with TestClient(main.app) as client:
        response = client.get(f"/api/history/{snapshot_id}/snapshot.jpg")
        assert response.status_code == 200
        assert response.content == b"jpeg-data"
        assert response.headers["content-type"] == "image/jpeg"


def test_still_image_roll_accepts_one_image(monkeypatch) -> None:
    monkeypatch.setattr(
        main.vision,
        "calculate_roll_from_still_image",
        lambda frame: [{"value": "+", "confidence": 0.82, "bbox": [5, 6, 40, 40]}],
    )
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    with TestClient(main.app) as client:
        response = client.post(
            "/api/still_image_roll",
            content=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "total_score": 1,
        "dice": [{"value": "+", "confidence": 0.82, "bbox": [5, 6, 40, 40]}],
        "is_fallback": False,
        "roll_confidence": 0.82,
        "fallback_reason": None,
    }
    assert main.history[-1].request_id == "still-image-upload"


def test_still_image_roll_rejects_unreadable_image() -> None:
    with TestClient(main.app) as client:
        response = client.post(
            "/api/still_image_roll",
            content=b"not an image",
            headers={"Content-Type": "image/jpeg"},
        )

    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]


def test_still_image_roll_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setattr(main, "STILL_IMAGE_MAX_BYTES", 5)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/still_image_roll",
            content=b"abcdef",
            headers={"Content-Type": "image/jpeg"},
        )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"]


def test_still_image_roll_rejects_negative_content_length() -> None:
    async def fake_stream():
        yield b""

    request = type(
        "FakeRequest",
        (),
        {
            "headers": {"content-length": "-1"},
            "stream": lambda self: fake_stream(),
        },
    )()

    with pytest.raises(main.HTTPException) as error:
        asyncio.run(main.read_limited_request_body(request, 5))

    assert error.value.status_code == 400
    assert "Invalid Content-Length" in error.value.detail


def test_settlement_tolerates_small_camera_noise() -> None:
    processor = main.VisionProcessor(
        main.VisionConfig(
            motion_threshold=100,
            settlement_seconds=1.0,
        )
    )

    assert processor.update_settlement(10, now_seconds=0.0) is False
    assert processor.update_settlement(120, now_seconds=0.4) is False
    assert processor.update_settlement(20, now_seconds=1.1) is True


def test_settlement_resets_when_motion_resumes() -> None:
    processor = main.VisionProcessor(
        main.VisionConfig(
            motion_threshold=100,
            settlement_seconds=1.0,
        )
    )

    assert processor.update_settlement(10, now_seconds=0.0) is False
    assert processor.update_settlement(200, now_seconds=0.4) is False
    assert processor.update_settlement(10, now_seconds=1.1) is False
    assert processor.update_settlement(10, now_seconds=2.2) is True


def test_symbol_classifier_reads_centered_fate_faces() -> None:
    processor = main.VisionProcessor(
        main.VisionConfig(
            symbol_threshold_value=180,
            blank_pixel_ratio_threshold=0.03,
        )
    )

    minus_crop = np.full((100, 100, 3), 235, dtype=np.uint8)
    cv2.line(minus_crop, (25, 50), (75, 50), (20, 20, 20), 8)
    assert processor.classify_symbol(minus_crop)[0] == "-"

    plus_crop = np.full((100, 100, 3), 235, dtype=np.uint8)
    cv2.line(plus_crop, (25, 50), (75, 50), (20, 20, 20), 8)
    cv2.line(plus_crop, (50, 25), (50, 75), (20, 20, 20), 8)
    assert processor.classify_symbol(plus_crop)[0] == "+"

    blank_crop = np.full((100, 100, 3), 235, dtype=np.uint8)
    assert processor.classify_symbol(blank_crop)[0] == "blank"


def test_symbol_classifier_supports_colored_dice_with_light_and_dark_symbols() -> None:
    processor = main.VisionProcessor(
        main.VisionConfig(
            symbol_threshold_value=180,
            blank_pixel_ratio_threshold=0.03,
        )
    )

    blue_blank = np.full((120, 120, 3), (180, 70, 20), dtype=np.uint8)
    assert processor.classify_symbol(blue_blank)[0] == "blank"

    orange_blank = np.full((120, 120, 3), (20, 120, 235), dtype=np.uint8)
    orange_value, orange_confidence = processor.classify_symbol(orange_blank)
    assert orange_value == "blank"
    assert orange_confidence >= 0.70

    pink_blank = np.full((120, 120, 3), (190, 40, 235), dtype=np.uint8)
    pink_value, pink_confidence = processor.classify_symbol(pink_blank)
    assert pink_value == "blank"
    assert pink_confidence >= 0.70

    teal_plus = np.full((120, 120, 3), (110, 75, 20), dtype=np.uint8)
    cv2.line(teal_plus, (35, 60), (85, 60), (80, 220, 240), 9)
    cv2.line(teal_plus, (60, 35), (60, 85), (80, 220, 240), 9)
    assert processor.classify_symbol(teal_plus)[0] == "+"

    pink_minus = np.full((120, 120, 3), (190, 40, 235), dtype=np.uint8)
    cv2.line(pink_minus, (35, 60), (85, 60), (245, 245, 245), 9)
    assert processor.classify_symbol(pink_minus)[0] == "-"

    small_purple_minus = np.full((80, 80, 3), (80, 45, 55), dtype=np.uint8)
    cv2.line(small_purple_minus, (32, 40), (48, 38), (30, 150, 215), 4)
    assert processor.classify_symbol(small_purple_minus)[0] == "-"

    white_plus = np.full((120, 120, 3), 235, dtype=np.uint8)
    cv2.line(white_plus, (35, 60), (85, 60), (20, 20, 20), 9)
    cv2.line(white_plus, (60, 35), (60, 85), (20, 20, 20), 9)
    assert processor.classify_symbol(white_plus)[0] == "+"


def test_blank_die_body_detects_as_full_square() -> None:
    processor = main.VisionProcessor(
        main.VisionConfig(
            contour_min_area=1000,
            contour_max_area=20000,
        )
    )
    frame = np.full((180, 180, 3), 35, dtype=np.uint8)
    cv2.rectangle(frame, (50, 45), (130, 125), (220, 35, 180), -1)

    boxes = processor.find_dice_contours(frame)

    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert x <= 55
    assert y <= 50
    assert w >= 70
    assert h >= 70


def test_reconnect_camera_releases_old_capture_and_reopens_configured_device(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    old_capture = FakeCapture()
    replacement = FakeCapture()
    monkeypatch.setattr(main, "cap", old_capture)
    monkeypatch.setattr(main, "camera_index", 2)
    monkeypatch.setattr(main, "open_camera_device", lambda index, require_signal=True: replacement if index == 2 else None)

    assert main.reconnect_camera() is True
    assert old_capture.released is True
    assert main.cap is replacement


def test_reconnect_camera_pauses_ndi_retries_when_source_is_missing(monkeypatch) -> None:
    class FakeReceiver:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    old_receiver = FakeReceiver()
    monkeypatch.setattr(main.app_config, "camera_source_type", "ndi")
    monkeypatch.setattr(main.app_config, "ndi_source_name", "Lobby Camera")
    monkeypatch.setattr(main, "ndi_receiver", old_receiver)
    monkeypatch.setattr(main, "ndi_reconnect_pause_until", 0.0)
    monkeypatch.setattr(main.time, "monotonic", lambda: 100.0)

    def fake_ndi_receiver(source_name: str):
        raise main.NdiSourceNotFoundError(source_name)

    monkeypatch.setattr(main, "NdiReceiver", fake_ndi_receiver)

    assert main.reconnect_camera() is False
    assert old_receiver.released is True
    assert main.ndi_receiver is None
    assert main.ndi_reconnect_pause_until == 130.0


def test_fallback_roll_requires_active_websocket() -> None:
    with TestClient(main.app) as client:
        response = client.post("/api/fallback_roll")
        assert response.status_code == 409
        assert "No active game bridge WebSocket connection" in response.json()["detail"]


def test_fallback_roll_requires_active_roll_request() -> None:
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/game-bridge") as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connect.ok"

            response = client.post("/api/fallback_roll")

            assert response.status_code == 409
            assert "No active roll request" in response.json()["detail"]
            assert main.active_request_id is None


def test_fallback_roll_rejects_calculating_state(monkeypatch) -> None:
    class FakeSocket:
        pass

    monkeypatch.setattr(main, "game_bridge_socket", FakeSocket())
    monkeypatch.setattr(main, "active_request_id", "calculating-request")
    monkeypatch.setattr(
        main.runtime_state,
        "system",
        main.SystemState(
            status="CALCULATING",
            message="Dice settled. Calculating roll.",
            active_dice_count=3,
        ),
    )

    with TestClient(main.app) as client:
        response = client.post("/api/fallback_roll")

    assert response.status_code == 409
    assert "No active roll request" in response.json()["detail"]


def test_duplicate_websocket_bridge_connection_is_rejected() -> None:
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/game-bridge") as first_socket:
            connected = first_socket.receive_json()
            assert connected["type"] == "connect.ok"

            with client.websocket_connect("/ws/game-bridge") as second_socket:
                duplicate_error = second_socket.receive_json()
                assert duplicate_error["type"] == "error"
                assert duplicate_error["event"] == "ROLL_ERROR"
                assert duplicate_error["data"]["reason"] == "BRIDGE_ALREADY_CONNECTED"
                with pytest.raises(WebSocketDisconnect):
                    second_socket.receive_json()

            first_socket.send_json({"type": "ping", "request_id": "first-still-active"})
            pong = first_socket.receive_json()
            assert pong["type"] == "pong"
            assert pong["request_id"] == "first-still-active"


def test_websocket_request_ack_and_fallback_roll_complete() -> None:
    with TestClient(main.app) as client:
        request_id = str(uuid4())
        with client.websocket_connect("/ws/game-bridge") as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connect.ok"

            websocket.send_json(
                {
                    "type": "roll.request",
                    "request_id": request_id,
                    "config": {"expected_dice_count": 3, "timeout_seconds": 30},
                }
            )
            ack = websocket.receive_json()
            assert ack["type"] == "roll.ack"
            assert ack["event"] == "ROLL_ACK"
            assert ack["request_id"] == request_id
            assert ack["data"]["expected_dice_count"] == 3

            fallback_response = client.post("/api/fallback_roll")
            assert fallback_response.status_code == 200

            complete = websocket.receive_json()
            assert complete["type"] == "roll.result"
            assert complete["event"] == "ROLL_COMPLETE"
            assert complete["request_id"] == request_id
            assert complete["data"]["is_fallback"] is True
            assert complete["data"]["roll_confidence"] is None
            assert complete["data"]["fallback_reason"] == "MANUAL_FALLBACK_TRIGGERED"
            assert len(complete["data"]["dice"]) == 3


def test_websocket_ping_and_config_update_protocol() -> None:
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/game-bridge") as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connect.ok"

            websocket.send_json({"type": "ping", "request_id": "ping-001"})
            pong = websocket.receive_json()
            assert pong["type"] == "pong"
            assert pong["event"] == "PONG"
            assert pong["request_id"] == "ping-001"
            assert "ts" in pong["data"]

            websocket.send_json(
                {
                    "type": "config.update",
                    "request_id": "config-001",
                    "config": {"expected_dice_count": 4, "timeout_seconds": 12},
                }
            )
            updated = websocket.receive_json()
            assert updated["type"] == "config.updated"
            assert updated["request_id"] == "config-001"
            assert updated["data"]["expected_dice_count"] == 4
            assert updated["data"]["timeout_seconds"] == 12
            assert main.runtime_state.expected_dice_count == 4


def test_bridge_command_rejects_overlong_request_id() -> None:
    with pytest.raises(ValidationError):
        main.BridgeCommand(type="roll.request", request_id="x" * 121)


def test_process_settled_frame_cleans_request_on_dice_count_mismatch(monkeypatch) -> None:
    scheduled = []

    def fake_schedule(coro):
        scheduled.append(coro)
        coro.close()

    monkeypatch.setattr(main.vision, "calculate_roll", lambda frame: [])
    monkeypatch.setattr(main, "schedule_coroutine", fake_schedule)
    main.runtime_state.expected_dice_count = 3
    main.active_request_id = "mismatch-request"

    main.process_settled_frame(np.zeros((80, 80, 3), dtype=np.uint8))

    assert main.runtime_state.system.status == "ERROR"
    assert main.runtime_state.system.message == "Detected 0 dice, expected 3."
    assert main.active_request_id is None
    assert main.watch_timeout_task is None
    assert main.error_reset_task is None
    assert len(scheduled) == 2


def test_still_image_snapshot_uses_uploaded_image_results(monkeypatch) -> None:
    parsed_results = [{"value": "+", "confidence": 0.82, "bbox": [5, 6, 40, 40]}]
    overlay_results = []

    monkeypatch.setattr(main.vision, "calculate_roll_from_still_image", lambda frame: parsed_results)

    def fake_draw_debug_overlay(frame, results=None):
        overlay_results.append(results)
        return frame

    monkeypatch.setattr(main, "draw_debug_overlay", fake_draw_debug_overlay)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    with TestClient(main.app) as client:
        response = client.post(
            "/api/still_image_roll",
            content=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg"},
        )

    assert response.status_code == 200
    assert overlay_results == [parsed_results]


def test_troubleshooting_ui_uses_text_apis_for_dynamic_tables() -> None:
    script = (Path("templates") / "app.js").read_text(encoding="utf-8")
    assert "function appendTextCell" in script
    assert "historyBody.replaceChildren(...rows)" in script
    assert "suitabilityBody.replaceChildren(...rows)" in script
    assert "innerHTML" not in script
