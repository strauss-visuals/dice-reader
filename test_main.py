from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

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
    monkeypatch.setattr(main, "open_camera_device", lambda index: FakeCapture() if index == 0 else None)
    main.history.clear()
    main.roll_snapshots.clear()
    yield
    main.cap = None
    main.app_config = original_config
    main.apply_config_to_runtime()
    main.history.clear()
    main.roll_snapshots.clear()


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
            assert set(item.keys()) == {"index", "available"}


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
    monkeypatch.setattr(main, "open_camera_device", lambda index: replacement if index == 2 else None)

    assert main.reconnect_camera() is True
    assert old_capture.released is True
    assert main.cap is replacement


def test_fallback_roll_requires_active_websocket() -> None:
    with TestClient(main.app) as client:
        response = client.post("/api/fallback_roll")
        assert response.status_code == 409
        assert "No active game bridge WebSocket connection" in response.json()["detail"]


def test_websocket_request_ack_and_fallback_roll_complete() -> None:
    with TestClient(main.app) as client:
        request_id = str(uuid4())
        with client.websocket_connect("/ws/game-bridge") as websocket:
            websocket.send_json(
                {
                    "action": "REQUEST_ROLL",
                    "request_id": request_id,
                    "config": {"expected_dice_count": 3, "timeout_seconds": 30},
                }
            )
            ack = websocket.receive_json()
            assert ack["event"] == "ROLL_ACK"
            assert ack["request_id"] == request_id
            assert ack["data"]["expected_dice_count"] == 3

            fallback_response = client.post("/api/fallback_roll")
            assert fallback_response.status_code == 200

            complete = websocket.receive_json()
            assert complete["event"] == "ROLL_COMPLETE"
            assert complete["request_id"] == request_id
            assert complete["data"]["is_fallback"] is True
            assert complete["data"]["roll_confidence"] is None
            assert complete["data"]["fallback_reason"] == "MANUAL_FALLBACK_TRIGGERED"
            assert len(complete["data"]["dice"]) == 3
