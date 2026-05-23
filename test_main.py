from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main


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
            assert len(complete["data"]["dice"]) == 3
