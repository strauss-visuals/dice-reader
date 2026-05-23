# V1 Release Checklist

Use this checklist before cutting the V1 release.

## 1. Environment and startup
- [ ] Run `run.bat` on a clean machine.
- [ ] Confirm server binds to `127.0.0.1` only.
- [ ] Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
- [ ] Confirm live camera feed appears.
- [ ] Confirm startup diagnostics pass (camera + `config.json`).

## 2. Core behavior
- [ ] WebSocket client connects to `/ws/game-bridge`.
- [ ] `REQUEST_ROLL` with `request_id` returns `ROLL_ACK`.
- [ ] Completed roll returns `ROLL_COMPLETE` with same `request_id`.
- [ ] Duplicate request handling behaves as expected.
- [ ] Heartbeat `PING` receives `PONG`.
- [ ] Fallback roll endpoint works and broadcasts result.

## 3. Vision and UI
- [ ] ROI is visible and adjustable.
- [ ] Feed views switch: `raw`, `motion_mask`, `edges`, `thresholded`.
- [ ] Vision sliders update backend config immediately.
- [ ] System Status panel updates in real time.
- [ ] Recent Rolls table updates and shows new results.

## 4. Observability and diagnostics
- [ ] `app.log` is created and written to.
- [ ] State transitions and request IDs are logged.
- [ ] `/api/history` returns last roll results.
- [ ] Export Diagnostics downloads zip with:
  - [ ] `config.json`
  - [ ] `app.log`
  - [ ] `state_snapshot.json`

## 5. Safety and shutdown
- [ ] Stop with `Ctrl + C`.
- [ ] Confirm camera is released (`cap.release()`).
- [ ] Confirm OpenCV windows are destroyed.

## 6. Tests
- [ ] Install test deps: `python -m pip install -r requirements-test.txt`
- [ ] Run tests: `python -m pytest -q`
- [ ] Expected result: all tests pass.

## 7. Git release steps

### Commit message template
```text
feat(v1): release freeze for optical fate dice reader

- finalize websocket protocol hardening (request_id, ACK/error correlation, heartbeat)
- complete fallback/state machine unhappy-path handling
- add observability (rotating logs, history, diagnostics export)
- add phase test coverage and operator documentation
```

### Tag template
```text
v1.0.0
```

### Annotated tag command
```powershell
git tag -a v1.0.0 -m "V1 release: Optical Fate Dice Reader"
```

### Push branch and tag
```powershell
git push origin main
git push origin v1.0.0
```
