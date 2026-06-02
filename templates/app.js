const fallbackButton = document.getElementById("fallbackButton");
const result = document.getElementById("result");
const statusPill = document.getElementById("statusPill");
const statusReason = document.getElementById("statusReason");
const qualityPill = document.getElementById("qualityPill");
const qualityScore = document.getElementById("qualityScore");
const qualityMessage = document.getElementById("qualityMessage");
const analyzeSuitabilityButton = document.getElementById("analyzeSuitabilityButton");
const suitabilityBody = document.getElementById("suitabilityBody");
const videoFeed = document.getElementById("videoFeed");
const roiCanvas = document.getElementById("roiCanvas");
const cameraSelect = document.getElementById("cameraSelect");
const ndiSourceSelect = document.getElementById("ndiSourceSelect");
const refreshNdiButton = document.getElementById("refreshNdiButton");
const useNdiButton = document.getElementById("useNdiButton");
const stageSelect = document.getElementById("stageSelect");
const motionThreshold = document.getElementById("motionThreshold");
const motionDiffThreshold = document.getElementById("motionDiffThreshold");
const settlementSeconds = document.getElementById("settlementSeconds");
const contourMinArea = document.getElementById("contourMinArea");
const contourMaxArea = document.getElementById("contourMaxArea");
const symbolThresholdValue = document.getElementById("symbolThresholdValue");
const motionThresholdValue = document.getElementById("motionThresholdValue");
const motionDiffThresholdValue = document.getElementById("motionDiffThresholdValue");
const settlementSecondsValue = document.getElementById("settlementSecondsValue");
const contourMinAreaValue = document.getElementById("contourMinAreaValue");
const contourMaxAreaValue = document.getElementById("contourMaxAreaValue");
const symbolThresholdValueOut = document.getElementById("symbolThresholdValueOut");
const exportDiagnosticsButton = document.getElementById("exportDiagnosticsButton");
const historyBody = document.getElementById("historyBody");
const roiX = document.getElementById("roiX");
const roiY = document.getElementById("roiY");
const roiW = document.getElementById("roiW");
const roiH = document.getElementById("roiH");
const saveRoiButton = document.getElementById("saveRoiButton");
const wizardProgress = document.getElementById("wizardProgress");
const wizardTitle = document.getElementById("wizardTitle");
const wizardHint = document.getElementById("wizardHint");
const wizardBackButton = document.getElementById("wizardBackButton");
const wizardNextButton = document.getElementById("wizardNextButton");
const snapshotModal = document.getElementById("snapshotModal");
const snapshotModalImage = document.getElementById("snapshotModalImage");
const closeSnapshotModalButton = document.getElementById("closeSnapshotModalButton");
const profileName = document.getElementById("profileName");
const profileSelect = document.getElementById("profileSelect");
const saveProfileButton = document.getElementById("saveProfileButton");
const loadProfileButton = document.getElementById("loadProfileButton");
const stillImageInput = document.getElementById("stillImageInput");
const stillImageButton = document.getElementById("stillImageButton");
const stillImageResult = document.getElementById("stillImageResult");
const stillImagePreview = document.getElementById("stillImagePreview");
const bridgeStatusPill = document.getElementById("bridgeStatusPill");
const bridgeRequestId = document.getElementById("bridgeRequestId");
const bridgeDiceCount = document.getElementById("bridgeDiceCount");
const bridgeTimeoutSeconds = document.getElementById("bridgeTimeoutSeconds");
const bridgeConnectButton = document.getElementById("bridgeConnectButton");
const bridgeDisconnectButton = document.getElementById("bridgeDisconnectButton");
const bridgeConnectMessageButton = document.getElementById("bridgeConnectMessageButton");
const bridgePingButton = document.getElementById("bridgePingButton");
const bridgeConfigButton = document.getElementById("bridgeConfigButton");
const bridgeRollButton = document.getElementById("bridgeRollButton");
const bridgeClearLogButton = document.getElementById("bridgeClearLogButton");
const bridgeMessageInput = document.getElementById("bridgeMessageInput");
const bridgeSendCustomButton = document.getElementById("bridgeSendCustomButton");
const bridgeMessageLog = document.getElementById("bridgeMessageLog");

const wizardSteps = [
  {
    title: "Step 1: Camera Check",
    hint: "Select the camera source. Confirm you can see the tray in Raw view.",
    onEnter: () => {
      stageSelect.value = "raw";
      updateVideoStage("raw");
    },
  },
  {
    title: "Step 2: ROI Setup",
    hint: "Set ROI X/Y/Width/Height so the ROI box tightly frames only the dice tray, then click Save ROI.",
    onEnter: () => {
      stageSelect.value = "edges";
      updateVideoStage("edges");
    },
  },
  {
    title: "Step 3: Symbol Tuning",
    hint: "Use Thresholded view and tune sliders until +, -, and blank faces separate clearly.",
    onEnter: () => {
      stageSelect.value = "thresholded";
      updateVideoStage("thresholded");
    },
  },
];

let wizardStepIndex = 0;
let currentRoi = null;
let roiDragStart = null;
let bridgeSocket = null;

function updateStatusView(state) {
  const status = state.status || "ERROR";
  const message = state.message || "";

  statusPill.textContent = status;
  statusPill.className = `statusPill status-${status}`;

  if (status === "ERROR") {
    statusReason.classList.add("error");
    statusReason.textContent = message || "Unknown error";
    return;
  }

  statusReason.classList.remove("error");
  statusReason.textContent = message || "No active message";
}

async function fetchStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const state = await response.json();
    updateStatusView(state);
  } catch (error) {
    statusPill.textContent = "ERROR";
    statusPill.className = "statusPill status-ERROR";
    statusReason.classList.add("error");
    statusReason.textContent = `Status fetch failed: ${error}`;
  }
}

async function fetchCalibrationQuality() {
  try {
    const response = await fetch("/api/calibration_quality");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const quality = await response.json();
    qualityPill.textContent = quality.status;
    qualityPill.className = `statusPill quality-${quality.status}`;
    qualityScore.textContent = `${quality.score} / 100`;
    qualityMessage.textContent = quality.message;
  } catch (error) {
    qualityPill.textContent = "POOR";
    qualityPill.className = "statusPill quality-POOR";
    qualityMessage.textContent = `Quality fetch failed: ${error}`;
  }
}

function renderHistoryRows(items) {
  if (!items.length) {
    historyBody.innerHTML = "<tr><td colspan=\"8\">No rolls yet.</td></tr>";
    return;
  }

  historyBody.innerHTML = items
    .slice()
    .reverse()
    .map((item) => {
      const diceText = item.dice.map((die) => {
        const confidence = Number(die.confidence);
        const warningClass = confidence < 0.7 ? " low" : "";
        return `<span class="confidenceBadge${warningClass}">${die.value} ${confidence.toFixed(2)}</span>`;
      }).join(" ");
      const confidenceText = item.roll_confidence === null || item.roll_confidence === undefined
        ? "-"
        : Number(item.roll_confidence).toFixed(3);
      const reasonText = item.fallback_reason || "-";
      const snapshotLink = item.snapshot_id
        ? `<button type="button" class="snapshotPreviewButton" data-snapshot-id="${item.snapshot_id}">Open</button>`
        : "-";
      return `<tr>
        <td>${item.timestamp_utc}</td>
        <td>${item.request_id || "-"}</td>
        <td>${item.total_score}</td>
        <td>${item.is_fallback ? "yes" : "no"}</td>
        <td>${confidenceText}</td>
        <td>${reasonText}</td>
        <td>${diceText}</td>
        <td>${snapshotLink}</td>
      </tr>`;
    })
    .join("");
}

function closeSnapshotModal() {
  snapshotModal.classList.remove("open");
  snapshotModal.setAttribute("aria-hidden", "true");
  snapshotModalImage.removeAttribute("src");
}

function renderSuitabilityReport(report) {
  if (!report.items.length) {
    suitabilityBody.innerHTML = "<tr><td colspan=\"6\">No dice detected.</td></tr>";
    return;
  }

  suitabilityBody.innerHTML = report.items
    .map((item) => {
      return `
        <tr>
          <td>${item.index}</td>
          <td>${item.value}</td>
          <td>${Number(item.confidence).toFixed(2)}</td>
          <td>${item.color_label}</td>
          <td>${item.status}</td>
          <td>${item.reason}</td>
        </tr>
      `;
    })
    .join("");
}

async function fetchDiceSuitability() {
  const response = await fetch("/api/dice_suitability");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const report = await response.json();
  renderSuitabilityReport(report);
  return report;
}

function openSnapshotModal(snapshotId) {
  const src = `/api/history/${encodeURIComponent(snapshotId)}/snapshot.jpg?t=${Date.now()}`;
  snapshotModalImage.src = src;
  snapshotModal.classList.add("open");
  snapshotModal.setAttribute("aria-hidden", "false");
}

async function fetchHistory() {
  try {
    const response = await fetch("/api/history");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const items = await response.json();
    renderHistoryRows(items);
  } catch (error) {
    result.textContent = `History fetch failed: ${error}`;
  }
}

function updateVideoStage(stage) {
  videoFeed.src = `/video_feed?stage=${encodeURIComponent(stage)}&t=${Date.now()}`;
}

function drawRoiOverlay(roi = currentRoi) {
  const width = videoFeed.clientWidth;
  const height = videoFeed.clientHeight;
  if (!width || !height) {
    return;
  }
  roiCanvas.width = width;
  roiCanvas.height = height;
  const context = roiCanvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  if (!roi) {
    return;
  }
  const sourceWidth = videoFeed.naturalWidth || width;
  const sourceHeight = videoFeed.naturalHeight || height;
  const scaleX = width / sourceWidth;
  const scaleY = height / sourceHeight;
  context.strokeStyle = "#f04d7c";
  context.lineWidth = 3;
  context.setLineDash([8, 4]);
  context.strokeRect(roi[0] * scaleX, roi[1] * scaleY, roi[2] * scaleX, roi[3] * scaleY);
}

function videoPointFromEvent(event) {
  const bounds = roiCanvas.getBoundingClientRect();
  const sourceWidth = videoFeed.naturalWidth || bounds.width;
  const sourceHeight = videoFeed.naturalHeight || bounds.height;
  return {
    x: Math.round(Math.max(0, Math.min(bounds.width, event.clientX - bounds.left)) * sourceWidth / bounds.width),
    y: Math.round(Math.max(0, Math.min(bounds.height, event.clientY - bounds.top)) * sourceHeight / bounds.height),
  };
}

function setRoiFields(roi) {
  currentRoi = roi;
  roiX.value = roi[0];
  roiY.value = roi[1];
  roiW.value = roi[2];
  roiH.value = roi[3];
  drawRoiOverlay();
}

function renderSliderValues() {
  motionThresholdValue.textContent = motionThreshold.value;
  motionDiffThresholdValue.textContent = motionDiffThreshold.value;
  settlementSecondsValue.textContent = `${Number(settlementSeconds.value).toFixed(1)}s`;
  contourMinAreaValue.textContent = contourMinArea.value;
  contourMaxAreaValue.textContent = contourMaxArea.value;
  symbolThresholdValueOut.textContent = symbolThresholdValue.value;
}

async function loadVisionConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const cfg = await response.json();
  motionThreshold.value = cfg.vision.motion_threshold;
  motionDiffThreshold.value = cfg.vision.motion_diff_threshold || 30;
  settlementSeconds.value = cfg.vision.settlement_seconds || 1.5;
  contourMinArea.value = cfg.vision.contour_min_area;
  contourMaxArea.value = cfg.vision.contour_max_area;
  symbolThresholdValue.value = cfg.vision.symbol_threshold_value;
  renderSliderValues();
  return cfg;
}

function renderWizardStep() {
  const step = wizardSteps[wizardStepIndex];
  wizardProgress.textContent = `Step ${wizardStepIndex + 1} of ${wizardSteps.length}`;
  wizardTitle.innerHTML = `<strong>${step.title}</strong>`;
  wizardHint.textContent = step.hint;
  wizardBackButton.disabled = wizardStepIndex === 0;
  wizardNextButton.textContent = wizardStepIndex === wizardSteps.length - 1 ? "Finish" : "Next";
  step.onEnter();
}

function parseIntSafe(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

function makeBridgeConfig() {
  return {
    expected_dice_count: Math.max(1, parseIntSafe(bridgeDiceCount.value, 3)),
    timeout_seconds: Math.max(1, parseIntSafe(bridgeTimeoutSeconds.value, 30)),
  };
}

function makeBridgeMessage(type) {
  const requestId = bridgeRequestId.value.trim();
  const base = { type };
  if (requestId) {
    base.request_id = requestId;
  }
  if (type === "connect") {
    base.client_name = "troubleshooting-ui";
  }
  if (type === "config.update" || type === "roll.request") {
    base.config = makeBridgeConfig();
  }
  if (type === "roll.request" && !base.request_id) {
    base.request_id = crypto.randomUUID();
    bridgeRequestId.value = base.request_id;
  }
  return base;
}

function setBridgeMessageTemplate(type) {
  bridgeMessageInput.value = JSON.stringify(makeBridgeMessage(type), null, 2);
}

function setBridgeStatus(status) {
  bridgeStatusPill.textContent = status;
  bridgeStatusPill.className = status === "CONNECTED"
    ? "statusPill status-WATCHING"
    : "statusPill status-IDLE";
}

function appendBridgeLog(direction, payload) {
  const timestamp = new Date().toISOString();
  const body = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  bridgeMessageLog.textContent += `[${timestamp}] ${direction}\n${body}\n\n`;
  bridgeMessageLog.scrollTop = bridgeMessageLog.scrollHeight;
}

function connectBridgeSocket() {
  if (bridgeSocket && bridgeSocket.readyState === WebSocket.OPEN) {
    appendBridgeLog("info", "Already connected.");
    return;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  bridgeSocket = new WebSocket(`${protocol}//${window.location.host}/ws/game-bridge`);
  setBridgeStatus("CONNECTING");

  bridgeSocket.addEventListener("open", () => {
    setBridgeStatus("CONNECTED");
    appendBridgeLog("open", "/ws/game-bridge");
  });
  bridgeSocket.addEventListener("message", (event) => {
    try {
      appendBridgeLog("received", JSON.parse(event.data));
    } catch {
      appendBridgeLog("received", event.data);
    }
  });
  bridgeSocket.addEventListener("close", () => {
    setBridgeStatus("DISCONNECTED");
    appendBridgeLog("close", "/ws/game-bridge");
  });
  bridgeSocket.addEventListener("error", () => {
    appendBridgeLog("error", "WebSocket error.");
  });
}

function sendBridgeJson(message) {
  if (!bridgeSocket || bridgeSocket.readyState !== WebSocket.OPEN) {
    throw new Error("connect the WebSocket first");
  }
  bridgeSocket.send(JSON.stringify(message));
  appendBridgeLog("sent", message);
}

function sendBridgeTemplate(type) {
  const message = makeBridgeMessage(type);
  bridgeMessageInput.value = JSON.stringify(message, null, 2);
  sendBridgeJson(message);
}

async function loadRoiConfig() {
  const response = await fetch("/api/roi");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const data = await response.json();
  const roi = data.roi || [0, 0, 300, 220];
  setRoiFields(roi);
}

async function pushRoiConfig() {
  const payload = {
    roi: [
      parseIntSafe(roiX.value, 0),
      parseIntSafe(roiY.value, 0),
      Math.max(1, parseIntSafe(roiW.value, 300)),
      Math.max(1, parseIntSafe(roiH.value, 220)),
    ],
  };
  const response = await fetch("/api/roi", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const saved = await response.json();
  setRoiFields(saved.roi);
  return saved;
}

function renderCameraOptions(cameras, activeIndex) {
  cameraSelect.innerHTML = cameras
    .map((camera) => {
      const selected = camera.index === activeIndex ? "selected" : "";
      let stateLabel = "unavailable";
      if (camera.available && camera.has_signal) {
        stateLabel = "signal";
      } else if (camera.available) {
        stateLabel = "no signal";
      }
      return `<option value="${camera.index}" ${selected}>Camera ${camera.index} (${stateLabel})</option>`;
    })
    .join("");
}

async function loadCameraOptions(activeIndex) {
  const response = await fetch("/api/cameras");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const cameras = await response.json();
  renderCameraOptions(cameras, activeIndex);
}

function renderNdiSources(sources, activeName) {
  if (!sources.length) {
    ndiSourceSelect.innerHTML = "<option value=\"\">No NDI sources found</option>";
    return;
  }

  ndiSourceSelect.innerHTML = sources
    .map((source) => {
      const selected = source.name === activeName ? "selected" : "";
      return `<option value="${source.name}" ${selected}>${source.name}</option>`;
    })
    .join("");
}

async function loadNdiSources(activeName = null) {
  const response = await fetch("/api/ndi/sources");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const sources = await response.json();
  renderNdiSources(sources, activeName);
}

async function pushCameraSelection() {
  const payload = { camera_index: Number(cameraSelect.value) };
  const response = await fetch("/api/camera", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function pushNdiSelection() {
  if (!ndiSourceSelect.value) {
    throw new Error("Choose an NDI source first.");
  }
  const response = await fetch("/api/ndi/source", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_name: ndiSourceSelect.value }),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function pushVisionConfig() {
  const payload = {
    motion_threshold: Number(motionThreshold.value),
    motion_diff_threshold: Number(motionDiffThreshold.value),
    settlement_seconds: Number(settlementSeconds.value),
    contour_min_area: Number(contourMinArea.value),
    contour_max_area: Number(contourMaxArea.value),
    symbol_threshold_value: Number(symbolThresholdValue.value),
  };
  const response = await fetch("/api/config/vision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
}

function renderProfileOptions(profiles) {
  const names = Object.keys(profiles).sort();
  profileSelect.innerHTML = names.length
    ? names.map((name) => `<option value="${name}">${name}</option>`).join("")
    : "<option value=\"\">No profiles saved</option>";
}

async function loadProfiles() {
  const response = await fetch("/api/calibration_profiles");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  renderProfileOptions(await response.json());
}

async function saveProfile() {
  const name = profileName.value.trim();
  if (!name) {
    throw new Error("enter a profile name first");
  }
  const response = await fetch("/api/calibration_profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  await loadProfiles();
  profileSelect.value = name;
}

async function applySelectedProfile() {
  const name = profileSelect.value;
  if (!name) {
    throw new Error("select a saved profile first");
  }
  const response = await fetch("/api/calibration_profiles/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  await loadVisionConfig();
  await loadRoiConfig();
  updateVideoStage(stageSelect.value || "raw");
}

async function analyzeStillImage() {
  const file = stillImageInput.files[0];
  if (!file) {
    stillImageResult.textContent = "Choose an image file first.";
    return;
  }

  stillImageButton.disabled = true;
  stillImageResult.textContent = "Analyzing image...";
  try {
    const response = await fetch("/api/still_image_roll", {
      method: "POST",
      headers: { "Content-Type": file.type || "image/jpeg" },
      body: file,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    stillImageResult.textContent = JSON.stringify(data, null, 2);
    await fetchHistory();
  } catch (error) {
    stillImageResult.textContent = `Image analysis failed: ${error}`;
  } finally {
    stillImageButton.disabled = false;
  }
}

fallbackButton.addEventListener("click", async () => {
  result.textContent = "Generating fallback roll...";
  try {
    const response = await fetch("/api/fallback_roll", { method: "POST" });
    const data = await response.json();
    result.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    result.textContent = `Request failed: ${error}`;
  }
});

stageSelect.addEventListener("change", () => {
  updateVideoStage(stageSelect.value);
});

videoFeed.addEventListener("load", () => {
  drawRoiOverlay();
});

window.addEventListener("resize", () => {
  drawRoiOverlay();
});

roiCanvas.addEventListener("pointerdown", (event) => {
  roiDragStart = videoPointFromEvent(event);
  roiCanvas.setPointerCapture(event.pointerId);
});

roiCanvas.addEventListener("pointermove", (event) => {
  if (!roiDragStart) {
    return;
  }
  const current = videoPointFromEvent(event);
  const preview = [
    Math.min(roiDragStart.x, current.x),
    Math.min(roiDragStart.y, current.y),
    Math.abs(current.x - roiDragStart.x),
    Math.abs(current.y - roiDragStart.y),
  ];
  drawRoiOverlay(preview);
});

roiCanvas.addEventListener("pointerup", async (event) => {
  if (!roiDragStart) {
    return;
  }
  const end = videoPointFromEvent(event);
  const drawnRoi = [
    Math.min(roiDragStart.x, end.x),
    Math.min(roiDragStart.y, end.y),
    Math.max(1, Math.abs(end.x - roiDragStart.x)),
    Math.max(1, Math.abs(end.y - roiDragStart.y)),
  ];
  roiDragStart = null;
  setRoiFields(drawnRoi);
  try {
    await pushRoiConfig();
    result.textContent = "ROI saved from video selection.";
  } catch (error) {
    result.textContent = `ROI save failed: ${error}`;
  }
});

wizardBackButton.addEventListener("click", () => {
  wizardStepIndex = Math.max(0, wizardStepIndex - 1);
  renderWizardStep();
});

wizardNextButton.addEventListener("click", () => {
  if (wizardStepIndex < wizardSteps.length - 1) {
    wizardStepIndex += 1;
    renderWizardStep();
    return;
  }
  result.textContent = "Calibration wizard complete. You can still fine-tune controls anytime.";
});

cameraSelect.addEventListener("change", async () => {
  result.textContent = "Switching camera...";
  try {
    await pushCameraSelection();
    updateVideoStage(stageSelect.value || "raw");
    await fetchStatus();
    result.textContent = `Camera switched to index ${cameraSelect.value}`;
  } catch (error) {
    result.textContent = `Camera switch failed: ${error}`;
  }
});

refreshNdiButton.addEventListener("click", async () => {
  result.textContent = "Scanning NDI sources...";
  try {
    await loadNdiSources();
    result.textContent = "NDI scan complete.";
  } catch (error) {
    result.textContent = `NDI scan failed: ${error}`;
  }
});

useNdiButton.addEventListener("click", async () => {
  result.textContent = "Switching to NDI source...";
  try {
    const cfg = await pushNdiSelection();
    updateVideoStage(stageSelect.value || "raw");
    await fetchStatus();
    result.textContent = `NDI source selected: ${cfg.ndi_source_name}`;
  } catch (error) {
    result.textContent = `NDI switch failed: ${error}`;
  }
});

saveRoiButton.addEventListener("click", async () => {
  result.textContent = "Saving ROI...";
  try {
    await pushRoiConfig();
    updateVideoStage(stageSelect.value || "raw");
    result.textContent = "ROI saved.";
  } catch (error) {
    result.textContent = `ROI save failed: ${error}`;
  }
});

saveProfileButton.addEventListener("click", async () => {
  try {
    await saveProfile();
    result.textContent = `Calibration profile saved: ${profileSelect.value}`;
  } catch (error) {
    result.textContent = `Profile save failed: ${error}`;
  }
});

loadProfileButton.addEventListener("click", async () => {
  try {
    await applySelectedProfile();
    result.textContent = `Calibration profile loaded: ${profileSelect.value}`;
  } catch (error) {
    result.textContent = `Profile load failed: ${error}`;
  }
});

stillImageInput.addEventListener("change", () => {
  const file = stillImageInput.files[0];
  if (!file) {
    stillImagePreview.removeAttribute("src");
    stillImagePreview.hidden = true;
    stillImageResult.textContent = "";
    return;
  }
  stillImagePreview.src = URL.createObjectURL(file);
  stillImagePreview.hidden = false;
  stillImageResult.textContent = "Image selected. Click Analyze Image.";
});

stillImageButton.addEventListener("click", () => {
  analyzeStillImage();
});

bridgeConnectButton.addEventListener("click", () => {
  connectBridgeSocket();
});

bridgeDisconnectButton.addEventListener("click", () => {
  if (bridgeSocket) {
    bridgeSocket.close();
  }
});

bridgeConnectMessageButton.addEventListener("click", () => {
  try {
    sendBridgeTemplate("connect");
  } catch (error) {
    appendBridgeLog("error", String(error));
  }
});

bridgePingButton.addEventListener("click", () => {
  try {
    sendBridgeTemplate("ping");
  } catch (error) {
    appendBridgeLog("error", String(error));
  }
});

bridgeConfigButton.addEventListener("click", () => {
  try {
    sendBridgeTemplate("config.update");
  } catch (error) {
    appendBridgeLog("error", String(error));
  }
});

bridgeRollButton.addEventListener("click", () => {
  try {
    sendBridgeTemplate("roll.request");
  } catch (error) {
    appendBridgeLog("error", String(error));
  }
});

bridgeSendCustomButton.addEventListener("click", () => {
  try {
    sendBridgeJson(JSON.parse(bridgeMessageInput.value));
  } catch (error) {
    appendBridgeLog("error", String(error));
  }
});

bridgeClearLogButton.addEventListener("click", () => {
  bridgeMessageLog.textContent = "";
});

[motionThreshold, motionDiffThreshold, settlementSeconds, contourMinArea, contourMaxArea, symbolThresholdValue].forEach((slider) => {
  slider.addEventListener("input", async () => {
    renderSliderValues();
    try {
      await pushVisionConfig();
    } catch (error) {
      result.textContent = `Config update failed: ${error}`;
    }
  });
});

exportDiagnosticsButton.addEventListener("click", () => {
  window.location.href = "/api/export_diagnostics";
});

analyzeSuitabilityButton.addEventListener("click", async () => {
  result.textContent = "Analyzing dice suitability...";
  try {
    const report = await fetchDiceSuitability();
    result.textContent = `Dice suitability analyzed: ${report.dice_count} detected.`;
  } catch (error) {
    result.textContent = `Dice suitability failed: ${error}`;
  }
});

historyBody.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (!target.classList.contains("snapshotPreviewButton")) {
    return;
  }
  const snapshotId = target.dataset.snapshotId;
  if (!snapshotId) {
    result.textContent = "Snapshot preview failed: missing snapshot id.";
    return;
  }
  openSnapshotModal(snapshotId);
});

closeSnapshotModalButton.addEventListener("click", () => {
  closeSnapshotModal();
});

snapshotModal.addEventListener("click", (event) => {
  if (event.target === snapshotModal) {
    closeSnapshotModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && snapshotModal.classList.contains("open")) {
    closeSnapshotModal();
  }
});

async function init() {
  bridgeRequestId.value = crypto.randomUUID();
  setBridgeMessageTemplate("roll.request");
  fetchStatus();
  fetchCalibrationQuality();
  fetchHistory();
  setInterval(fetchStatus, 1000);
  setInterval(fetchCalibrationQuality, 1000);
  setInterval(fetchHistory, 1000);
  let cfg = null;
  try {
    cfg = await loadVisionConfig();
    await Promise.all([loadRoiConfig(), loadProfiles()]);
  } catch (error) {
    result.textContent = `Initial config load failed: ${error}`;
  }
  renderWizardStep();
  if (cfg) {
    loadCameraOptions(cfg.camera_index).catch((error) => {
      result.textContent = `Camera scan failed: ${error}`;
    });
    loadNdiSources(cfg.ndi_source_name).catch((error) => {
      result.textContent = `NDI scan failed: ${error}`;
    });
  }
}

init();
