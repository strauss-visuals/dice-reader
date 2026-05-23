const fallbackButton = document.getElementById("fallbackButton");
const result = document.getElementById("result");
const statusPill = document.getElementById("statusPill");
const statusReason = document.getElementById("statusReason");
const videoFeed = document.getElementById("videoFeed");
const stageSelect = document.getElementById("stageSelect");
const motionThreshold = document.getElementById("motionThreshold");
const contourMinArea = document.getElementById("contourMinArea");
const contourMaxArea = document.getElementById("contourMaxArea");
const symbolThresholdValue = document.getElementById("symbolThresholdValue");
const motionThresholdValue = document.getElementById("motionThresholdValue");
const contourMinAreaValue = document.getElementById("contourMinAreaValue");
const contourMaxAreaValue = document.getElementById("contourMaxAreaValue");
const symbolThresholdValueOut = document.getElementById("symbolThresholdValueOut");
const exportDiagnosticsButton = document.getElementById("exportDiagnosticsButton");
const historyBody = document.getElementById("historyBody");

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

function renderHistoryRows(items) {
  if (!items.length) {
    historyBody.innerHTML = "<tr><td colspan=\"5\">No rolls yet.</td></tr>";
    return;
  }

  historyBody.innerHTML = items
    .slice()
    .reverse()
    .map((item) => {
      const diceText = item.dice.map((die) => die.value).join(" ");
      return `<tr>
        <td>${item.timestamp_utc}</td>
        <td>${item.request_id || "-"}</td>
        <td>${item.total_score}</td>
        <td>${item.is_fallback ? "yes" : "no"}</td>
        <td>${diceText}</td>
      </tr>`;
    })
    .join("");
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

function renderSliderValues() {
  motionThresholdValue.textContent = motionThreshold.value;
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
  contourMinArea.value = cfg.vision.contour_min_area;
  contourMaxArea.value = cfg.vision.contour_max_area;
  symbolThresholdValue.value = cfg.vision.symbol_threshold_value;
  renderSliderValues();
}

async function pushVisionConfig() {
  const payload = {
    motion_threshold: Number(motionThreshold.value),
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

[motionThreshold, contourMinArea, contourMaxArea, symbolThresholdValue].forEach((slider) => {
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

async function init() {
  try {
    await loadVisionConfig();
  } catch (error) {
    result.textContent = `Initial config load failed: ${error}`;
  }
  updateVideoStage(stageSelect.value || "raw");
  fetchStatus();
  fetchHistory();
  setInterval(fetchStatus, 1000);
  setInterval(fetchHistory, 1000);
}

init();
