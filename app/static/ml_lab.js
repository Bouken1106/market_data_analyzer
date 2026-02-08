const mlForm = document.getElementById("ml-form");
const mlSymbolSearchArea = document.getElementById("ml-symbol-search-area");
const mlSymbolInput = document.getElementById("ml-symbol");
const mlSymbolDropdown = document.getElementById("ml-symbol-dropdown");
const mlCatalogMetaEl = document.getElementById("ml-catalog-meta");
const mlRefreshCatalogBtn = document.getElementById("ml-refresh-catalog");
const mlModelGridEl = document.getElementById("ml-model-grid");
const mlActiveModelPillEl = document.getElementById("ml-active-model-pill");
const mlConfigTitleEl = document.getElementById("ml-config-title");
const mlModelDescriptionEl = document.getElementById("ml-model-description");
const runBtn = document.getElementById("ml-run-btn");
const statusEl = document.getElementById("ml-status");
const progressWrapEl = document.getElementById("ml-progress-wrap");
const progressBarEl = document.getElementById("ml-progress-bar");
const progressTextEl = document.getElementById("ml-progress-text");
const quantileCanvas = document.getElementById("quantile-function-canvas");
const fanCanvas = document.getElementById("fan-chart-canvas");
const nextDayCanvas = document.getElementById("next-day-canvas");
const backtestCanvas = document.getElementById("backtest-canvas");
const nextDayScaleEl = document.getElementById("next-day-scale");
const nextDayDistTypeEl = document.getElementById("next-day-dist-type");
const quantileScaleEl = document.getElementById("quantile-scale");
const fanScaleEl = document.getElementById("fan-scale");
const quantileZoomInBtn = document.getElementById("quantile-zoom-in");
const quantileZoomOutBtn = document.getElementById("quantile-zoom-out");
const quantileZoomResetBtn = document.getElementById("quantile-zoom-reset");
const fanZoomInBtn = document.getElementById("fan-zoom-in");
const fanZoomOutBtn = document.getElementById("fan-zoom-out");
const fanZoomResetBtn = document.getElementById("fan-zoom-reset");
const quantileLegendEl = document.getElementById("quantile-legend");
const fanMetaEl = document.getElementById("fan-meta");
const nextDayMetaEl = document.getElementById("next-day-meta");
const metricPinballEl = document.getElementById("metric-pinball");
const metricCov90El = document.getElementById("metric-cov90");
const metricCov50El = document.getElementById("metric-cov50");
const metricSamplesEl = document.getElementById("metric-samples");
const btReturnStrategyEl = document.getElementById("bt-return-strategy");
const btReturnBuyholdEl = document.getElementById("bt-return-buyhold");
const btOutperfEl = document.getElementById("bt-outperf");
const btCapitalStrategyEl = document.getElementById("bt-capital-strategy");
const btCapitalBuyholdEl = document.getElementById("bt-capital-buyhold");
const btWindowEl = document.getElementById("bt-window");

const COLORS = {
  bg: "#0b111d",
  axis: "#334862",
  grid: "#1f3046",
  label: "#99abc3",
  median: "#15d1ff",
  actual: "#ffc857",
  band90: "rgba(55, 132, 201, 0.22)",
  band50: "rgba(21, 209, 255, 0.28)",
  curves: ["#14d5ff", "#6bb8ff", "#ff9f6e", "#8fe388", "#e0c1ff", "#ffe07a"],
};

let latestPayload = null;
let mlSymbolCatalog = [];
let mlModels = [];
let activeModelId = "quantile_lstm";
let isRunning = false;
const MAX_DROPDOWN_ITEMS = 120;
const FALLBACK_ML_MODELS = [
  {
    id: "quantile_lstm",
    name: "Quantile LSTM",
    short_description: "翌営業日の分位点分布を推定（現在利用可能）",
    status: "ready",
    status_label: "Ready",
    run_label: "Run Quantile LSTM",
    api_path: "/api/ml/quantile-lstm",
  },
  {
    id: "quantile_gru",
    name: "Quantile GRU",
    short_description: "LSTMより軽量な系列モデル（準備中）",
    status: "coming_soon",
    status_label: "Coming Soon",
    run_label: "Run Quantile GRU",
    api_path: "",
  },
  {
    id: "temporal_transformer",
    name: "Temporal Transformer",
    short_description: "注意機構ベースの時系列モデル（準備中）",
    status: "coming_soon",
    status_label: "Coming Soon",
    run_label: "Run Temporal Transformer",
    api_path: "",
  },
  {
    id: "xgboost_quantile",
    name: "XGBoost Quantile",
    short_description: "勾配ブースティングの分位点回帰（準備中）",
    status: "coming_soon",
    status_label: "Coming Soon",
    run_label: "Run XGBoost Quantile",
    api_path: "",
  },
];

function createZoomState() {
  return {
    xScale: 1,
    yScale: 1,
    min: 1,
    max: 30,
    xCenterRatio: 0.5,
    yCenterRatio: 0.5,
  };
}

const quantileZoom = createZoomState();
const fanZoom = createZoomState();

const chartViewState = {
  quantile: null,
  fan: null,
};

function normalizeSymbol(raw) {
  return String(raw || "").trim().toUpperCase().replace(/[^A-Z0-9.\-]/g, "");
}

function setMlCatalogMeta(message) {
  if (!mlCatalogMetaEl) return;
  mlCatalogMetaEl.textContent = message || "";
}

function showMlDropdown() {
  if (!mlSymbolDropdown) return;
  mlSymbolDropdown.classList.remove("hidden");
}

function hideMlDropdown() {
  if (!mlSymbolDropdown) return;
  mlSymbolDropdown.classList.add("hidden");
}

function pickMlCandidates(query) {
  const needle = String(query || "").trim().toUpperCase();
  const candidates = [];
  for (const item of mlSymbolCatalog) {
    if (!needle || item.symbol.startsWith(needle)) {
      candidates.push(item);
    }
    if (candidates.length >= MAX_DROPDOWN_ITEMS) {
      break;
    }
  }
  return candidates;
}

function renderMlDropdown() {
  if (!mlSymbolDropdown || !mlSymbolInput) return;
  mlSymbolDropdown.innerHTML = "";

  if (mlSymbolCatalog.length === 0) {
    const row = document.createElement("div");
    row.className = "dropdown-empty";
    row.textContent = "Symbol list is not loaded yet.";
    mlSymbolDropdown.appendChild(row);
    showMlDropdown();
    return;
  }

  const candidates = pickMlCandidates(mlSymbolInput.value);
  if (candidates.length === 0) {
    const row = document.createElement("div");
    row.className = "dropdown-empty";
    row.textContent = "No matching symbols";
    mlSymbolDropdown.appendChild(row);
    showMlDropdown();
    return;
  }

  for (const item of candidates) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dropdown-item";
    btn.dataset.symbol = item.symbol;
    btn.textContent = `${item.symbol} | ${item.name} (${item.exchange})`;
    mlSymbolDropdown.appendChild(btn);
  }

  showMlDropdown();
}

async function loadMlSymbolCatalog(refresh = false) {
  if (mlRefreshCatalogBtn) {
    mlRefreshCatalogBtn.disabled = true;
  }
  setMlCatalogMeta(refresh ? "Refreshing symbol catalog..." : "Loading symbol catalog...");

  try {
    const { response, result } = await fetchJson(
      refresh ? "/api/symbol-catalog?refresh=true" : "/api/symbol-catalog",
    );
    if (!response.ok) {
      setMlCatalogMeta(result.detail || "Failed to load symbol catalog");
      return;
    }

    const rawSymbols = Array.isArray(result.symbols) ? result.symbols : [];
    mlSymbolCatalog = rawSymbols.map((item) => ({
      symbol: normalizeSymbol(item?.symbol),
      name: String(item?.name || "").trim(),
      exchange: String(item?.exchange || "").trim(),
    }));

    const updatedText = result.updated_at ? `updated ${new Date(result.updated_at).toLocaleTimeString("ja-JP", { hour12: false })}` : "updated -";
    setMlCatalogMeta(`${mlSymbolCatalog.length.toLocaleString()} symbols loaded (${result.source || "unknown"}, ${updatedText})`);

    if (document.activeElement === mlSymbolInput) {
      renderMlDropdown();
    }
  } finally {
    if (mlRefreshCatalogBtn) {
      mlRefreshCatalogBtn.disabled = false;
    }
  }
}

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json().catch(() => ({}));
  return { response, result };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getOrCreateProgressElements() {
  let wrap = progressWrapEl || document.getElementById("ml-progress-wrap");
  let bar = progressBarEl || document.getElementById("ml-progress-bar");
  let text = progressTextEl || document.getElementById("ml-progress-text");

  if (wrap && bar && text) {
    return { wrap, bar, text };
  }
  if (!statusEl || !statusEl.parentElement) {
    return null;
  }

  wrap = document.createElement("div");
  wrap.id = "ml-progress-wrap";
  wrap.className = "ml-progress";
  wrap.setAttribute("aria-live", "polite");

  const track = document.createElement("div");
  track.className = "ml-progress-track";
  bar = document.createElement("div");
  bar.id = "ml-progress-bar";
  bar.className = "ml-progress-bar";
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuemax", "100");
  bar.setAttribute("aria-valuenow", "0");
  bar.style.width = "0%";
  track.appendChild(bar);

  text = document.createElement("p");
  text.id = "ml-progress-text";
  text.className = "hint";
  text.textContent = "0%";

  wrap.appendChild(track);
  wrap.appendChild(text);
  statusEl.parentElement.insertBefore(wrap, statusEl.nextSibling);
  return { wrap, bar, text };
}

function updateProgress(value, message = "") {
  const els = getOrCreateProgressElements();
  if (!els) return;
  const { wrap, bar, text } = els;
  const safe = Math.max(0, Math.min(100, Number(value) || 0));
  wrap.classList.remove("hidden");
  bar.style.width = `${safe}%`;
  bar.setAttribute("aria-valuenow", String(Math.round(safe)));
  text.textContent = message ? `${Math.round(safe)}% | ${message}` : `${Math.round(safe)}%`;
}

function hideProgress() {
  const els = getOrCreateProgressElements();
  if (!els) return;
  const { wrap, bar, text } = els;
  wrap.classList.add("hidden");
  bar.style.width = "0%";
  bar.setAttribute("aria-valuenow", "0");
  text.textContent = "0%";
}

function getActiveModel() {
  return mlModels.find((model) => model.id === activeModelId) || null;
}

function canRunModel(model) {
  return Boolean(model && model.status === "ready" && model.api_path);
}

function syncRunButtonState() {
  const active = getActiveModel();
  if (runBtn) {
    runBtn.textContent = active?.run_label || "Run Model";
    runBtn.disabled = isRunning || !canRunModel(active);
  }
}

function applyActiveModelUi() {
  const active = getActiveModel();
  if (!active) return;
  if (mlActiveModelPillEl) {
    mlActiveModelPillEl.textContent = active.name;
  }
  if (mlConfigTitleEl) {
    mlConfigTitleEl.textContent = `${active.name} Config`;
  }
  if (mlModelDescriptionEl) {
    mlModelDescriptionEl.textContent = active.short_description || "";
  }
  syncRunButtonState();
}

function resetMetricCards() {
  metricPinballEl.textContent = "-";
  metricCov90El.textContent = "-";
  metricCov50El.textContent = "-";
  metricSamplesEl.textContent = "-";
  btReturnStrategyEl.textContent = "-";
  btReturnBuyholdEl.textContent = "-";
  btOutperfEl.textContent = "-";
  btCapitalStrategyEl.textContent = "-";
  btCapitalBuyholdEl.textContent = "-";
  btWindowEl.textContent = "-";
  quantileLegendEl.textContent = "テスト期間の代表日を表示します。";
  fanMetaEl.textContent = "q50（中央値）、50%帯、90%帯、実測値を表示します。";
  nextDayMetaEl.textContent = "最新終値から翌営業日の上昇/下落確率を表示します。";
}

function drawModelPlaceholders() {
  const modelName = getActiveModel()?.name || "Selected Model";
  drawPlaceholder(nextDayCanvas, `Run ${modelName} to draw next-day distribution`);
  drawPlaceholder(backtestCanvas, `Run ${modelName} to draw 60-day realized backtest`);
  drawPlaceholder(quantileCanvas, `Run ${modelName} to draw quantile curves`);
  drawPlaceholder(fanCanvas, `Run ${modelName} to draw fan chart`);
}

function renderMlModelCards() {
  if (!mlModelGridEl) return;
  mlModelGridEl.innerHTML = "";

  for (const model of mlModels) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ml-model-card";
    if (model.id === activeModelId) {
      button.classList.add("active");
    }
    if (model.status !== "ready") {
      button.classList.add("pending");
    }
    button.dataset.modelId = model.id;
    button.innerHTML =
      `<span class="ml-model-card-head">`
      + `<span class="ml-model-name">${model.name}</span>`
      + `<span class="ml-model-state">${model.status_label || model.status}</span>`
      + `</span>`
      + `<span class="ml-model-summary">${model.short_description || ""}</span>`;
    mlModelGridEl.appendChild(button);
  }
}

function activateModel(modelId) {
  if (!mlModels.some((model) => model.id === modelId)) return;
  activeModelId = modelId;
  latestPayload = null;
  resetZoom(quantileZoom);
  resetZoom(fanZoom);
  resetMetricCards();
  renderMlModelCards();
  applyActiveModelUi();
  drawModelPlaceholders();

  const active = getActiveModel();
  if (!active) return;
  if (canRunModel(active)) {
    setStatus(`${active.name} を選択しました。設定後に実行してください。`);
  } else {
    setStatus(`${active.name} は準備中です。利用可能なモデルを選択してください。`);
  }
}

function normalizeMlModels(rawModels) {
  if (!Array.isArray(rawModels)) return [];
  return rawModels
    .map((item) => {
      const id = String(item?.id || "").trim();
      if (!id) return null;
      const status = String(item?.status || "").trim().toLowerCase() === "ready" ? "ready" : "coming_soon";
      return {
        id,
        name: String(item?.name || id),
        short_description: String(item?.short_description || ""),
        status,
        status_label: String(item?.status_label || (status === "ready" ? "Ready" : "Coming Soon")),
        run_label: String(item?.run_label || `Run ${String(item?.name || id)}`),
        api_path: String(item?.api_path || ""),
      };
    })
    .filter(Boolean);
}

async function loadMlModels() {
  let models = [];
  try {
    const { response, result } = await fetchJson("/api/ml/models");
    if (response.ok) {
      models = normalizeMlModels(result?.models);
    }
  } catch (_error) {
    models = [];
  }

  if (models.length === 0) {
    models = FALLBACK_ML_MODELS.slice();
  }

  mlModels = models;
  if (!mlModels.some((model) => model.id === activeModelId)) {
    const firstReady = mlModels.find((model) => canRunModel(model));
    activeModelId = firstReady?.id || mlModels[0]?.id || "quantile_lstm";
  }
  activateModel(activeModelId);
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value, digits = 6) {
  if (!Number.isFinite(value)) return "-";
  return Number(value).toFixed(digits);
}

function formatCompact(value) {
  if (!Number.isFinite(value)) return "-";
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  });
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(canvas.clientWidth));
  if (!canvas.dataset.baseHeight) {
    const initialHeight = Number(canvas.getAttribute("height")) || 320;
    canvas.dataset.baseHeight = String(initialHeight);
  }
  const height = Number(canvas.dataset.baseHeight) || 320;
  canvas.style.height = `${height}px`;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function drawPlaceholder(canvas, text) {
  const { ctx, width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = COLORS.label;
  ctx.textAlign = "center";
  ctx.font = "14px sans-serif";
  ctx.fillText(text, width / 2, height / 2);
}

function computeBounds(seriesList) {
  const values = [];
  seriesList.forEach((series) => {
    (series || []).forEach((value) => {
      const num = Number(value);
      if (Number.isFinite(num)) values.push(num);
    });
  });

  if (values.length === 0) {
    return { min: -1, max: 1 };
  }

  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function resetZoom(zoomState) {
  zoomState.xScale = 1;
  zoomState.yScale = 1;
  zoomState.xCenterRatio = 0.5;
  zoomState.yCenterRatio = 0.5;
}

function updateZoom(zoomState, factor) {
  const nextX = clamp((zoomState.xScale || 1) * factor, zoomState.min, zoomState.max);
  const nextY = clamp((zoomState.yScale || 1) * factor, zoomState.min, zoomState.max);
  zoomState.xScale = nextX;
  zoomState.yScale = nextY;
}

function getYWindow(fullYBounds, zoomState) {
  const fullSpan = fullYBounds.max - fullYBounds.min;
  if (!Number.isFinite(fullSpan) || fullSpan <= 0) {
    return { ...fullYBounds };
  }

  const yScale = clamp(zoomState.yScale || 1, zoomState.min, zoomState.max);
  const visibleSpan = fullSpan / yScale;
  const half = visibleSpan / 2;
  const center = fullYBounds.min + (clamp(zoomState.yCenterRatio, 0, 1) * fullSpan);
  const clampedCenter = clamp(center, fullYBounds.min + half, fullYBounds.max - half);

  return {
    min: clampedCenter - half,
    max: clampedCenter + half,
  };
}

function getXWindow(total, zoomState) {
  const maxIndex = Math.max(0, total - 1);
  if (maxIndex === 0) {
    return {
      start: 0,
      end: 0,
      startIndex: 0,
      endIndex: 0,
    };
  }

  const xScale = clamp(zoomState.xScale || 1, zoomState.min, zoomState.max);
  const visibleSpan = maxIndex / xScale;
  const half = visibleSpan / 2;
  const center = clamp(zoomState.xCenterRatio, 0, 1) * maxIndex;
  const clampedCenter = clamp(center, half, maxIndex - half);

  const start = clampedCenter - half;
  const end = clampedCenter + half;

  return {
    start,
    end,
    startIndex: Math.max(0, Math.floor(start)),
    endIndex: Math.min(maxIndex, Math.ceil(end)),
  };
}

function xAt(index, xWindow, left, right) {
  const denom = Math.max(1e-9, xWindow.end - xWindow.start);
  return left + (((index - xWindow.start) / denom) * (right - left));
}

function yAt(value, min, max, top, bottom) {
  const ratio = (value - min) / (max - min);
  return bottom - (ratio * (bottom - top));
}

function integrateDensity(points) {
  if (!Array.isArray(points) || points.length < 2) return 0;
  let area = 0;
  for (let i = 1; i < points.length; i += 1) {
    const left = points[i - 1];
    const right = points[i];
    const dx = Number(right?.x) - Number(left?.x);
    if (!Number.isFinite(dx) || dx <= 0) continue;
    const dLeft = Number(left?.d);
    const dRight = Number(right?.d);
    if (!Number.isFinite(dLeft) || !Number.isFinite(dRight)) continue;
    area += 0.5 * (dLeft + dRight) * dx;
  }
  return area;
}

function interpolateQuantileAtTau(values, taus, targetTau) {
  if (!Array.isArray(values) || !Array.isArray(taus) || values.length !== taus.length || values.length === 0) {
    return Number.NaN;
  }

  const pairs = [];
  for (let i = 0; i < values.length; i += 1) {
    const x = Number(values[i]);
    const t = Number(taus[i]);
    if (!Number.isFinite(x) || !Number.isFinite(t)) continue;
    pairs.push({ x, t });
  }
  if (pairs.length === 0) return Number.NaN;

  pairs.sort((a, b) => a.t - b.t);
  if (targetTau <= pairs[0].t) return pairs[0].x;
  if (targetTau >= pairs[pairs.length - 1].t) return pairs[pairs.length - 1].x;

  for (let i = 1; i < pairs.length; i += 1) {
    const left = pairs[i - 1];
    const right = pairs[i];
    if (targetTau > right.t) continue;
    const span = right.t - left.t;
    if (span <= 1e-12) return right.x;
    const w = clamp((targetTau - left.t) / span, 0, 1);
    return left.x + ((right.x - left.x) * w);
  }

  return pairs[pairs.length - 1].x;
}

function interpolateByX(points, xTarget, yKey) {
  if (!Array.isArray(points) || points.length === 0) return Number.NaN;

  const rows = points
    .map((p) => ({ x: Number(p?.x), y: Number(p?.[yKey]) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    .sort((a, b) => a.x - b.x);
  if (rows.length === 0) return Number.NaN;

  if (xTarget <= rows[0].x) return rows[0].y;
  if (xTarget >= rows[rows.length - 1].x) return rows[rows.length - 1].y;

  for (let i = 1; i < rows.length; i += 1) {
    const left = rows[i - 1];
    const right = rows[i];
    if (xTarget > right.x) continue;
    const span = right.x - left.x;
    if (span <= 1e-12) return right.y;
    const w = clamp((xTarget - left.x) / span, 0, 1);
    return left.y + ((right.y - left.y) * w);
  }

  return rows[rows.length - 1].y;
}

function buildGaussianKernel(radius, sigma = Math.max(0.8, radius / 2)) {
  if (!Number.isFinite(radius) || radius <= 0) return [1];
  const r = Math.max(1, Math.floor(radius));
  const kernel = [];
  let sum = 0;
  for (let i = -r; i <= r; i += 1) {
    const weight = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(weight);
    sum += weight;
  }
  if (sum <= 0) return [1];
  return kernel.map((w) => w / sum);
}

function smoothSeries(values, radius) {
  if (!Array.isArray(values) || values.length === 0) return [];
  const kernel = buildGaussianKernel(radius);
  if (kernel.length === 1) return values.slice();

  const out = new Array(values.length);
  const center = Math.floor(kernel.length / 2);
  const lastIdx = values.length - 1;

  for (let i = 0; i < values.length; i += 1) {
    let weightedSum = 0;
    let weightSum = 0;
    for (let k = 0; k < kernel.length; k += 1) {
      const idx = clamp(i + (k - center), 0, lastIdx);
      const value = Number(values[idx]);
      if (!Number.isFinite(value)) continue;
      const weight = kernel[k];
      weightedSum += value * weight;
      weightSum += weight;
    }
    out[i] = weightSum > 0 ? (weightedSum / weightSum) : Number(values[i] || 0);
  }
  return out;
}

function buildSmoothPdfFromQuantiles(values, taus, normalizeArea) {
  if (!Array.isArray(values) || !Array.isArray(taus) || values.length !== taus.length || values.length < 3) {
    return [];
  }

  const pairs = [];
  for (let i = 0; i < values.length; i += 1) {
    const x = Number(values[i]);
    const t = Number(taus[i]);
    if (!Number.isFinite(x) || !Number.isFinite(t)) continue;
    pairs.push({ x, t: clamp(t, 0, 1) });
  }
  if (pairs.length < 3) return [];

  pairs.sort((a, b) => a.x - b.x);

  const dedup = [];
  for (const point of pairs) {
    const last = dedup[dedup.length - 1];
    if (!last) {
      dedup.push({ x: point.x, t: point.t });
      continue;
    }
    if (Math.abs(point.x - last.x) <= 1e-12) {
      last.t = Math.max(last.t, point.t);
      continue;
    }
    dedup.push({ x: point.x, t: point.t });
  }
  if (dedup.length < 3) return [];

  let runningTau = 0;
  for (let i = 0; i < dedup.length; i += 1) {
    runningTau = Math.max(runningTau, dedup[i].t);
    dedup[i].t = clamp(runningTau, 0, 1);
  }

  const xStart = dedup[0].x;
  const xEnd = dedup[dedup.length - 1].x;
  if (!Number.isFinite(xStart) || !Number.isFinite(xEnd) || xEnd <= xStart) return [];

  const gridCount = Math.max(160, Math.min(420, dedup.length * 2));
  const xs = new Array(gridCount);
  const cdfRaw = new Array(gridCount);

  let segIdx = 1;
  for (let i = 0; i < gridCount; i += 1) {
    const ratio = i / Math.max(1, gridCount - 1);
    const x = xStart + (ratio * (xEnd - xStart));
    xs[i] = x;

    while (segIdx < dedup.length && dedup[segIdx].x < x) {
      segIdx += 1;
    }

    if (segIdx <= 0) {
      cdfRaw[i] = dedup[0].t;
      continue;
    }
    if (segIdx >= dedup.length) {
      cdfRaw[i] = dedup[dedup.length - 1].t;
      continue;
    }

    const left = dedup[segIdx - 1];
    const right = dedup[segIdx];
    const span = Math.max(1e-12, right.x - left.x);
    const w = clamp((x - left.x) / span, 0, 1);
    cdfRaw[i] = left.t + ((right.t - left.t) * w);
  }

  const cdfSmoothed = smoothSeries(cdfRaw, 4);
  let cdfPrev = 0;
  for (let i = 0; i < cdfSmoothed.length; i += 1) {
    const clamped = clamp(Number(cdfSmoothed[i]), 0, 1);
    cdfPrev = Math.max(cdfPrev, clamped);
    cdfSmoothed[i] = cdfPrev;
  }

  const density = new Array(gridCount);
  for (let i = 0; i < gridCount; i += 1) {
    if (i === 0) {
      const dx = xs[1] - xs[0];
      density[i] = dx > 0 ? (cdfSmoothed[1] - cdfSmoothed[0]) / dx : 0;
      continue;
    }
    if (i === gridCount - 1) {
      const dx = xs[i] - xs[i - 1];
      density[i] = dx > 0 ? (cdfSmoothed[i] - cdfSmoothed[i - 1]) / dx : 0;
      continue;
    }
    const dx = xs[i + 1] - xs[i - 1];
    density[i] = dx > 0 ? (cdfSmoothed[i + 1] - cdfSmoothed[i - 1]) / dx : 0;
  }

  const densitySmoothed = smoothSeries(density.map((d) => Math.max(0, Number(d) || 0)), 3);
  let pdfPoints = xs.map((x, idx) => ({ x, d: Math.max(0, Number(densitySmoothed[idx]) || 0) }));

  if (normalizeArea) {
    const area = integrateDensity(pdfPoints);
    if (!Number.isFinite(area) || area <= 1e-12) return [];
    pdfPoints = pdfPoints.map((p) => ({ x: p.x, d: p.d / area }));
  }

  return pdfPoints;
}

function drawBaseAxes(ctx, width, height, yBounds, yFormatter, xTicks) {
  const left = 70;
  const right = width - 24;
  const top = 20;
  const bottom = height - 44;

  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  ctx.strokeStyle = COLORS.grid;
  for (let step = 1; step <= 4; step += 1) {
    const y = top + (((bottom - top) / 5) * step);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }

  ctx.fillStyle = COLORS.label;
  ctx.font = "11px sans-serif";
  ctx.textAlign = "right";
  for (let step = 0; step <= 4; step += 1) {
    const value = yBounds.min + (((yBounds.max - yBounds.min) / 4) * step);
    const y = yAt(value, yBounds.min, yBounds.max, top, bottom);
    ctx.fillText(yFormatter(value), left - 8, y + 3);
  }

  ctx.textAlign = "center";
  (xTicks || []).forEach((item) => {
    const x = left + (item.ratio * (right - left));
    ctx.fillText(item.label, x, bottom + 18);
  });

  return {
    left,
    right,
    top,
    bottom,
    plotWidth: right - left,
    plotHeight: bottom - top,
  };
}

function buildIndexTicks(labels, xWindow) {
  const maxIndex = Math.max(0, labels.length - 1);
  if (maxIndex === 0) {
    return [{ ratio: 0.5, label: labels[0] || "" }];
  }

  const leftIdx = clamp(Math.round(xWindow.start), 0, maxIndex);
  const midIdx = clamp(Math.round((xWindow.start + xWindow.end) / 2), 0, maxIndex);
  const rightIdx = clamp(Math.round(xWindow.end), 0, maxIndex);

  return [
    { ratio: 0.0, label: String(labels[leftIdx] || "") },
    { ratio: 0.5, label: String(labels[midIdx] || "") },
    { ratio: 1.0, label: String(labels[rightIdx] || "") },
  ];
}

function renderQuantileFunction() {
  if (!latestPayload?.quantile_function?.curves?.length) {
    chartViewState.quantile = null;
    drawPlaceholder(quantileCanvas, "No quantile function data");
    return;
  }

  const mode = quantileScaleEl.value;
  const curves = latestPayload.quantile_function.curves;
  const taus = latestPayload.quantile_function.taus || [];
  if (taus.length === 0) {
    chartViewState.quantile = null;
    drawPlaceholder(quantileCanvas, "No quantiles");
    return;
  }

  const series = curves.map((curve) =>
    mode === "prices" ? (curve.price_quantiles || []) : (curve.return_quantiles || [])
  );

  const fullYBounds = computeBounds(series);
  const yBounds = getYWindow(fullYBounds, quantileZoom);
  const xWindow = getXWindow(taus.length, quantileZoom);

  const { ctx, width, height } = fitCanvas(quantileCanvas);
  const chart = drawBaseAxes(
    ctx,
    width,
    height,
    yBounds,
    (value) => (mode === "prices" ? formatCompact(value) : `${(value * 100).toFixed(2)}%`),
    buildIndexTicks(taus.map((item) => Number(item).toFixed(3)), xWindow)
  );

  curves.forEach((curve, curveIdx) => {
    const line = series[curveIdx];
    if (!Array.isArray(line) || line.length !== taus.length) return;

    ctx.strokeStyle = COLORS.curves[curveIdx % COLORS.curves.length];
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let idx = xWindow.startIndex; idx <= xWindow.endIndex; idx += 1) {
      const x = xAt(idx, xWindow, chart.left, chart.right);
      const y = yAt(Number(line[idx]), yBounds.min, yBounds.max, chart.top, chart.bottom);
      if (idx === xWindow.startIndex) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  const legendLines = curves.map((curve, idx) => {
    const actual = mode === "prices" ? Number(curve.actual_price) : Number(curve.actual_return);
    const actualLabel = mode === "prices" ? formatCompact(actual) : `${(actual * 100).toFixed(2)}%`;
    return `${idx + 1}. ${curve.date} (actual: ${actualLabel})`;
  });
  quantileLegendEl.textContent = legendLines.join(" | ");

  chartViewState.quantile = {
    canvas: quantileCanvas,
    fullYBounds,
    yBounds,
    xWindow,
    totalPoints: taus.length,
    plotWidth: chart.plotWidth,
    plotHeight: chart.plotHeight,
  };
}

function fillBand(ctx, lower, upper, chart, yBounds, xWindow, color) {
  if (!Array.isArray(lower) || !Array.isArray(upper) || lower.length === 0 || lower.length !== upper.length) {
    return;
  }

  ctx.fillStyle = color;
  ctx.beginPath();

  for (let idx = xWindow.startIndex; idx <= xWindow.endIndex; idx += 1) {
    const x = xAt(idx, xWindow, chart.left, chart.right);
    const y = yAt(Number(upper[idx]), yBounds.min, yBounds.max, chart.top, chart.bottom);
    if (idx === xWindow.startIndex) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  for (let idx = xWindow.endIndex; idx >= xWindow.startIndex; idx -= 1) {
    const x = xAt(idx, xWindow, chart.left, chart.right);
    const y = yAt(Number(lower[idx]), yBounds.min, yBounds.max, chart.top, chart.bottom);
    ctx.lineTo(x, y);
  }

  ctx.closePath();
  ctx.fill();
}

function renderFanChart() {
  const fan = latestPayload?.fan_chart;
  if (!fan?.dates?.length) {
    chartViewState.fan = null;
    drawPlaceholder(fanCanvas, "No fan chart data");
    return;
  }

  const mode = fanScaleEl.value;
  const q05 = mode === "prices" ? fan.q05_prices : fan.q05_returns;
  const q25 = mode === "prices" ? fan.q25_prices : fan.q25_returns;
  const q50 = mode === "prices" ? fan.q50_prices : fan.q50_returns;
  const q75 = mode === "prices" ? fan.q75_prices : fan.q75_returns;
  const q95 = mode === "prices" ? fan.q95_prices : fan.q95_returns;
  const actual = mode === "prices" ? fan.actual_prices : fan.actual_returns;

  const fullYBounds = computeBounds([q05, q25, q50, q75, q95, actual]);
  const yBounds = getYWindow(fullYBounds, fanZoom);
  const xWindow = getXWindow(fan.dates.length, fanZoom);

  const { ctx, width, height } = fitCanvas(fanCanvas);
  const chart = drawBaseAxes(
    ctx,
    width,
    height,
    yBounds,
    (value) => (mode === "prices" ? formatCompact(value) : `${(value * 100).toFixed(2)}%`),
    buildIndexTicks(fan.dates, xWindow)
  );

  fillBand(ctx, q05, q95, chart, yBounds, xWindow, COLORS.band90);
  fillBand(ctx, q25, q75, chart, yBounds, xWindow, COLORS.band50);

  ctx.strokeStyle = COLORS.median;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let idx = xWindow.startIndex; idx <= xWindow.endIndex; idx += 1) {
    const x = xAt(idx, xWindow, chart.left, chart.right);
    const y = yAt(Number(q50[idx]), yBounds.min, yBounds.max, chart.top, chart.bottom);
    if (idx === xWindow.startIndex) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = COLORS.actual;
  for (let idx = xWindow.startIndex; idx <= xWindow.endIndex; idx += 1) {
    const x = xAt(idx, xWindow, chart.left, chart.right);
    const y = yAt(Number(actual[idx]), yBounds.min, yBounds.max, chart.top, chart.bottom);
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  fanMetaEl.textContent = `${fan.dates[0]} 〜 ${fan.dates[fan.dates.length - 1]} | points: ${fan.dates.length}`;

  chartViewState.fan = {
    canvas: fanCanvas,
    fullYBounds,
    yBounds,
    xWindow,
    totalPoints: fan.dates.length,
    plotWidth: chart.plotWidth,
    plotHeight: chart.plotHeight,
  };
}

function renderMetrics(payload) {
  const metrics = payload?.metrics || {};
  const splits = payload?.splits || {};

  metricPinballEl.textContent = formatNumber(Number(metrics.mean_pinball_loss), 6);
  metricCov90El.textContent = `${formatPercent(Number(metrics.coverage_90))} (ideal 90%)`;
  metricCov50El.textContent = `${formatPercent(Number(metrics.coverage_50))} (ideal 50%)`;

  const trainCount = Number(splits?.train?.count || 0);
  const valCount = Number(splits?.val?.count || 0);
  const testCount = Number(splits?.test?.count || 0);
  metricSamplesEl.textContent = `${trainCount} / ${valCount} / ${testCount}`;
}

function formatMoney(value) {
  if (!Number.isFinite(value)) return "-";
  return `$${Number(value).toLocaleString("en-US", { maximumFractionDigits: 2, minimumFractionDigits: 2 })}`;
}

function renderBacktest60d(payload) {
  const backtest = payload?.backtest_60d;
  if (!backtest) {
    btReturnStrategyEl.textContent = "-";
    btReturnBuyholdEl.textContent = "-";
    btOutperfEl.textContent = "-";
    btCapitalStrategyEl.textContent = "-";
    btCapitalBuyholdEl.textContent = "-";
    btWindowEl.textContent = "-";
    drawPlaceholder(backtestCanvas, "No backtest data");
    return;
  }

  const strategyReturn = Number(backtest.final_return_strategy);
  const buyHoldReturn = Number(backtest.final_return_buy_hold);
  const outperf = Number(backtest.outperformance);
  const strategyCapital = Number(backtest.final_capital_strategy);
  const buyHoldCapital = Number(backtest.final_capital_buy_hold);
  const days = Number(backtest.days);
  const from = String(backtest.from || "-");
  const to = String(backtest.to || "-");
  const avgAlloc = Number(backtest.avg_allocation_stock);
  const avgCap = Number(backtest.avg_cap_stock);
  const cappedDays = Number(backtest.capped_days);
  const safeAvgAlloc = Number.isFinite(avgAlloc) ? avgAlloc : 0;
  const safeAvgCap = Number.isFinite(avgCap) ? avgCap : 0;
  const safeCappedDays = Number.isFinite(cappedDays) ? cappedDays : 0;

  btReturnStrategyEl.textContent = `${(strategyReturn * 100).toFixed(2)}%`;
  btReturnBuyholdEl.textContent = `${(buyHoldReturn * 100).toFixed(2)}%`;
  btOutperfEl.textContent = `${(outperf * 100).toFixed(2)}%`;
  btCapitalStrategyEl.textContent = formatMoney(strategyCapital);
  btCapitalBuyholdEl.textContent = formatMoney(buyHoldCapital);
  btWindowEl.textContent =
    `${days}D (${from} → ${to}) | avg alloc ${(safeAvgAlloc * 100).toFixed(1)}% `
    + `/ avg cap ${(safeAvgCap * 100).toFixed(1)}% / capped ${safeCappedDays}D`;

  const path = Array.isArray(backtest.path) ? backtest.path : [];
  if (path.length < 2) {
    drawPlaceholder(backtestCanvas, "Not enough backtest points");
    return;
  }

  const strategySeries = path.map((row) => Number(row.strategy_capital));
  const buyHoldSeries = path.map((row) => Number(row.buy_hold_capital));
  const cashSeries = path.map((row) => Number(row.cash_capital));
  const dates = path.map((row) => String(row.date || ""));

  const all = [...strategySeries, ...buyHoldSeries, ...cashSeries].filter((v) => Number.isFinite(v));
  if (all.length === 0) {
    drawPlaceholder(backtestCanvas, "Invalid backtest values");
    return;
  }

  let yMin = Math.min(...all);
  let yMax = Math.max(...all);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  const { ctx, width, height } = fitCanvas(backtestCanvas);
  const chart = drawBaseAxes(
    ctx,
    width,
    height,
    { min: yMin, max: yMax },
    (value) => formatCompact(value),
    [
      { ratio: 0.0, label: dates[0] || "" },
      { ratio: 0.5, label: dates[Math.floor(dates.length / 2)] || "" },
      { ratio: 1.0, label: dates[dates.length - 1] || "" },
    ]
  );

  const drawLine = (series, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < series.length; i += 1) {
      const x = chart.left + ((i / Math.max(1, series.length - 1)) * chart.plotWidth);
      const y = yAt(Number(series[i]), yMin, yMax, chart.top, chart.bottom);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  drawLine(cashSeries, "rgba(160, 176, 196, 0.9)");
  drawLine(buyHoldSeries, "rgba(255, 165, 0, 0.92)");
  drawLine(strategySeries, "rgba(71, 213, 148, 0.95)");

  ctx.font = "11px sans-serif";
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(160, 176, 196, 0.95)";
  ctx.fillText("Cash", chart.left + 4, chart.top + 12);
  ctx.fillStyle = "rgba(255, 165, 0, 0.95)";
  ctx.fillText("Fixed-Shares Hold", chart.left + 64, chart.top + 12);
  ctx.fillStyle = "rgba(71, 213, 148, 0.95)";
  ctx.fillText("Strategy", chart.left + 156, chart.top + 12);
}

function renderNextDayDistribution(payload) {
  const forecast = payload?.next_day_forecast;
  if (!forecast?.taus?.length || !forecast?.return_quantiles?.length) {
    drawPlaceholder(nextDayCanvas, "No next-day distribution data");
    nextDayMetaEl.textContent = "翌営業日の予測データがありません。";
    return;
  }

  const taus = forecast.taus.map((v) => Number(v));
  const distType = nextDayDistTypeEl?.value === "cdf" ? "cdf" : "pdf";
  const mode = nextDayScaleEl?.value === "prices" ? "prices" : "returns";
  const retQ = mode === "prices"
    ? forecast.price_quantiles.map((v) => Number(v))
    : forecast.return_quantiles.map((v) => Number(v));
  const currentClose = Number(forecast.current_close);
  const q05Price = mode === "prices" ? Number(forecast.q05_price) : Number(forecast.q05_return);
  const q50Price = mode === "prices" ? Number(forecast.q50_price) : Number(forecast.q50_return);
  const q95Price = mode === "prices" ? Number(forecast.q95_price) : Number(forecast.q95_return);
  const upProb = Number(forecast.up_probability);
  const downProb = Number(forecast.down_probability);

  const { ctx, width, height } = fitCanvas(nextDayCanvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, width, height);

  const left = 56;
  const right = width - 24;
  const top = 24;
  const bottom = height - 44;
  const chartTop = top + 88;
  const plotWidth = right - left;
  const plotHeight = bottom - chartTop;

  const barY = top + 18;
  const barH = 16;
  const downW = plotWidth * Math.max(0, Math.min(1, downProb));
  const upW = plotWidth * Math.max(0, Math.min(1, upProb));

  ctx.strokeStyle = "#31455f";
  ctx.lineWidth = 1;
  ctx.strokeRect(left, barY, plotWidth, barH);

  ctx.fillStyle = "rgba(255, 93, 111, 0.38)";
  ctx.fillRect(left, barY, downW, barH);
  ctx.fillStyle = "rgba(75, 215, 148, 0.38)";
  ctx.fillRect(left + downW, barY, upW, barH);

  ctx.fillStyle = "#e8eef8";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`Down ${((downProb || 0) * 100).toFixed(1)}%`, left, barY - 6);
  ctx.textAlign = "right";
  ctx.fillText(`Up ${((upProb || 0) * 100).toFixed(1)}%`, right, barY - 6);

  const minRet = Math.min(...retQ);
  const maxRet = Math.max(...retQ);
  const pad = Math.max(0.003, (maxRet - minRet) * 0.1);
  const xMin = minRet - pad;
  const xMax = maxRet + pad;

  const xAtValue = (value) => left + (((value - xMin) / Math.max(1e-9, xMax - xMin)) * plotWidth);
  const inRangeX = (value) => Number.isFinite(value) && value >= xMin && value <= xMax;
  const markerDefs = [
    { tau: 0.10, label: "10%" },
    { tau: 0.50, label: "50%" },
    { tau: 0.90, label: "90%" },
  ];
  const quantileMarkers = markerDefs
    .map((marker) => ({
      ...marker,
      x: interpolateQuantileAtTau(retQ, taus, marker.tau),
    }))
    .filter((marker) => Number.isFinite(marker.x));
  const currentLineX = mode === "prices" ? currentClose : 0;
  const currentLineLabel = mode === "prices" ? "Current Close" : "Current (0%)";
  const cdfPoints = [];
  for (let i = 0; i < retQ.length; i += 1) {
    const x = retQ[i];
    const t = taus[i];
    if (!Number.isFinite(x) || !Number.isFinite(t)) continue;
    cdfPoints.push({ x, t: Math.max(0, Math.min(1, t)) });
  }
  cdfPoints.sort((a, b) => a.x - b.x);
  let cdfRunningTau = 0;
  for (let i = 0; i < cdfPoints.length; i += 1) {
    cdfRunningTau = Math.max(cdfRunningTau, cdfPoints[i].t);
    cdfPoints[i].t = cdfRunningTau;
  }
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath();
  ctx.moveTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.moveTo(left, chartTop);
  ctx.lineTo(left, bottom);
  ctx.stroke();

  if (distType === "pdf") {
    const pdfPoints = buildSmoothPdfFromQuantiles(retQ, taus, mode === "prices");
    if (pdfPoints.length < 2) {
      drawPlaceholder(nextDayCanvas, "Not enough points for smooth probability density");
      return;
    }

    const maxDensity = Math.max(...pdfPoints.map((p) => p.d), 1e-9);
    const yMaxDensity = maxDensity * 1.15;
    const yAtDensity = (density) => bottom - ((density / Math.max(1e-9, yMaxDensity)) * plotHeight);

    ctx.fillStyle = COLORS.label;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("0", left - 8, bottom + 3);
    ctx.fillText(yMaxDensity.toFixed(1), left - 8, chartTop + 3);

    ctx.fillStyle = "rgba(33, 182, 255, 0.16)";
    ctx.beginPath();
    pdfPoints.forEach((point, idx) => {
      const x = xAtValue(point.x);
      const y = yAtDensity(point.d);
      if (idx === 0) {
        ctx.moveTo(x, bottom);
        ctx.lineTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.lineTo(xAtValue(pdfPoints[pdfPoints.length - 1].x), bottom);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "rgba(33, 182, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < pdfPoints.length; i += 1) {
      const x = xAtValue(pdfPoints[i].x);
      const y = yAtDensity(pdfPoints[i].d);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (Number.isFinite(currentLineX)) {
      const x = xAtValue(clamp(currentLineX, xMin, xMax));
      ctx.strokeStyle = "rgba(255, 200, 87, 0.95)";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x, chartTop);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.fillStyle = "rgba(255, 220, 136, 0.95)";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(currentLineLabel, Math.min(x + 4, right - 56), chartTop - 8);
    }

    ctx.setLineDash([4, 3]);
    ctx.textAlign = "center";
    quantileMarkers.forEach((marker) => {
      if (!inRangeX(marker.x)) return;
      const densityAt = interpolateByX(pdfPoints, marker.x, "d");
      if (!Number.isFinite(densityAt)) return;
      const x = xAtValue(marker.x);
      const y = yAtDensity(densityAt);
      ctx.strokeStyle = "rgba(154, 173, 196, 0.7)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, chartTop);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#cfe0f6";
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(marker.label, x, chartTop - 8);
      ctx.setLineDash([4, 3]);
    });
    ctx.setLineDash([]);
  } else {
    if (cdfPoints.length < 2) {
      drawPlaceholder(nextDayCanvas, "Not enough points for cumulative distribution");
      return;
    }

    const yAtCdf = (prob) => bottom - (Math.max(0, Math.min(1, prob)) * plotHeight);
    ctx.fillStyle = COLORS.label;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("0", left - 8, bottom + 3);
    ctx.fillText("1.0", left - 8, chartTop + 3);

    ctx.strokeStyle = "rgba(33, 182, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < cdfPoints.length; i += 1) {
      const x = xAtValue(cdfPoints[i].x);
      const y = yAtCdf(cdfPoints[i].t);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (Number.isFinite(currentLineX)) {
      const x = xAtValue(clamp(currentLineX, xMin, xMax));
      ctx.strokeStyle = "rgba(255, 200, 87, 0.95)";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x, chartTop);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.fillStyle = "rgba(255, 220, 136, 0.95)";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(currentLineLabel, Math.min(x + 4, right - 56), chartTop - 8);
    }

    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.setLineDash([4, 3]);
    quantileMarkers.forEach((marker) => {
      if (!inRangeX(marker.x)) return;
      const x = xAtValue(marker.x);
      ctx.strokeStyle = "rgba(154, 173, 196, 0.65)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, chartTop);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      const y = yAtCdf(marker.tau);
      ctx.setLineDash([]);
      ctx.fillStyle = "#cfe0f6";
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(marker.label, x, chartTop - 8);
      ctx.setLineDash([4, 3]);
    });
    ctx.setLineDash([]);
  }

  ctx.fillStyle = COLORS.label;
  ctx.textAlign = "left";
  ctx.fillText(
    mode === "prices" ? formatCompact(xMin) : `${(xMin * 100).toFixed(2)}%`,
    left,
    bottom + 16
  );
  ctx.textAlign = "center";
  ctx.fillText(
    mode === "prices"
      ? `Price (next business day) / ${distType.toUpperCase()}`
      : `Return (next business day) / ${distType.toUpperCase()}`,
    (left + right) / 2,
    bottom + 16
  );
  ctx.textAlign = "right";
  ctx.fillText(
    mode === "prices" ? formatCompact(xMax) : `${(xMax * 100).toFixed(2)}%`,
    right,
    bottom + 16
  );

  nextDayMetaEl.textContent =
    `${forecast.as_of_date} close: ${formatCompact(currentClose)} → ${forecast.target_date} `
    + `| median: ${mode === "prices" ? formatCompact(q50Price) : `${(q50Price * 100).toFixed(2)}%`} `
    + `| 90% range: ${mode === "prices"
      ? `${formatCompact(q05Price)} - ${formatCompact(q95Price)}`
      : `${(q05Price * 100).toFixed(2)}% - ${(q95Price * 100).toFixed(2)}%`}`;
}

function parseErrorMessage(body, fallback) {
  if (!body) return fallback;
  if (typeof body.detail === "string" && body.detail.trim()) return body.detail;
  if (typeof body.message === "string" && body.message.trim()) return body.message;
  if (typeof body.error === "string" && body.error.trim()) return body.error;
  return fallback;
}

async function runModelForecast() {
  const activeModel = getActiveModel();
  if (!canRunModel(activeModel)) {
    setStatus("このモデルはまだ実行できません。Ready のモデルを選択してください。");
    return;
  }

  const symbol = normalizeSymbol(mlSymbolInput?.value || "");
  if (!symbol) {
    setStatus("Symbolを入力してください。", true);
    return;
  }

  const requestPayload = {
    symbol,
    years: Number(document.getElementById("ml-years").value || "5"),
    sequence_length: Number(document.getElementById("ml-seq-len").value || "60"),
    hidden_size: Number(document.getElementById("ml-hidden-size").value || "64"),
    num_layers: Number(document.getElementById("ml-num-layers").value || "2"),
    dropout: Number(document.getElementById("ml-dropout").value || "0.2"),
    max_epochs: Number(document.getElementById("ml-max-epochs").value || "80"),
    patience: Number(document.getElementById("ml-patience").value || "10"),
    refresh: Boolean(document.getElementById("ml-refresh").checked),
  };

  isRunning = true;
  syncRunButtonState();
  setStatus(`${activeModel.name} の学習と推論を実行中です...`);
  updateProgress(0, "ジョブを開始しています。");

  try {
    const { response: createRes, result: createBody } = await fetchJson(
      "/api/ml/quantile-lstm/jobs",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestPayload),
      },
    );
    if (!createRes.ok) {
      throw new Error(parseErrorMessage(createBody, "ジョブ開始に失敗しました。"));
    }

    const jobId = String(createBody.job_id || "").trim();
    if (!jobId) {
      throw new Error("ジョブIDを取得できませんでした。");
    }

    const maxPolls = 60 * 30;
    let body = null;
    for (let pollCount = 0; pollCount < maxPolls; pollCount += 1) {
      const { response: statusRes, result: statusBody } = await fetchJson(`/api/ml/jobs/${encodeURIComponent(jobId)}`);
      if (!statusRes.ok) {
        throw new Error(parseErrorMessage(statusBody, "ジョブ状態の取得に失敗しました。"));
      }

      updateProgress(Number(statusBody.progress || 0), String(statusBody.message || ""));

      if (statusBody.status === "completed") {
        body = statusBody.result;
        updateProgress(100, "完了しました。");
        break;
      }
      if (statusBody.status === "failed") {
        throw new Error(parseErrorMessage(statusBody, "予測実行に失敗しました。"));
      }

      await sleep(900);
    }

    if (!body) {
      throw new Error("タイムアウト: 学習が長時間終了しませんでした。");
    }

    latestPayload = body;
    resetZoom(quantileZoom);
    resetZoom(fanZoom);
    renderMetrics(body);
    renderBacktest60d(body);
    renderNextDayDistribution(body);
    renderQuantileFunction();
    renderFanChart();
    const epochs = Number(body?.training?.epochs_trained || 0);
    const valLoss = Number(body?.training?.best_val_pinball_loss || 0);
    setStatus(`完了: ${activeModel.name} ${symbol} | epochs=${epochs}, best val pinball=${formatNumber(valLoss, 6)}`);
  } catch (error) {
    updateProgress(Number(progressBarEl?.getAttribute("aria-valuenow") || 0), "失敗しました。");
    setStatus(error instanceof Error ? error.message : "予測実行に失敗しました。", true);
  } finally {
    isRunning = false;
    syncRunButtonState();
  }
}

function setupDragPan(canvas, zoomState, getView, rerender) {
  const drag = {
    active: false,
    startX: 0,
    startY: 0,
    startXCenterRatio: 0.5,
    startYCenterRatio: 0.5,
    view: null,
  };

  canvas.addEventListener("mousedown", (event) => {
    if (!latestPayload) return;
    const view = getView();
    if (!view) return;
    if ((zoomState.xScale <= 1) && (zoomState.yScale <= 1)) return;

    drag.active = true;
    drag.startX = event.clientX;
    drag.startY = event.clientY;
    drag.startXCenterRatio = zoomState.xCenterRatio;
    drag.startYCenterRatio = zoomState.yCenterRatio;
    drag.view = view;

    canvas.classList.add("dragging");
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    if (!drag.active || !drag.view) return;

    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    const view = drag.view;

    if (zoomState.xScale > 1 && view.totalPoints > 1) {
      const maxIndex = Math.max(1, view.totalPoints - 1);
      const visibleSpan = maxIndex / zoomState.xScale;
      const deltaIndex = (dx / Math.max(1, view.plotWidth)) * visibleSpan;
      const baseCenter = drag.startXCenterRatio * maxIndex;
      const center = baseCenter - deltaIndex;
      zoomState.xCenterRatio = clamp(center / maxIndex, 0, 1);
    }

    if (zoomState.yScale > 1) {
      const fullSpan = view.fullYBounds.max - view.fullYBounds.min;
      if (fullSpan > 0) {
        const visibleSpan = fullSpan / zoomState.yScale;
        const deltaValue = (dy / Math.max(1, view.plotHeight)) * visibleSpan;
        const baseCenter = view.fullYBounds.min + (drag.startYCenterRatio * fullSpan);
        const center = baseCenter + deltaValue;
        zoomState.yCenterRatio = clamp((center - view.fullYBounds.min) / fullSpan, 0, 1);
      }
    }

    rerender();
  });

  window.addEventListener("mouseup", () => {
    if (!drag.active) return;
    drag.active = false;
    drag.view = null;
    canvas.classList.remove("dragging");
  });

  canvas.addEventListener("mouseleave", () => {
    if (!drag.active) return;
    // continue drag outside canvas until mouseup, keep class state.
  });
}

mlForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runModelForecast();
});

if (mlModelGridEl) {
  mlModelGridEl.addEventListener("click", (event) => {
    const card = event.target.closest(".ml-model-card");
    if (!card) return;
    const modelId = card.dataset.modelId;
    if (!modelId) return;
    activateModel(modelId);
  });
}

if (mlSymbolInput) {
  mlSymbolInput.addEventListener("focus", () => {
    renderMlDropdown();
  });

  mlSymbolInput.addEventListener("input", () => {
    renderMlDropdown();
  });

  mlSymbolInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideMlDropdown();
      return;
    }
    if (event.key === "Enter") {
      const firstCandidate = mlSymbolDropdown?.querySelector(".dropdown-item");
      if (!firstCandidate) return;
      event.preventDefault();
      mlSymbolInput.value = firstCandidate.dataset.symbol || "";
      hideMlDropdown();
    }
  });
}

if (mlSymbolDropdown) {
  mlSymbolDropdown.addEventListener("mousedown", (event) => {
    event.preventDefault();
  });

  mlSymbolDropdown.addEventListener("click", (event) => {
    const button = event.target.closest(".dropdown-item");
    if (!button || !mlSymbolInput) return;
    mlSymbolInput.value = button.dataset.symbol || "";
    hideMlDropdown();
    mlSymbolInput.focus();
  });
}

if (mlSymbolSearchArea) {
  document.addEventListener("click", (event) => {
    if (!mlSymbolSearchArea.contains(event.target)) {
      hideMlDropdown();
    }
  });
}

if (mlRefreshCatalogBtn) {
  mlRefreshCatalogBtn.addEventListener("click", async () => {
    await loadMlSymbolCatalog(true);
    if (mlSymbolInput) {
      mlSymbolInput.focus();
      renderMlDropdown();
    }
  });
}

quantileScaleEl.addEventListener("change", () => {
  resetZoom(quantileZoom);
  if (latestPayload) renderQuantileFunction();
});

fanScaleEl.addEventListener("change", () => {
  resetZoom(fanZoom);
  if (latestPayload) renderFanChart();
});

if (nextDayScaleEl) {
  nextDayScaleEl.addEventListener("change", () => {
    if (latestPayload) renderNextDayDistribution(latestPayload);
  });
}

if (nextDayDistTypeEl) {
  nextDayDistTypeEl.addEventListener("change", () => {
    if (latestPayload) renderNextDayDistribution(latestPayload);
  });
}

quantileZoomInBtn.addEventListener("click", () => {
  updateZoom(quantileZoom, 1.25);
  if (latestPayload) renderQuantileFunction();
});

quantileZoomOutBtn.addEventListener("click", () => {
  updateZoom(quantileZoom, 1 / 1.25);
  if (latestPayload) renderQuantileFunction();
});

quantileZoomResetBtn.addEventListener("click", () => {
  resetZoom(quantileZoom);
  if (latestPayload) renderQuantileFunction();
});

fanZoomInBtn.addEventListener("click", () => {
  updateZoom(fanZoom, 1.25);
  if (latestPayload) renderFanChart();
});

fanZoomOutBtn.addEventListener("click", () => {
  updateZoom(fanZoom, 1 / 1.25);
  if (latestPayload) renderFanChart();
});

fanZoomResetBtn.addEventListener("click", () => {
  resetZoom(fanZoom);
  if (latestPayload) renderFanChart();
});

quantileCanvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    updateZoom(quantileZoom, event.deltaY < 0 ? 1.12 : 1 / 1.12);
    if (latestPayload) renderQuantileFunction();
  },
  { passive: false }
);

fanCanvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    updateZoom(fanZoom, event.deltaY < 0 ? 1.12 : 1 / 1.12);
    if (latestPayload) renderFanChart();
  },
  { passive: false }
);

setupDragPan(
  quantileCanvas,
  quantileZoom,
  () => chartViewState.quantile,
  () => {
    if (latestPayload) renderQuantileFunction();
  }
);

setupDragPan(
  fanCanvas,
  fanZoom,
  () => chartViewState.fan,
  () => {
    if (latestPayload) renderFanChart();
  }
);

window.addEventListener("resize", () => {
  if (!latestPayload) return;
  renderBacktest60d(latestPayload);
  renderNextDayDistribution(latestPayload);
  renderQuantileFunction();
  renderFanChart();
});

resetMetricCards();
drawModelPlaceholders();
hideProgress();

loadMlSymbolCatalog().catch(() => {
  setMlCatalogMeta("Failed to load symbol catalog");
});

loadMlModels().catch(() => {
  mlModels = FALLBACK_ML_MODELS.slice();
  activateModel("quantile_lstm");
});
