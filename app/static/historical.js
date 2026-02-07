const HISTORICAL_YEARS = 5;
const CHART_HEIGHT = 320;
const ZOOM_MIN_POINTS = 20;
const ZOOM_IN_FACTOR = 0.8;
const ZOOM_OUT_FACTOR = 1.25;

const symbolTitleEl = document.getElementById("symbol-title");
const symbolSubtitleEl = document.getElementById("symbol-subtitle");
const refreshHistoryBtn = document.getElementById("refresh-history");
const refreshCreditsBtn = document.getElementById("refresh-credits");
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const resetZoomBtn = document.getElementById("reset-zoom");
const creditsLeftEl = document.getElementById("credits-left");
const creditsUpdatedEl = document.getElementById("credits-updated");
const historySourceEl = document.getElementById("history-source");
const historyRangeEl = document.getElementById("history-range");
const historyCountEl = document.getElementById("history-count");
const historyMetaEl = document.getElementById("history-meta");
const historyCanvas = document.getElementById("history-canvas");
const historyTooltipEl = document.getElementById("history-tooltip");

let currentSymbol = "";
let currentPayload = null;
let isDragging = false;
let dragStartX = 0;
let dragViewportStart = 0;
let dragViewportEnd = 0;

const chartState = {
  points: [],
  viewportStart: 0,
  viewportEnd: 0,
  hoveredIndex: null,
  renderInfo: null,
};

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleTimeString("ja-JP", { hour12: false });
}

function formatPrice(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "-";
  }
  return num.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

function setHistoryMeta(message, isError = false) {
  historyMetaEl.textContent = message || "";
  historyMetaEl.classList.toggle("error", Boolean(isError));
}

function setCredits(status) {
  const dailyLeft = status?.daily_credits_left;
  const dailyLimit = status?.daily_credits_limit;
  const isEstimated = Boolean(status?.daily_credits_is_estimated);
  if (dailyLeft === null || dailyLeft === undefined) {
    creditsLeftEl.textContent = "-";
  } else if (dailyLimit === null || dailyLimit === undefined) {
    creditsLeftEl.textContent = `${dailyLeft}${isEstimated ? " (est)" : ""}`;
  } else {
    creditsLeftEl.textContent = `${dailyLeft} / ${dailyLimit}${isEstimated ? " (est)" : ""}`;
  }
  creditsUpdatedEl.textContent = formatTime(status?.daily_credits_updated_at);
}

function fitCanvas(canvas, cssHeight = CHART_HEIGHT) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = Math.max(320, Math.floor(canvas.clientWidth));
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: cssWidth, height: cssHeight };
}

function hideTooltip() {
  historyTooltipEl.textContent = "";
  historyTooltipEl.classList.add("hidden");
}

function showTooltip(index, x, y, chartWidth, chartHeight) {
  const point = chartState.points[index];
  if (!point) {
    hideTooltip();
    return;
  }

  const text = `${point.t}\n$${formatPrice(point.c)}`;
  historyTooltipEl.textContent = text;

  const tooltipWidth = 150;
  const tooltipHeight = 48;
  let left = x + 14;
  if (left + tooltipWidth > chartWidth - 8) {
    left = x - tooltipWidth - 14;
  }
  left = Math.max(8, left);

  let top = y - tooltipHeight - 8;
  if (top < 8) {
    top = y + 8;
  }
  top = Math.max(8, Math.min(top, chartHeight - tooltipHeight - 8));

  historyTooltipEl.style.left = `${left}px`;
  historyTooltipEl.style.top = `${top}px`;
  historyTooltipEl.classList.remove("hidden");
}

function drawPlaceholder(message) {
  const { ctx, width, height } = fitCanvas(historyCanvas, CHART_HEIGHT);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#eef3ef";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#66737d";
  ctx.font = "14px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(message, width / 2, height / 2);
  chartState.renderInfo = null;
  hideTooltip();
}

function resetViewport() {
  if (chartState.points.length < 2) {
    return;
  }
  chartState.viewportStart = 0;
  chartState.viewportEnd = chartState.points.length - 1;
}

function clampViewport(start, end) {
  const maxIndex = chartState.points.length - 1;
  if (maxIndex <= 0) {
    return { start: 0, end: 0 };
  }

  let clampedStart = start;
  let clampedEnd = end;
  if (clampedEnd < clampedStart) {
    const tmp = clampedStart;
    clampedStart = clampedEnd;
    clampedEnd = tmp;
  }

  const minSpan = Math.max(1, Math.min(ZOOM_MIN_POINTS - 1, maxIndex));
  const fullSpan = maxIndex;
  let span = clampedEnd - clampedStart;
  if (span < minSpan) {
    const center = (clampedStart + clampedEnd) / 2;
    clampedStart = center - (minSpan / 2);
    clampedEnd = center + (minSpan / 2);
    span = minSpan;
  }

  if (span >= fullSpan) {
    return { start: 0, end: maxIndex };
  }

  if (clampedStart < 0) {
    clampedEnd -= clampedStart;
    clampedStart = 0;
  }
  if (clampedEnd > maxIndex) {
    clampedStart -= (clampedEnd - maxIndex);
    clampedEnd = maxIndex;
  }

  clampedStart = Math.max(0, clampedStart);
  clampedEnd = Math.min(maxIndex, clampedEnd);
  return { start: clampedStart, end: clampedEnd };
}

function drawChartFromState() {
  const points = chartState.points;
  if (points.length < 2) {
    drawPlaceholder("Not enough data points");
    return;
  }

  const viewport = clampViewport(chartState.viewportStart, chartState.viewportEnd);
  chartState.viewportStart = viewport.start;
  chartState.viewportEnd = viewport.end;

  const { ctx, width, height } = fitCanvas(historyCanvas, CHART_HEIGHT);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfa";
  ctx.fillRect(0, 0, width, height);

  const left = 64;
  const right = width - 24;
  const top = 18;
  const bottom = height - 46;
  const plotWidth = right - left;
  const plotHeight = bottom - top;
  const viewSpan = Math.max(1, chartState.viewportEnd - chartState.viewportStart);

  const visibleStart = Math.max(0, Math.floor(chartState.viewportStart));
  const visibleEnd = Math.min(points.length - 1, Math.ceil(chartState.viewportEnd));

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let index = visibleStart; index <= visibleEnd; index += 1) {
    const value = Number(points[index]?.c);
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    drawPlaceholder("Invalid data points");
    return;
  }
  if (min === max) {
    min -= 1;
    max += 1;
  }

  const yScale = plotHeight / (max - min);
  const xFromIndex = (index) => left + (((index - chartState.viewportStart) / viewSpan) * plotWidth);
  const yFromValue = (value) => bottom - ((value - min) * yScale);

  ctx.strokeStyle = "#d9e0d7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  ctx.strokeStyle = "#e8eee7";
  ctx.lineWidth = 1;
  for (let step = 1; step <= 3; step += 1) {
    const y = top + ((plotHeight / 4) * step);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#1d6f62";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  for (let index = visibleStart; index <= visibleEnd; index += 1) {
    const value = Number(points[index]?.c);
    if (!Number.isFinite(value)) {
      continue;
    }
    const x = xFromIndex(index);
    const y = yFromValue(value);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  if (started) {
    ctx.stroke();
  }

  ctx.fillStyle = "#55626f";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(formatPrice(max), 8, top + 6);
  ctx.fillText(formatPrice(min), 8, bottom);

  const startLabelIndex = Math.max(0, Math.min(points.length - 1, Math.round(chartState.viewportStart)));
  const endLabelIndex = Math.max(0, Math.min(points.length - 1, Math.round(chartState.viewportEnd)));
  ctx.textAlign = "left";
  ctx.fillText(String(points[startLabelIndex]?.t || "-"), left, height - 16);
  ctx.textAlign = "right";
  ctx.fillText(String(points[endLabelIndex]?.t || "-"), right, height - 16);

  const hoverIndex = chartState.hoveredIndex;
  if (
    Number.isInteger(hoverIndex)
    && hoverIndex >= visibleStart
    && hoverIndex <= visibleEnd
  ) {
    const hoverValue = Number(points[hoverIndex]?.c);
    if (Number.isFinite(hoverValue)) {
      const hoverX = xFromIndex(hoverIndex);
      const hoverY = yFromValue(hoverValue);

      ctx.strokeStyle = "#6b8793";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX, top);
      ctx.lineTo(hoverX, bottom);
      ctx.stroke();

      ctx.fillStyle = "#1d6f62";
      ctx.beginPath();
      ctx.arc(hoverX, hoverY, 4, 0, Math.PI * 2);
      ctx.fill();

      showTooltip(hoverIndex, hoverX, hoverY, width, height);
    } else {
      hideTooltip();
    }
  } else {
    hideTooltip();
  }

  chartState.renderInfo = {
    left,
    right,
    top,
    bottom,
    plotWidth,
    viewStart: chartState.viewportStart,
    viewEnd: chartState.viewportEnd,
  };
}

function applyPayloadToChart(payload, resetView = false) {
  const points = Array.isArray(payload?.points) ? payload.points : [];
  chartState.points = points;
  if (points.length < 2) {
    drawPlaceholder("Not enough data points");
    return;
  }

  if (resetView || chartState.viewportEnd <= chartState.viewportStart || chartState.viewportEnd > points.length - 1) {
    resetViewport();
  } else {
    const viewport = clampViewport(chartState.viewportStart, chartState.viewportEnd);
    chartState.viewportStart = viewport.start;
    chartState.viewportEnd = viewport.end;
  }
  drawChartFromState();
}

function canvasXFromEvent(event) {
  const rect = historyCanvas.getBoundingClientRect();
  return event.clientX - rect.left;
}

function getIndexFromCanvasX(canvasX) {
  const render = chartState.renderInfo;
  if (!render) return null;
  if (canvasX < render.left || canvasX > render.right) return null;
  const ratio = (canvasX - render.left) / Math.max(1, render.plotWidth);
  const rawIndex = render.viewStart + (ratio * (render.viewEnd - render.viewStart));
  const index = Math.round(rawIndex);
  return Math.max(0, Math.min(chartState.points.length - 1, index));
}

function updateHoverFromCanvasX(canvasX) {
  const index = getIndexFromCanvasX(canvasX);
  if (index === null) {
    if (chartState.hoveredIndex !== null) {
      chartState.hoveredIndex = null;
      drawChartFromState();
    }
    return;
  }
  if (chartState.hoveredIndex !== index) {
    chartState.hoveredIndex = index;
    drawChartFromState();
  }
}

function zoomAtCanvasX(canvasX, factor) {
  const points = chartState.points;
  if (points.length < 2 || !chartState.renderInfo) return;

  const maxIndex = points.length - 1;
  const fullSpan = maxIndex;
  const minSpan = Math.max(1, Math.min(ZOOM_MIN_POINTS - 1, fullSpan));
  const currentSpan = Math.max(1, chartState.viewportEnd - chartState.viewportStart);
  let targetSpan = currentSpan * factor;
  targetSpan = Math.max(minSpan, Math.min(fullSpan, targetSpan));
  if (Math.abs(targetSpan - currentSpan) < 0.001) {
    return;
  }

  const centerIndex = getIndexFromCanvasX(canvasX) ?? Math.round((chartState.viewportStart + chartState.viewportEnd) / 2);
  const ratio = (centerIndex - chartState.viewportStart) / currentSpan;
  let nextStart = centerIndex - (targetSpan * ratio);
  let nextEnd = nextStart + targetSpan;

  const viewport = clampViewport(nextStart, nextEnd);
  chartState.viewportStart = viewport.start;
  chartState.viewportEnd = viewport.end;
  drawChartFromState();
}

function zoomWithFactor(factor) {
  const render = chartState.renderInfo;
  if (!render) return;
  const centerX = render.left + (render.plotWidth / 2);
  zoomAtCanvasX(centerX, factor);
}

function resetZoom() {
  if (chartState.points.length < 2) return;
  chartState.hoveredIndex = null;
  resetViewport();
  drawChartFromState();
}

function startDrag(canvasX) {
  if (!chartState.renderInfo) return;
  isDragging = true;
  dragStartX = canvasX;
  dragViewportStart = chartState.viewportStart;
  dragViewportEnd = chartState.viewportEnd;
  historyCanvas.classList.add("dragging");
  hideTooltip();
}

function dragTo(canvasX) {
  if (!isDragging || !chartState.renderInfo) return;
  const span = Math.max(1, dragViewportEnd - dragViewportStart);
  const deltaX = canvasX - dragStartX;
  const deltaIndex = (deltaX / Math.max(1, chartState.renderInfo.plotWidth)) * span;
  const viewport = clampViewport(dragViewportStart - deltaIndex, dragViewportEnd - deltaIndex);
  chartState.viewportStart = viewport.start;
  chartState.viewportEnd = viewport.end;
  chartState.hoveredIndex = null;
  drawChartFromState();
}

function endDrag() {
  if (!isDragging) return;
  isDragging = false;
  historyCanvas.classList.remove("dragging");
}

async function loadCredits(refresh = false) {
  const response = await fetch(refresh ? "/api/credits?refresh=true" : "/api/credits");
  const result = await response.json().catch(() => ({}));
  if (!response.ok || !result.status) {
    return;
  }
  setCredits(result.status);
}

async function loadHistorical(refresh = false) {
  if (!currentSymbol) {
    return;
  }

  refreshHistoryBtn.disabled = true;
  setHistoryMeta("Loading historical data...");
  drawPlaceholder("Loading...");

  try {
    const url = `/api/historical/${encodeURIComponent(currentSymbol)}?years=${HISTORICAL_YEARS}${refresh ? "&refresh=true" : ""}`;
    const response = await fetch(url);
    const result = await response.json().catch(() => ({}));

    if (!response.ok || !result.ok) {
      setHistoryMeta(result.detail || "Failed to load historical data", true);
      drawPlaceholder("Historical data unavailable");
      return;
    }

    currentPayload = result;
    historySourceEl.textContent = result.source || "-";
    historyRangeEl.textContent = `${result.from} - ${result.to}`;
    historyCountEl.textContent = String(result.count ?? "-");
    symbolSubtitleEl.textContent = `${result.years}Y ${result.interval} (${result.from} - ${result.to})`;
    setHistoryMeta("Loaded historical data.");
    chartState.hoveredIndex = null;
    applyPayloadToChart(result, true);
    await loadCredits(false);
  } finally {
    refreshHistoryBtn.disabled = false;
  }
}

function getSymbolFromPath() {
  const path = window.location.pathname;
  const parts = path.split("/").filter(Boolean);
  if (parts.length < 2) return "";
  return decodeURIComponent(parts[1] || "").trim().toUpperCase();
}

refreshHistoryBtn.addEventListener("click", async () => {
  await loadHistorical(true);
});

refreshCreditsBtn.addEventListener("click", async () => {
  refreshCreditsBtn.disabled = true;
  try {
    await loadCredits(true);
  } finally {
    refreshCreditsBtn.disabled = false;
  }
});

zoomInBtn.addEventListener("click", () => {
  zoomWithFactor(ZOOM_IN_FACTOR);
});

zoomOutBtn.addEventListener("click", () => {
  zoomWithFactor(ZOOM_OUT_FACTOR);
});

resetZoomBtn.addEventListener("click", () => {
  resetZoom();
});

historyCanvas.addEventListener(
  "wheel",
  (event) => {
    if (!chartState.renderInfo) return;
    event.preventDefault();
    const canvasX = canvasXFromEvent(event);
    const factor = event.deltaY < 0 ? ZOOM_IN_FACTOR : ZOOM_OUT_FACTOR;
    zoomAtCanvasX(canvasX, factor);
  },
  { passive: false }
);

historyCanvas.addEventListener("mousedown", (event) => {
  if (event.button !== 0) return;
  event.preventDefault();
  startDrag(canvasXFromEvent(event));
});

window.addEventListener("mousemove", (event) => {
  if (!isDragging) return;
  const canvasX = canvasXFromEvent(event);
  dragTo(canvasX);
});

historyCanvas.addEventListener("mousemove", (event) => {
  if (isDragging) return;
  updateHoverFromCanvasX(canvasXFromEvent(event));
});

window.addEventListener("mouseup", () => {
  endDrag();
});

historyCanvas.addEventListener("mouseleave", () => {
  if (isDragging) return;
  chartState.hoveredIndex = null;
  drawChartFromState();
});

historyCanvas.addEventListener("dblclick", () => {
  resetZoom();
});

window.addEventListener("resize", () => {
  if (currentPayload) {
    applyPayloadToChart(currentPayload, false);
  } else {
    drawPlaceholder("Loading...");
  }
});

async function init() {
  currentSymbol = getSymbolFromPath();
  if (!currentSymbol) {
    symbolTitleEl.textContent = "Historical Data";
    symbolSubtitleEl.textContent = "Symbol not found";
    setHistoryMeta("Invalid symbol", true);
    drawPlaceholder("Invalid symbol");
    return;
  }

  symbolTitleEl.textContent = `${currentSymbol} Historical Data`;
  symbolSubtitleEl.textContent = `Loading ${HISTORICAL_YEARS}Y history...`;
  drawPlaceholder("Loading...");

  await loadCredits(false);
  await loadHistorical(false);
}

void init();
