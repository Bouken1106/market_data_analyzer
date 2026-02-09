const CHART_HEIGHT = 360;
const ZOOM_MIN_POINTS = 20;
const ZOOM_IN_FACTOR = 0.8;
const ZOOM_OUT_FACTOR = 1.25;
const INTERVAL_KEYS = ["1min", "5min", "1day"];

const CHART_COLORS = {
  placeholderText: "#8fa1bb",
  canvasBg: "#0a121f",
  canvasTop: "#0b1425",
  canvasBottom: "#050a14",
  gloss: "rgba(154, 221, 255, 0.04)",
  axis: "#324158",
  grid: "rgba(45, 66, 92, 0.42)",
  line: "#46deff",
  lineGlow: "rgba(70, 222, 255, 0.75)",
  lineArea: "rgba(19, 199, 255, 0.14)",
  vwap: "#74f2a6",
  volume: "rgba(109, 161, 224, 0.7)",
  label: "#9cadc4",
  crosshair: "#4f6e8d",
  point: "#13c7ff",
};

const symbolTitleEl = document.getElementById("symbol-title");
const symbolSubtitleEl = document.getElementById("symbol-subtitle");
const backBtn = document.getElementById("go-back");
const refreshHistoryBtn = document.getElementById("refresh-history");
const clearHistoryCacheBtn = document.getElementById("clear-history-cache");
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const resetZoomBtn = document.getElementById("reset-zoom");
const interval1mBtn = document.getElementById("interval-1m");
const interval5mBtn = document.getElementById("interval-5m");
const interval1dBtn = document.getElementById("interval-1d");
const rangeMaxBtn = document.getElementById("range-max");
const range10yBtn = document.getElementById("range-10y");
const range5yBtn = document.getElementById("range-5y");
const range1yBtn = document.getElementById("range-1y");
const rangeYtdBtn = document.getElementById("range-ytd");
const currentPriceEl = document.getElementById("current-price");
const riskVol30El = document.getElementById("risk-vol30");
const riskDd30El = document.getElementById("risk-dd30");
const riskVar95El = document.getElementById("risk-var95");
const techAtrEl = document.getElementById("tech-atr");
const priceGapEl = document.getElementById("price-gap");
const historyMetaEl = document.getElementById("history-meta");
const historyCanvas = document.getElementById("history-canvas");
const historyTooltipEl = document.getElementById("history-tooltip");

const ovSymbolNameEl = document.getElementById("ov-symbol-name");
const ovCurrentEl = document.getElementById("ov-current");
const ovChangeEl = document.getElementById("ov-change");
const ovDayRangeEl = document.getElementById("ov-day-range");
const ovVolumeEl = document.getElementById("ov-volume");
const ovTurnoverEl = document.getElementById("ov-turnover");
const ovSpreadEl = document.getElementById("ov-spread");
const ovUpdatedEl = document.getElementById("ov-updated");

const techVwapEl = document.getElementById("tech-vwap");
const techMaEl = document.getElementById("tech-ma");
const techBetaEl = document.getElementById("tech-beta");
const techCorrEl = document.getElementById("tech-corr");
const marketSpyEl = document.getElementById("market-spy");
const marketQqqEl = document.getElementById("market-qqq");
const loadQqqBtn = document.getElementById("load-qqq");
const supportSectorEl = document.getElementById("support-sector");
const supportBoardEl = document.getElementById("support-board");
const supportEventsEl = document.getElementById("support-events");
const supportCorporateEl = document.getElementById("support-corporate");
const supportNewsEl = document.getElementById("support-news");

let currentSymbol = "";
let currentPayload = null;
let isDragging = false;
let dragStartX = 0;
let dragViewportStart = 0;
let dragViewportEnd = 0;
let activeInterval = "1day";

const chartSeriesByInterval = {
  "1min": [],
  "5min": [],
  "1day": [],
};

const chartState = {
  points: [],
  viewportStart: 0,
  viewportEnd: 0,
  hoveredIndex: null,
  renderInfo: null,
};

function formatPrice(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

function formatCompact(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(num);
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatSignedPrice(value) {
  if (!Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatPrice(value)}`;
}

function formatIsoTime(value) {
  if (!value) return "-";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return String(value);
  return dt.toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function setHistoryMeta(message, isError = false) {
  historyMetaEl.textContent = message || "";
  historyMetaEl.classList.toggle("error", Boolean(isError));
}

function setCurrentPrice(value) {
  const num = Number(value);
  currentPriceEl.textContent = Number.isFinite(num) ? `$${formatPrice(num)}` : "-";
}

function quantile(sortedValues, q) {
  if (!Array.isArray(sortedValues) || sortedValues.length === 0) return null;
  const clamped = Math.min(1, Math.max(0, q));
  const pos = (sortedValues.length - 1) * clamped;
  const low = Math.floor(pos);
  const high = Math.ceil(pos);
  if (low === high) return sortedValues[low];
  const weight = pos - low;
  return sortedValues[low] * (1 - weight) + (sortedValues[high] * weight);
}

function setRiskMetrics(dayPoints, overview) {
  const closes = (Array.isArray(dayPoints) ? dayPoints : [])
    .map((item) => Number(item?.c))
    .filter((num) => Number.isFinite(num));

  if (closes.length < 3) {
    riskVol30El.textContent = "-";
    riskDd30El.textContent = "-";
    riskVar95El.textContent = "-";
    return;
  }

  const returns = [];
  for (let index = 1; index < closes.length; index += 1) {
    const prev = closes[index - 1];
    const curr = closes[index];
    if (prev <= 0) continue;
    returns.push((curr / prev) - 1);
  }

  const recentReturns = returns.slice(-30);
  if (recentReturns.length >= 2) {
    const mean = recentReturns.reduce((acc, value) => acc + value, 0) / recentReturns.length;
    const variance = recentReturns.reduce((acc, value) => acc + ((value - mean) ** 2), 0) / (recentReturns.length - 1);
    const annualizedVol = Math.sqrt(Math.max(0, variance)) * Math.sqrt(252) * 100;
    riskVol30El.textContent = `${annualizedVol.toFixed(2)}%`;

    const sortedReturns = [...recentReturns].sort((a, b) => a - b);
    const p05 = quantile(sortedReturns, 0.05);
    const var95 = Number.isFinite(p05) ? Math.max(0, -p05 * 100) : null;
    riskVar95El.textContent = Number.isFinite(var95) ? `${var95.toFixed(2)}%` : "-";
  } else {
    riskVol30El.textContent = "-";
    riskVar95El.textContent = "-";
  }

  const closes30 = closes.slice(-30);
  if (closes30.length >= 2) {
    let peak = closes30[0];
    let maxDrawdown = 0;
    for (const close of closes30) {
      peak = Math.max(peak, close);
      if (peak <= 0) continue;
      const drawdown = ((peak - close) / peak) * 100;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
    riskDd30El.textContent = `${maxDrawdown.toFixed(2)}%`;
  } else {
    riskDd30El.textContent = "-";
  }

  const atr = Number(overview?.technical?.atr_14);
  techAtrEl.textContent = Number.isFinite(atr) ? `$${formatPrice(atr)}` : "-";
  const gapAbs = Number(overview?.price?.gap_abs);
  const gapPct = Number(overview?.price?.gap_pct);
  priceGapEl.textContent = Number.isFinite(gapAbs) && Number.isFinite(gapPct)
    ? `${formatSignedPrice(gapAbs)} (${formatPercent(gapPct)})`
    : "-";
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

function paintHistoricalBackdrop(ctx, width, height) {
  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, CHART_COLORS.canvasTop);
  bg.addColorStop(0.45, CHART_COLORS.canvasBg);
  bg.addColorStop(1, CHART_COLORS.canvasBottom);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  const glow = ctx.createRadialGradient(0, 0, 0, width * 0.22, height * 0.1, width * 0.75);
  glow.addColorStop(0, "rgba(24, 146, 255, 0.1)");
  glow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);

  const gloss = ctx.createLinearGradient(0, 0, 0, height * 0.3);
  gloss.addColorStop(0, CHART_COLORS.gloss);
  gloss.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gloss;
  ctx.fillRect(0, 0, width, height * 0.34);
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

  const close = Number(point.c);
  const volume = Number(point.v);
  const text = `${point.t}\nClose: $${formatPrice(close)}\nVol: ${formatCompact(volume)}`;
  historyTooltipEl.textContent = text;

  const tooltipWidth = 170;
  const tooltipHeight = 64;
  let left = x + 14;
  if (left + tooltipWidth > chartWidth - 8) left = x - tooltipWidth - 14;
  left = Math.max(8, left);

  let top = y - tooltipHeight - 8;
  if (top < 8) top = y + 8;
  top = Math.max(8, Math.min(top, chartHeight - tooltipHeight - 8));

  historyTooltipEl.style.left = `${left}px`;
  historyTooltipEl.style.top = `${top}px`;
  historyTooltipEl.classList.remove("hidden");
}

function drawPlaceholder(message) {
  const { ctx, width, height } = fitCanvas(historyCanvas, CHART_HEIGHT);
  ctx.clearRect(0, 0, width, height);
  paintHistoricalBackdrop(ctx, width, height);
  ctx.fillStyle = CHART_COLORS.placeholderText;
  ctx.font = "14px sans-serif";
  ctx.textAlign = "center";
  ctx.shadowBlur = 10;
  ctx.shadowColor = "rgba(45, 181, 255, 0.35)";
  ctx.fillText(message, width / 2, height / 2);
  ctx.shadowBlur = 0;
  chartState.renderInfo = null;
  hideTooltip();
}

function resetViewport() {
  if (chartState.points.length < 2) return;
  chartState.viewportStart = 0;
  chartState.viewportEnd = chartState.points.length - 1;
}

function clampViewport(start, end) {
  const maxIndex = chartState.points.length - 1;
  if (maxIndex <= 0) return { start: 0, end: 0 };

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

  if (span >= fullSpan) return { start: 0, end: maxIndex };

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
  paintHistoricalBackdrop(ctx, width, height);

  const left = 64;
  const right = width - 20;
  const top = 18;
  const priceBottom = height - 92;
  const volumeTop = height - 74;
  const volumeBottom = height - 32;
  const plotWidth = right - left;
  const priceHeight = priceBottom - top;
  const volumeHeight = volumeBottom - volumeTop;
  const viewSpan = Math.max(1, chartState.viewportEnd - chartState.viewportStart);

  const visibleStart = Math.max(0, Math.floor(chartState.viewportStart));
  const visibleEnd = Math.min(points.length - 1, Math.ceil(chartState.viewportEnd));

  const vwapValue = activeInterval === "1min"
    ? Number(currentPayload?.technical?.vwap_1m)
    : activeInterval === "5min"
      ? Number(currentPayload?.technical?.vwap_5m)
      : null;

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let maxVolume = 0;

  for (let index = visibleStart; index <= visibleEnd; index += 1) {
    const close = Number(points[index]?.c);
    if (Number.isFinite(close)) {
      min = Math.min(min, close);
      max = Math.max(max, close);
    }
    const volume = Number(points[index]?.v);
    if (Number.isFinite(volume) && volume > 0) {
      maxVolume = Math.max(maxVolume, volume);
    }
  }

  if (Number.isFinite(vwapValue)) {
    min = Math.min(min, vwapValue);
    max = Math.max(max, vwapValue);
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    drawPlaceholder("Invalid data points");
    return;
  }
  if (min === max) {
    min -= 1;
    max += 1;
  }

  const yScale = priceHeight / (max - min);
  const xFromIndex = (index) => left + (((index - chartState.viewportStart) / viewSpan) * plotWidth);
  const yFromValue = (value) => priceBottom - ((value - min) * yScale);

  ctx.strokeStyle = CHART_COLORS.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, volumeBottom);
  ctx.lineTo(right, volumeBottom);
  ctx.stroke();

  ctx.strokeStyle = CHART_COLORS.grid;
  ctx.setLineDash([4, 4]);
  for (let step = 1; step <= 3; step += 1) {
    const y = top + ((priceHeight / 4) * step);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  ctx.fillStyle = CHART_COLORS.volume;
  for (let index = visibleStart; index <= visibleEnd; index += 1) {
    const volume = Number(points[index]?.v);
    if (!Number.isFinite(volume) || volume <= 0 || maxVolume <= 0) continue;
    const x = xFromIndex(index);
    const barWidth = Math.max(1, plotWidth / Math.max(visibleEnd - visibleStart + 1, 1) * 0.7);
    const barHeight = (volume / maxVolume) * volumeHeight;
    ctx.fillRect(x - (barWidth / 2), volumeBottom - barHeight, barWidth, barHeight);
  }

  const area = ctx.createLinearGradient(0, top, 0, priceBottom);
  area.addColorStop(0, CHART_COLORS.lineArea);
  area.addColorStop(1, "rgba(19, 199, 255, 0.02)");
  ctx.fillStyle = area;
  ctx.beginPath();
  let areaStarted = false;
  let areaEndX = left;
  for (let index = visibleStart; index <= visibleEnd; index += 1) {
    const value = Number(points[index]?.c);
    if (!Number.isFinite(value)) continue;
    const x = xFromIndex(index);
    const y = yFromValue(value);
    areaEndX = x;
    if (!areaStarted) {
      ctx.moveTo(x, priceBottom);
      ctx.lineTo(x, y);
      areaStarted = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  if (areaStarted) {
    ctx.lineTo(areaEndX, priceBottom);
    ctx.closePath();
    ctx.fill();
  }

  ctx.strokeStyle = CHART_COLORS.line;
  ctx.lineWidth = 2.2;
  ctx.shadowBlur = 10;
  ctx.shadowColor = CHART_COLORS.lineGlow;
  ctx.beginPath();
  let started = false;
  for (let index = visibleStart; index <= visibleEnd; index += 1) {
    const value = Number(points[index]?.c);
    if (!Number.isFinite(value)) continue;
    const x = xFromIndex(index);
    const y = yFromValue(value);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  if (started) ctx.stroke();
  ctx.shadowBlur = 0;

  if (Number.isFinite(vwapValue)) {
    const y = yFromValue(vwapValue);
    ctx.strokeStyle = CHART_COLORS.vwap;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = CHART_COLORS.label;
  ctx.font = "12px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(formatPrice(max), 8, top + 6);
  ctx.fillText(formatPrice(min), 8, priceBottom);
  const startLabelIndex = Math.max(0, Math.min(points.length - 1, Math.round(chartState.viewportStart)));
  const endLabelIndex = Math.max(0, Math.min(points.length - 1, Math.round(chartState.viewportEnd)));
  ctx.textAlign = "left";
  ctx.fillText(String(points[startLabelIndex]?.t || "-"), left, height - 12);
  ctx.textAlign = "right";
  ctx.fillText(String(points[endLabelIndex]?.t || "-"), right, height - 12);

  const hoverIndex = chartState.hoveredIndex;
  if (Number.isInteger(hoverIndex) && hoverIndex >= visibleStart && hoverIndex <= visibleEnd) {
    const hoverValue = Number(points[hoverIndex]?.c);
    if (Number.isFinite(hoverValue)) {
      const hoverX = xFromIndex(hoverIndex);
      const hoverY = yFromValue(hoverValue);

      ctx.strokeStyle = CHART_COLORS.crosshair;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX, top);
      ctx.lineTo(hoverX, volumeBottom);
      ctx.stroke();

      ctx.fillStyle = CHART_COLORS.point;
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
    priceBottom,
    volumeBottom,
    plotWidth,
    viewStart: chartState.viewportStart,
    viewEnd: chartState.viewportEnd,
  };
}

function applySeriesToChart(resetView = false) {
  chartState.points = chartSeriesByInterval[activeInterval] || [];
  if (chartState.points.length < 2) {
    drawPlaceholder("Not enough data points");
    return;
  }

  if (resetView || chartState.viewportEnd <= chartState.viewportStart || chartState.viewportEnd > chartState.points.length - 1) {
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
  if (Math.abs(targetSpan - currentSpan) < 0.001) return;

  const centerIndex = getIndexFromCanvasX(canvasX) ?? Math.round((chartState.viewportStart + chartState.viewportEnd) / 2);
  const ratio = (centerIndex - chartState.viewportStart) / currentSpan;
  const nextStart = centerIndex - (targetSpan * ratio);
  const nextEnd = nextStart + targetSpan;

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

function setIntervalButtonState() {
  interval1mBtn.classList.toggle("active", activeInterval === "1min");
  interval5mBtn.classList.toggle("active", activeInterval === "5min");
  interval1dBtn.classList.toggle("active", activeInterval === "1day");
}

function setIntervalButtonAvailability() {
  interval1mBtn.disabled = chartSeriesByInterval["1min"].length < 2;
  interval5mBtn.disabled = chartSeriesByInterval["5min"].length < 2;
  interval1dBtn.disabled = chartSeriesByInterval["1day"].length < 2;
  const dayReady = activeInterval === "1day" && chartSeriesByInterval["1day"].length >= 2;
  rangeMaxBtn.disabled = !dayReady;
  range10yBtn.disabled = !dayReady;
  range5yBtn.disabled = !dayReady;
  range1yBtn.disabled = !dayReady;
  rangeYtdBtn.disabled = !dayReady;
}

async function setActiveInterval(nextInterval) {
  if (!INTERVAL_KEYS.includes(nextInterval)) return;
  if (
    (nextInterval === "1min" || nextInterval === "5min")
    && (!Array.isArray(chartSeriesByInterval[nextInterval]) || chartSeriesByInterval[nextInterval].length < 2)
  ) {
    await ensureIntradayLoaded(false);
  }
  if (!Array.isArray(chartSeriesByInterval[nextInterval]) || chartSeriesByInterval[nextInterval].length < 2) return;
  activeInterval = nextInterval;
  setIntervalButtonAvailability();
  setIntervalButtonState();
  chartState.hoveredIndex = null;
  applySeriesToChart(true);
}

async function ensureIntradayLoaded(refresh = false) {
  if (!currentSymbol) return false;
  const has1m = Array.isArray(chartSeriesByInterval["1min"]) && chartSeriesByInterval["1min"].length >= 2;
  const has5m = Array.isArray(chartSeriesByInterval["5min"]) && chartSeriesByInterval["5min"].length >= 2;
  if (!refresh && has1m && has5m) return true;

  try {
    const url = `/api/security-overview/${encodeURIComponent(currentSymbol)}/intraday${refresh ? "?refresh=true" : ""}`;
    const response = await fetch(url);
    const result = await response.json().catch(() => ({}));
    if (!response.ok || !result.ok) {
      return false;
    }

    chartSeriesByInterval["1min"] = Array.isArray(result?.charts?.["1min"]) ? result.charts["1min"] : [];
    chartSeriesByInterval["5min"] = Array.isArray(result?.charts?.["5min"]) ? result.charts["5min"] : [];

    const vwap1m = Number(result?.technical?.vwap_1m);
    const vwap5m = Number(result?.technical?.vwap_5m);
    if (Number.isFinite(vwap1m) || Number.isFinite(vwap5m)) {
      const left = Number.isFinite(vwap1m) ? `$${formatPrice(vwap1m)}` : "-";
      const right = Number.isFinite(vwap5m) ? `$${formatPrice(vwap5m)}` : "-";
      techVwapEl.textContent = `${left} / ${right}`;
    }

    setIntervalButtonAvailability();
    return true;
  } catch (_error) {
    return false;
  }
}

function parsePointDate(raw) {
  const text = String(raw || "").trim();
  if (!text) return null;
  const dateText = text.includes(" ") ? text.split(" ")[0] : text;
  const dt = new Date(`${dateText}T00:00:00`);
  if (Number.isNaN(dt.getTime())) return null;
  return dt;
}

function findStartIndexByDate(targetDate) {
  const points = chartSeriesByInterval["1day"];
  if (!Array.isArray(points) || points.length < 2) return 0;
  for (let idx = 0; idx < points.length; idx += 1) {
    const dt = parsePointDate(points[idx]?.t);
    if (!dt) continue;
    if (dt >= targetDate) return idx;
  }
  return 0;
}

function applyDateRangePreset(kind) {
  if (activeInterval !== "1day") return;
  const points = chartSeriesByInterval["1day"];
  if (!Array.isArray(points) || points.length < 2) return;

  const lastIndex = points.length - 1;
  const lastDate = parsePointDate(points[lastIndex]?.t);
  if (!lastDate) {
    resetZoom();
    return;
  }

  if (kind === "max") {
    chartState.viewportStart = 0;
    chartState.viewportEnd = lastIndex;
    chartState.hoveredIndex = null;
    drawChartFromState();
    return;
  }

  let startDate = null;
  if (kind === "10y") {
    startDate = new Date(lastDate);
    startDate.setFullYear(startDate.getFullYear() - 10);
  } else if (kind === "5y") {
    startDate = new Date(lastDate);
    startDate.setFullYear(startDate.getFullYear() - 5);
  } else if (kind === "1y") {
    startDate = new Date(lastDate);
    startDate.setFullYear(startDate.getFullYear() - 1);
  } else if (kind === "ytd") {
    startDate = new Date(lastDate.getFullYear(), 0, 1);
  }

  if (!startDate) {
    return;
  }

  const startIndex = findStartIndexByDate(startDate);
  chartState.viewportStart = startIndex;
  chartState.viewportEnd = lastIndex;
  chartState.hoveredIndex = null;
  drawChartFromState();
}

function setOverviewFields(payload) {
  const symbol = payload?.symbol || currentSymbol;
  const name = payload?.name || "";
  ovSymbolNameEl.textContent = name ? `${symbol} / ${name}` : symbol;

  const price = payload?.price || {};
  const volume = payload?.volume || {};
  const spread = payload?.spread || {};
  const technical = payload?.technical || {};
  const market = payload?.market || {};
  const support = payload?.support_status || {};

  const current = Number(price.current);
  const changeAbs = Number(price.change_abs);
  const changePct = Number(price.change_pct);
  const dayHigh = Number(price.day_high);
  const dayLow = Number(price.day_low);
  const volToday = Number(volume.today);
  const volAvg = Number(volume.avg20);
  const volRatio = Number(volume.avg_ratio);
  const turnover = Number(volume.turnover);
  const bid = Number(spread.bid);
  const ask = Number(spread.ask);
  const spreadAbs = Number(spread.spread_abs);
  const spreadPct = Number(spread.spread_pct);

  ovCurrentEl.textContent = Number.isFinite(current) ? `$${formatPrice(current)}` : "-";
  ovChangeEl.textContent = Number.isFinite(changeAbs) && Number.isFinite(changePct)
    ? `${formatSignedPrice(changeAbs)} (${formatPercent(changePct)})`
    : "-";
  ovDayRangeEl.textContent = Number.isFinite(dayHigh) && Number.isFinite(dayLow)
    ? `$${formatPrice(dayHigh)} / $${formatPrice(dayLow)}`
    : "-";
  ovVolumeEl.textContent = Number.isFinite(volToday)
    ? `${formatCompact(volToday)}${Number.isFinite(volAvg) && Number.isFinite(volRatio) ? ` (avg20 ${formatCompact(volAvg)}, x${volRatio.toFixed(2)})` : ""}`
    : "-";
  ovTurnoverEl.textContent = Number.isFinite(turnover) ? `$${formatCompact(turnover)}` : "-";
  ovSpreadEl.textContent = Number.isFinite(bid) && Number.isFinite(ask)
    ? `${formatPrice(bid)} / ${formatPrice(ask)} (${Number.isFinite(spreadAbs) ? formatPrice(spreadAbs) : "-"}, ${Number.isFinite(spreadPct) ? formatPercent(spreadPct) : "-"})`
    : "-";
  ovUpdatedEl.textContent = `${formatIsoTime(price.updated_at)}${price.delay_note ? ` | ${price.delay_note}` : ""}`;

  const vwap1m = Number(technical.vwap_1m);
  const vwap5m = Number(technical.vwap_5m);
  const ma20 = Number(technical.ma_short_20);
  const ma50 = Number(technical.ma_mid_50);
  const beta = Number(market.beta_60d_vs_spy);
  const corr = Number(market.corr_60d_vs_spy);

  techVwapEl.textContent = Number.isFinite(vwap1m) || Number.isFinite(vwap5m)
    ? `${Number.isFinite(vwap1m) ? `$${formatPrice(vwap1m)}` : "-"} / ${Number.isFinite(vwap5m) ? `$${formatPrice(vwap5m)}` : "-"}`
    : "-";
  techMaEl.textContent = Number.isFinite(ma20) || Number.isFinite(ma50)
    ? `${Number.isFinite(ma20) ? `$${formatPrice(ma20)}` : "-"} / ${Number.isFinite(ma50) ? `$${formatPrice(ma50)}` : "-"}`
    : "-";
  techBetaEl.textContent = Number.isFinite(beta) ? beta.toFixed(3) : "-";
  techCorrEl.textContent = Number.isFinite(corr) ? corr.toFixed(3) : "-";

  marketSpyEl.textContent = formatMarketProxy(market.sp500_proxy);
  marketQqqEl.textContent = formatMarketProxy(market.nasdaq_proxy);
  if (loadQqqBtn) {
    loadQqqBtn.disabled = market.nasdaq_proxy !== null;
  }

  supportSectorEl.textContent = support.sector_etf || "not_supported";
  supportBoardEl.textContent = support.order_book || "not_supported";
  supportEventsEl.textContent = support.earnings_calendar || "not_supported";
  supportCorporateEl.textContent = support.corporate_events || "not_supported";
  supportNewsEl.textContent = support.news_headlines || "not_supported";

  setCurrentPrice(current);
  setRiskMetrics(chartSeriesByInterval["1day"], payload);
}

function formatMarketProxy(item) {
  if (item === null) return "Not loaded (credit saving mode)";
  if (!item || typeof item !== "object") return "-";
  const symbol = item.symbol || "-";
  const price = Number(item.price);
  const pct = Number(item.change_pct);
  if (!Number.isFinite(price)) return symbol;
  return `${symbol} $${formatPrice(price)} (${formatPercent(pct)})`;
}

async function loadOverview(refresh = false) {
  if (!currentSymbol) return;

  refreshHistoryBtn.disabled = true;
  clearHistoryCacheBtn.disabled = true;
  setHistoryMeta("Loading market overview...");
  drawPlaceholder("Loading...");

  try {
    const url = `/api/security-overview/${encodeURIComponent(currentSymbol)}?include_intraday=false&include_market=true&include_qqq=false${refresh ? "&refresh=true" : ""}`;
    const response = await fetch(url);
    const result = await response.json().catch(() => ({}));

    if (!response.ok || !result.ok) {
      const fallbackOk = await loadDailyFallback(refresh);
      if (!fallbackOk) {
        setHistoryMeta(result.detail || "Failed to load overview", true);
        drawPlaceholder("Overview unavailable");
      }
      return;
    }

    currentPayload = result;
    chartSeriesByInterval["1min"] = Array.isArray(result?.charts?.["1min"]) ? result.charts["1min"] : [];
    chartSeriesByInterval["5min"] = Array.isArray(result?.charts?.["5min"]) ? result.charts["5min"] : [];
    chartSeriesByInterval["1day"] = Array.isArray(result?.charts?.["1day"]) ? result.charts["1day"] : [];

    if (!Array.isArray(chartSeriesByInterval[activeInterval]) || chartSeriesByInterval[activeInterval].length < 2) {
      const next = INTERVAL_KEYS.find((key) => Array.isArray(chartSeriesByInterval[key]) && chartSeriesByInterval[key].length >= 2);
      activeInterval = next || "1day";
    }

    setIntervalButtonAvailability();
    setIntervalButtonState();
    setOverviewFields(result);
    symbolSubtitleEl.textContent = `${result.exchange || "-"} | data source: ${result.source || "unknown"}`;
    setHistoryMeta(`Loaded market overview (${activeInterval}).`);
    chartState.hoveredIndex = null;
    applySeriesToChart(true);
    if (activeInterval === "1day") {
      applyDateRangePreset("max");
    }
  } finally {
    refreshHistoryBtn.disabled = false;
    clearHistoryCacheBtn.disabled = false;
  }
}

async function clearHistoryCacheAndReload() {
  if (!currentSymbol) return;
  refreshHistoryBtn.disabled = true;
  clearHistoryCacheBtn.disabled = true;
  setHistoryMeta("Clearing cache...");
  try {
    const response = await fetch(`/api/security-overview/${encodeURIComponent(currentSymbol)}/clear-cache`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok || !result.ok) {
      setHistoryMeta(result.detail || "Failed to clear cache", true);
      return;
    }
    setHistoryMeta("Cache cleared. Reloading...");
    await loadOverview(true);
  } finally {
    refreshHistoryBtn.disabled = false;
    clearHistoryCacheBtn.disabled = false;
  }
}

async function loadQqqOnDemand() {
  if (!currentSymbol || !loadQqqBtn) return;
  loadQqqBtn.disabled = true;
  try {
    const url = `/api/security-overview/${encodeURIComponent(currentSymbol)}?include_intraday=false&include_market=true&include_qqq=true`;
    const response = await fetch(url);
    const result = await response.json().catch(() => ({}));
    if (!response.ok || !result.ok) {
      marketQqqEl.textContent = "Load failed";
      return;
    }
    const market = result?.market || {};
    marketQqqEl.textContent = formatMarketProxy(market.nasdaq_proxy);
    if (market.nasdaq_proxy === null) {
      loadQqqBtn.disabled = false;
    }
  } catch (_error) {
    marketQqqEl.textContent = "Load failed";
    loadQqqBtn.disabled = false;
  }
}

async function loadDailyFallback(refresh = false) {
  try {
    const url = `/api/historical/${encodeURIComponent(currentSymbol)}?years=5${refresh ? "&refresh=true" : ""}`;
    const response = await fetch(url);
    const result = await response.json().catch(() => ({}));
    if (!response.ok || !result.ok || !Array.isArray(result.points) || result.points.length < 2) {
      return false;
    }

    chartSeriesByInterval["1min"] = [];
    chartSeriesByInterval["5min"] = [];
    chartSeriesByInterval["1day"] = result.points;
    activeInterval = "1day";
    setIntervalButtonAvailability();
    setIntervalButtonState();

    ovSymbolNameEl.textContent = currentSymbol;
    ovCurrentEl.textContent = "-";
    ovChangeEl.textContent = "-";
    ovDayRangeEl.textContent = "-";
    ovVolumeEl.textContent = "-";
    ovTurnoverEl.textContent = "-";
    ovSpreadEl.textContent = "-";
    ovUpdatedEl.textContent = "-";
    techVwapEl.textContent = "-";
    techMaEl.textContent = "-";
    techBetaEl.textContent = "-";
    techCorrEl.textContent = "-";
    marketSpyEl.textContent = "-";
    marketQqqEl.textContent = "-";
    if (loadQqqBtn) {
      loadQqqBtn.disabled = false;
    }
    supportSectorEl.textContent = "not_supported_on_fallback";
    supportBoardEl.textContent = "not_supported_on_fallback";
    supportEventsEl.textContent = "not_supported_on_fallback";
    supportCorporateEl.textContent = "not_supported_on_fallback";
    supportNewsEl.textContent = "not_supported_on_fallback";
    techAtrEl.textContent = "-";
    priceGapEl.textContent = "-";

    const lastClose = Number(result.points[result.points.length - 1]?.c);
    setCurrentPrice(lastClose);
    setRiskMetrics(result.points, {});
    symbolSubtitleEl.textContent = `${result.years || "-"}Y ${result.interval || "1day"} (${result.from || "-"} - ${result.to || "-"})`;
    setHistoryMeta("Overview API unavailable. Loaded daily historical fallback.");
    chartState.hoveredIndex = null;
    currentPayload = null;
    applySeriesToChart(true);
    return true;
  } catch (_error) {
    return false;
  }
}

function getSymbolFromPath() {
  const path = window.location.pathname;
  const parts = path.split("/").filter(Boolean);
  if (parts.length < 2) return "";
  return decodeURIComponent(parts[1] || "").trim().toUpperCase();
}

backBtn.addEventListener("click", () => {
  if (window.history.length > 1) {
    window.history.back();
    return;
  }
  window.location.href = "/";
});

refreshHistoryBtn.addEventListener("click", async () => {
  await loadOverview(true);
  if (activeInterval === "1min" || activeInterval === "5min") {
    const loaded = await ensureIntradayLoaded(true);
    if (loaded) {
      applySeriesToChart(true);
    }
  }
});

zoomInBtn.addEventListener("click", () => zoomWithFactor(ZOOM_IN_FACTOR));
zoomOutBtn.addEventListener("click", () => zoomWithFactor(ZOOM_OUT_FACTOR));
resetZoomBtn.addEventListener("click", () => resetZoom());
clearHistoryCacheBtn.addEventListener("click", async () => {
  await clearHistoryCacheAndReload();
});
if (loadQqqBtn) {
  loadQqqBtn.addEventListener("click", async () => {
    await loadQqqOnDemand();
  });
}

interval1mBtn.addEventListener("click", async () => { await setActiveInterval("1min"); });
interval5mBtn.addEventListener("click", async () => { await setActiveInterval("5min"); });
interval1dBtn.addEventListener("click", async () => { await setActiveInterval("1day"); });
rangeMaxBtn.addEventListener("click", () => applyDateRangePreset("max"));
range10yBtn.addEventListener("click", () => applyDateRangePreset("10y"));
range5yBtn.addEventListener("click", () => applyDateRangePreset("5y"));
range1yBtn.addEventListener("click", () => applyDateRangePreset("1y"));
rangeYtdBtn.addEventListener("click", () => applyDateRangePreset("ytd"));

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
  dragTo(canvasXFromEvent(event));
});

historyCanvas.addEventListener("mousemove", (event) => {
  if (isDragging) return;
  updateHoverFromCanvasX(canvasXFromEvent(event));
});

window.addEventListener("mouseup", () => endDrag());

historyCanvas.addEventListener("mouseleave", () => {
  if (isDragging) return;
  chartState.hoveredIndex = null;
  drawChartFromState();
});

historyCanvas.addEventListener("dblclick", () => resetZoom());

window.addEventListener("resize", () => {
  if (currentPayload) {
    applySeriesToChart(false);
  } else {
    drawPlaceholder("Loading...");
  }
});

async function init() {
  currentSymbol = getSymbolFromPath();
  if (!currentSymbol) {
    document.title = "Stock Overview";
    symbolTitleEl.textContent = "Stock Overview";
    symbolSubtitleEl.textContent = "Symbol not found";
    setHistoryMeta("Invalid symbol", true);
    drawPlaceholder("Invalid symbol");
    return;
  }

  document.title = `${currentSymbol} Stock Overview`;
  symbolTitleEl.textContent = `${currentSymbol} Market Overview`;
  symbolSubtitleEl.textContent = "Loading...";
  setIntervalButtonAvailability();
  setIntervalButtonState();
  drawPlaceholder("Loading...");

  await loadOverview(false);
}

void init();
