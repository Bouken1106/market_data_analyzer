const mlForm = document.getElementById("ml-form");
const runBtn = document.getElementById("ml-run-btn");
const statusEl = document.getElementById("ml-status");
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

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
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
    buildIndexTicks(taus.map((item) => Number(item).toFixed(2)), xWindow)
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

  btReturnStrategyEl.textContent = `${(strategyReturn * 100).toFixed(2)}%`;
  btReturnBuyholdEl.textContent = `${(buyHoldReturn * 100).toFixed(2)}%`;
  btOutperfEl.textContent = `${(outperf * 100).toFixed(2)}%`;
  btCapitalStrategyEl.textContent = formatMoney(strategyCapital);
  btCapitalBuyholdEl.textContent = formatMoney(buyHoldCapital);
  btWindowEl.textContent = `${days}D (${from} → ${to})`;

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
  ctx.fillText("Buy & Hold", chart.left + 64, chart.top + 12);
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
  const cdfPoints = [];
  for (let i = 0; i < retQ.length; i += 1) {
    const x = retQ[i];
    const t = taus[i];
    if (!Number.isFinite(x) || !Number.isFinite(t)) continue;
    cdfPoints.push({ x, t: Math.max(0, Math.min(1, t)) });
  }

  const densityPoints = [];
  for (let i = 1; i < retQ.length; i += 1) {
    const dx = retQ[i] - retQ[i - 1];
    const dt = taus[i] - taus[i - 1];
    if (!Number.isFinite(dx) || !Number.isFinite(dt)) continue;
    if (dx <= 1e-9) continue;
    const xMid = (retQ[i] + retQ[i - 1]) / 2;
    const density = dt / dx;
    if (!Number.isFinite(xMid) || !Number.isFinite(density) || density < 0) continue;
    densityPoints.push({ x: xMid, d: density });
  }

  const smooth = densityPoints.map((point, idx) => {
    const prev = densityPoints[Math.max(0, idx - 1)]?.d ?? point.d;
    const next = densityPoints[Math.min(densityPoints.length - 1, idx + 1)]?.d ?? point.d;
    return {
      x: point.x,
      d: ((prev + point.d + next) / 3),
    };
  });

  const zeroX = xAtValue(0);
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.beginPath();
  ctx.moveTo(zeroX, chartTop);
  ctx.lineTo(zeroX, bottom);
  ctx.stroke();

  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath();
  ctx.moveTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.moveTo(left, chartTop);
  ctx.lineTo(left, bottom);
  ctx.stroke();

  if (distType === "pdf") {
    if (smooth.length < 2) {
      drawPlaceholder(nextDayCanvas, "Not enough points for probability density");
      return;
    }

    const maxDensity = Math.max(...smooth.map((p) => p.d), 1e-9);
    const yMaxDensity = maxDensity * 1.15;
    const yAtDensity = (density) => bottom - ((density / Math.max(1e-9, yMaxDensity)) * plotHeight);

    ctx.fillStyle = COLORS.label;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("0", left - 8, bottom + 3);
    ctx.fillText(yMaxDensity.toFixed(1), left - 8, chartTop + 3);

    ctx.fillStyle = "rgba(33, 182, 255, 0.16)";
    ctx.beginPath();
    smooth.forEach((point, idx) => {
      const x = xAtValue(point.x);
      const y = yAtDensity(point.d);
      if (idx === 0) {
        ctx.moveTo(x, bottom);
        ctx.lineTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.lineTo(xAtValue(smooth[smooth.length - 1].x), bottom);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "rgba(33, 182, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < smooth.length; i += 1) {
      const x = xAtValue(smooth[i].x);
      const y = yAtDensity(smooth[i].d);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
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

    const cdfMarkers = [
      { idx: 9, label: "10%" },
      { idx: 49, label: "50%" },
      { idx: 89, label: "90%" },
    ];
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    cdfMarkers.forEach((marker) => {
      if (marker.idx < 0 || marker.idx >= retQ.length) return;
      const markerValue = Number(retQ[marker.idx]);
      if (!Number.isFinite(markerValue)) return;
      const x = xAtValue(markerValue);
      ctx.strokeStyle = "rgba(154, 173, 196, 0.65)";
      ctx.beginPath();
      ctx.moveTo(x, chartTop);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.fillStyle = "#cfe0f6";
      ctx.fillText(marker.label, x, chartTop - 8);
    });
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
  return fallback;
}

async function runQuantileForecast() {
  const symbol = normalizeSymbol(document.getElementById("ml-symbol").value);
  if (!symbol) {
    setStatus("Symbolを入力してください。", true);
    return;
  }

  const params = new URLSearchParams({
    symbol,
    years: String(document.getElementById("ml-years").value || "5"),
    sequence_length: String(document.getElementById("ml-seq-len").value || "60"),
    hidden_size: String(document.getElementById("ml-hidden-size").value || "64"),
    num_layers: String(document.getElementById("ml-num-layers").value || "2"),
    dropout: String(document.getElementById("ml-dropout").value || "0.2"),
    max_epochs: String(document.getElementById("ml-max-epochs").value || "80"),
    patience: String(document.getElementById("ml-patience").value || "10"),
  });

  if (document.getElementById("ml-refresh").checked) {
    params.set("refresh", "true");
  }

  runBtn.disabled = true;
  setStatus("学習と推論を実行中です...");

  try {
    const response = await fetch(`/api/ml/quantile-lstm?${params.toString()}`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(parseErrorMessage(body, "予測実行に失敗しました。"));
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
    setStatus(`完了: ${symbol} | epochs=${epochs}, best val pinball=${formatNumber(valLoss, 6)}`);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "予測実行に失敗しました。", true);
  } finally {
    runBtn.disabled = false;
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
  await runQuantileForecast();
});

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

drawPlaceholder(nextDayCanvas, "Run Quantile LSTM to draw next-day distribution");
drawPlaceholder(backtestCanvas, "Run Quantile LSTM to draw 60-day realized backtest");
drawPlaceholder(quantileCanvas, "Run Quantile LSTM to draw quantile curves");
drawPlaceholder(fanCanvas, "Run Quantile LSTM to draw fan chart");
