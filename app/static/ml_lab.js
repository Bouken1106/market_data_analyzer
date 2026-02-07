const mlForm = document.getElementById("ml-form");
const runBtn = document.getElementById("ml-run-btn");
const statusEl = document.getElementById("ml-status");
const quantileCanvas = document.getElementById("quantile-function-canvas");
const fanCanvas = document.getElementById("fan-chart-canvas");
const quantileScaleEl = document.getElementById("quantile-scale");
const fanScaleEl = document.getElementById("fan-scale");
const quantileLegendEl = document.getElementById("quantile-legend");
const fanMetaEl = document.getElementById("fan-meta");
const metricPinballEl = document.getElementById("metric-pinball");
const metricCov90El = document.getElementById("metric-cov90");
const metricCov50El = document.getElementById("metric-cov50");
const metricSamplesEl = document.getElementById("metric-samples");

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
  const height = Number(canvas.getAttribute("height")) || 320;
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

function xAt(index, total, left, right) {
  if (total <= 1) {
    return (left + right) / 2;
  }
  return left + ((index / (total - 1)) * (right - left));
}

function yAt(value, min, max, top, bottom) {
  const ratio = (value - min) / (max - min);
  return bottom - (ratio * (bottom - top));
}

function drawBaseAxes(ctx, width, height, bounds, yFormatter, xTicks) {
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
    const value = bounds.min + (((bounds.max - bounds.min) / 4) * step);
    const y = yAt(value, bounds.min, bounds.max, top, bottom);
    ctx.fillText(yFormatter(value), left - 8, y + 3);
  }

  ctx.textAlign = "center";
  (xTicks || []).forEach((item) => {
    const x = left + (item.ratio * (right - left));
    ctx.fillText(item.label, x, bottom + 18);
  });

  return { left, right, top, bottom };
}

function renderQuantileFunction() {
  if (!latestPayload?.quantile_function?.curves?.length) {
    drawPlaceholder(quantileCanvas, "No quantile function data");
    return;
  }

  const mode = quantileScaleEl.value;
  const curves = latestPayload.quantile_function.curves;
  const taus = latestPayload.quantile_function.taus || [];
  if (taus.length === 0) {
    drawPlaceholder(quantileCanvas, "No quantiles");
    return;
  }

  const series = curves.map((curve) =>
    mode === "prices" ? (curve.price_quantiles || []) : (curve.return_quantiles || [])
  );
  const bounds = computeBounds(series);
  const { ctx, width, height } = fitCanvas(quantileCanvas);
  const chart = drawBaseAxes(
    ctx,
    width,
    height,
    bounds,
    (value) => (mode === "prices" ? formatCompact(value) : `${(value * 100).toFixed(2)}%`),
    [
      { ratio: 0.0, label: "0.01" },
      { ratio: 0.5, label: "0.50" },
      { ratio: 1.0, label: "0.99" },
    ]
  );

  curves.forEach((curve, curveIdx) => {
    const line = series[curveIdx];
    if (!Array.isArray(line) || line.length !== taus.length) return;

    ctx.strokeStyle = COLORS.curves[curveIdx % COLORS.curves.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    line.forEach((value, idx) => {
      const x = xAt(idx, line.length, chart.left, chart.right);
      const y = yAt(Number(value), bounds.min, bounds.max, chart.top, chart.bottom);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  const legendLines = curves.map((curve, idx) => {
    const actual = mode === "prices" ? Number(curve.actual_price) : Number(curve.actual_return);
    const actualLabel = mode === "prices" ? formatCompact(actual) : `${(actual * 100).toFixed(2)}%`;
    return `${idx + 1}. ${curve.date} (actual: ${actualLabel})`;
  });
  quantileLegendEl.textContent = legendLines.join(" | ");
}

function fillBand(ctx, lower, upper, chart, bounds, color) {
  if (!Array.isArray(lower) || !Array.isArray(upper) || lower.length === 0 || lower.length !== upper.length) {
    return;
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  for (let idx = 0; idx < upper.length; idx += 1) {
    const x = xAt(idx, upper.length, chart.left, chart.right);
    const y = yAt(Number(upper[idx]), bounds.min, bounds.max, chart.top, chart.bottom);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let idx = lower.length - 1; idx >= 0; idx -= 1) {
    const x = xAt(idx, lower.length, chart.left, chart.right);
    const y = yAt(Number(lower[idx]), bounds.min, bounds.max, chart.top, chart.bottom);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
}

function renderFanChart() {
  const fan = latestPayload?.fan_chart;
  if (!fan?.dates?.length) {
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

  const bounds = computeBounds([q05, q25, q50, q75, q95, actual]);
  const { ctx, width, height } = fitCanvas(fanCanvas);
  const chart = drawBaseAxes(
    ctx,
    width,
    height,
    bounds,
    (value) => (mode === "prices" ? formatCompact(value) : `${(value * 100).toFixed(2)}%`),
    [
      { ratio: 0.0, label: fan.dates[0] || "" },
      { ratio: 0.5, label: fan.dates[Math.floor(fan.dates.length / 2)] || "" },
      { ratio: 1.0, label: fan.dates[fan.dates.length - 1] || "" },
    ]
  );

  fillBand(ctx, q05, q95, chart, bounds, COLORS.band90);
  fillBand(ctx, q25, q75, chart, bounds, COLORS.band50);

  ctx.strokeStyle = COLORS.median;
  ctx.lineWidth = 2;
  ctx.beginPath();
  q50.forEach((value, idx) => {
    const x = xAt(idx, q50.length, chart.left, chart.right);
    const y = yAt(Number(value), bounds.min, bounds.max, chart.top, chart.bottom);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = COLORS.actual;
  actual.forEach((value, idx) => {
    const x = xAt(idx, actual.length, chart.left, chart.right);
    const y = yAt(Number(value), bounds.min, bounds.max, chart.top, chart.bottom);
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });

  fanMetaEl.textContent = `${fan.dates[0]} 〜 ${fan.dates[fan.dates.length - 1]} | points: ${fan.dates.length}`;
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
    renderMetrics(body);
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

mlForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runQuantileForecast();
});

quantileScaleEl.addEventListener("change", () => {
  if (latestPayload) renderQuantileFunction();
});

fanScaleEl.addEventListener("change", () => {
  if (latestPayload) renderFanChart();
});

window.addEventListener("resize", () => {
  if (!latestPayload) return;
  renderQuantileFunction();
  renderFanChart();
});

drawPlaceholder(quantileCanvas, "Run Quantile LSTM to draw quantile curves");
drawPlaceholder(fanCanvas, "Run Quantile LSTM to draw fan chart");
