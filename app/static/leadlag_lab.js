const statusEl = document.getElementById("llg-status");
const runBtn = document.getElementById("llg-run");

const usSymbolsEl = document.getElementById("llg-us-symbols");
const jpSymbolsEl = document.getElementById("llg-jp-symbols");
const cyclicalSymbolsEl = document.getElementById("llg-cyclical-symbols");
const defensiveSymbolsEl = document.getElementById("llg-defensive-symbols");
const windowEl = document.getElementById("llg-window");
const lambdaEl = document.getElementById("llg-lambda");
const componentsEl = document.getElementById("llg-components");
const quantileEl = document.getElementById("llg-quantile");
const historyYearsEl = document.getElementById("llg-history-years");
const cfullStartEl = document.getElementById("llg-cfull-start");
const cfullEndEl = document.getElementById("llg-cfull-end");
const refreshEl = document.getElementById("llg-refresh");
const includeBacktestEl = document.getElementById("llg-include-backtest");
const transferMatrixEl = document.getElementById("llg-transfer-matrix");

const summaryGridEl = document.getElementById("llg-summary-grid");
const activeUsEl = document.getElementById("llg-active-us");
const activeJpEl = document.getElementById("llg-active-jp");
const excludedBodyEl = document.getElementById("llg-excluded-body");
const regularizationMetaEl = document.getElementById("llg-regularization-meta");
const d0BodyEl = document.getElementById("llg-d0-body");

const latestMetaEl = document.getElementById("llg-latest-meta");
const factorStripEl = document.getElementById("llg-factor-strip");
const latestBodyEl = document.getElementById("llg-latest-body");
const transferWrapEl = document.getElementById("llg-transfer-wrap");
const transferTextEl = document.getElementById("llg-transfer-matrix-text");

const strategyGridEl = document.getElementById("llg-strategy-grid");
const recentBodyEl = document.getElementById("llg-recent-body");

let running = false;

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function fmtNum(value, digits = 4) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function fmtPct(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${num.toFixed(digits)}%`;
}

function fetchJson(url, options) {
  return fetch(url, options).then(async (response) => {
    const result = await response.json().catch(() => ({}));
    return { response, result };
  });
}

function asCsvText(values) {
  return Array.isArray(values) ? values.join(",") : String(values || "");
}

function payloadFromForm() {
  return {
    us_symbols: usSymbolsEl.value,
    jp_symbols: jpSymbolsEl.value,
    cyclical_symbols: cyclicalSymbolsEl.value,
    defensive_symbols: defensiveSymbolsEl.value,
    rolling_window_days: Number(windowEl.value || 60),
    lambda_reg: Number(lambdaEl.value || 0.9),
    n_components: Number(componentsEl.value || 3),
    quantile_q: Number(quantileEl.value || 0.3),
    history_years: Number(historyYearsEl.value || 30),
    cfull_start: cfullStartEl.value,
    cfull_end: cfullEndEl.value,
    refresh: Boolean(refreshEl.checked),
    include_backtest: Boolean(includeBacktestEl.checked),
    include_transfer_matrix: Boolean(transferMatrixEl.checked),
  };
}

function setDefaults(defaults) {
  usSymbolsEl.value = asCsvText(defaults.us_symbols);
  jpSymbolsEl.value = asCsvText(defaults.jp_symbols);
  cyclicalSymbolsEl.value = asCsvText(defaults.cyclical_symbols);
  defensiveSymbolsEl.value = asCsvText(defaults.defensive_symbols);
  windowEl.value = defaults.rolling_window_days;
  lambdaEl.value = defaults.lambda_reg;
  componentsEl.value = defaults.n_components;
  quantileEl.value = defaults.quantile_q;
  historyYearsEl.value = defaults.history_years || 30;
  cfullStartEl.value = defaults.cfull_start;
  cfullEndEl.value = defaults.cfull_end;
}

function renderSummaryCards(items) {
  summaryGridEl.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "llg-metric";
    card.innerHTML = `<span class="label">${item.label}</span><strong>${item.value}</strong>`;
    summaryGridEl.appendChild(card);
  });
}

function renderChipList(targetEl, symbols) {
  targetEl.innerHTML = "";
  const values = Array.isArray(symbols) ? symbols : [];
  if (!values.length) {
    targetEl.innerHTML = '<span class="hint">-</span>';
    return;
  }
  values.forEach((symbol) => {
    const chip = document.createElement("span");
    chip.className = "pill chip-cyan";
    chip.textContent = symbol;
    targetEl.appendChild(chip);
  });
}

function renderExcluded(rows) {
  excludedBodyEl.innerHTML = "";
  const values = Array.isArray(rows) ? rows : [];
  if (!values.length) {
    excludedBodyEl.innerHTML = '<tr><td colspan="2">None</td></tr>';
    return;
  }
  values.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.symbol || "-"}</td><td>${row.reason || "-"}</td>`;
    excludedBodyEl.appendChild(tr);
  });
}

function renderRegularization(regularization, latestSignal) {
  regularizationMetaEl.textContent = `Cfull ${latestSignal ? latestSignal.signal_date : "-"} run / observations=${regularization?.c0?.length || 0} assets`;
  d0BodyEl.innerHTML = "";
  const names = Array.isArray(regularization?.prior_subspace?.direction_names)
    ? regularization.prior_subspace.direction_names
    : [];
  const d0 = Array.isArray(regularization?.d0) ? regularization.d0 : [];
  if (!names.length || !d0.length) {
    d0BodyEl.innerHTML = '<tr><td colspan="2">No regularization diagnostics</td></tr>';
    return;
  }
  names.forEach((name, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${name}</td><td>${fmtNum(d0[index], 6)}</td>`;
    d0BodyEl.appendChild(tr);
  });
}

function renderLatestSignal(latestSignal) {
  latestBodyEl.innerHTML = "";
  factorStripEl.innerHTML = "";

  if (!latestSignal) {
    latestMetaEl.textContent = "No signal.";
    latestBodyEl.innerHTML = '<tr><td colspan="3">No signal</td></tr>';
    transferWrapEl.classList.add("hidden");
    return;
  }

  latestMetaEl.textContent = `signal=${latestSignal.signal_date} / target=${latestSignal.target_date}`;
  const factors = Array.isArray(latestSignal.factors) ? latestSignal.factors : [];
  factors.forEach((value, index) => {
    const pill = document.createElement("span");
    pill.className = "pill chip-green";
    pill.textContent = `f${index + 1}: ${fmtNum(value, 4)}`;
    factorStripEl.appendChild(pill);
  });

  const rows = Array.isArray(latestSignal.predicted_rows) ? latestSignal.predicted_rows : [];
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.symbol || "-"}</td>
      <td>${fmtNum(row.signal, 5)}</td>
      <td>${row.realized_open_to_close === null ? "-" : fmtNum(row.realized_open_to_close, 5)}</td>
    `;
    latestBodyEl.appendChild(tr);
  });
  if (!rows.length) {
    latestBodyEl.innerHTML = '<tr><td colspan="3">No predicted rows</td></tr>';
  }

  if (latestSignal.transfer_matrix) {
    transferWrapEl.classList.remove("hidden");
    transferTextEl.textContent = JSON.stringify(latestSignal.transfer_matrix, null, 2);
  } else {
    transferWrapEl.classList.add("hidden");
    transferTextEl.textContent = "";
  }
}

function renderStrategy(strategy) {
  strategyGridEl.innerHTML = "";
  const summary = strategy?.summary || {};
  [
    ["Annual Return", fmtPct(summary.annual_return_pct)],
    ["Annual Volatility", fmtPct(summary.annual_volatility_pct)],
    ["Return / Risk", summary.return_risk_ratio === null || summary.return_risk_ratio === undefined ? "-" : fmtNum(summary.return_risk_ratio, 3)],
    ["Max Drawdown", fmtPct(summary.max_drawdown_pct)],
    ["Signal Days", summary.signal_days ?? "-"],
    ["Average Breadth", summary.average_breadth === null || summary.average_breadth === undefined ? "-" : fmtNum(summary.average_breadth, 2)],
  ].forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "llg-metric";
    card.innerHTML = `<span class="label">${label}</span><strong>${value}</strong>`;
    strategyGridEl.appendChild(card);
  });
}

function renderRecent(rows) {
  recentBodyEl.innerHTML = "";
  const values = Array.isArray(rows) ? rows : [];
  if (!values.length) {
    recentBodyEl.innerHTML = '<tr><td colspan="4">No recent signals</td></tr>';
    return;
  }
  values.slice().reverse().forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.signal_date || "-"}</td>
      <td>${row.target_date || "-"}</td>
      <td>${Array.isArray(row.top_symbols) ? row.top_symbols.join(", ") : "-"}</td>
      <td>${Array.isArray(row.bottom_symbols) ? row.bottom_symbols.join(", ") : "-"}</td>
    `;
    recentBodyEl.appendChild(tr);
  });
}

function renderResult(result) {
  const summary = result?.data_summary || {};
  const range = summary.range || {};
  renderSummaryCards([
    { label: "Included US", value: Array.isArray(summary.included_us_symbols) ? summary.included_us_symbols.length : 0 },
    { label: "Included JP", value: Array.isArray(summary.included_jp_symbols) ? summary.included_jp_symbols.length : 0 },
    { label: "Signals", value: range.generated_signals ?? "-" },
    { label: "Range", value: range.from && range.to ? `${range.from} -> ${range.to}` : "-" },
  ]);
  renderChipList(activeUsEl, summary.included_us_symbols);
  renderChipList(activeJpEl, summary.included_jp_symbols);
  renderExcluded(summary.excluded_symbols);
  renderRegularization(result?.regularization, result?.latest_signal);
  renderLatestSignal(result?.latest_signal);
  renderStrategy(result?.strategy);
  renderRecent(result?.recent_signals);
}

async function bootstrap() {
  const { response, result } = await fetchJson("/api/leadlag/config");
  if (!response.ok || !result.ok) {
    setStatus(result.detail || "Failed to load defaults.", true);
    return;
  }
  setDefaults(result.defaults || {});
  setStatus("既定設定を読み込みました。");
}

async function runAnalysis() {
  if (running) return;
  running = true;
  runBtn.disabled = true;
  setStatus("Analyzing lead-lag strategy...");

  const { response, result } = await fetchJson("/api/leadlag/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadFromForm()),
  });

  running = false;
  runBtn.disabled = false;

  if (!response.ok || !result.ok) {
    setStatus(result.detail || "Analysis failed.", true);
    return;
  }
  renderResult(result);
  setStatus("Analysis completed.");
}

runBtn?.addEventListener("click", () => {
  runAnalysis().catch((error) => {
    running = false;
    runBtn.disabled = false;
    setStatus(error instanceof Error ? error.message : "Unexpected error.", true);
  });
});

window.addEventListener("DOMContentLoaded", () => {
  bootstrap().catch((error) => {
    setStatus(error instanceof Error ? error.message : "Failed to initialize.", true);
  });
});
