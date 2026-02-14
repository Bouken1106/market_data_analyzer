const formEl = document.getElementById("stg-form");
const statusEl = document.getElementById("stg-status");
const runBtn = document.getElementById("stg-run");

const symbolsEl = document.getElementById("stg-symbols");
const methodEl = document.getElementById("stg-method");
const monthsEl = document.getElementById("stg-months");
const lookbackEl = document.getElementById("stg-lookback");
const rebalanceFreqEl = document.getElementById("stg-rebalance-frequency");
const rebalanceThresholdEl = document.getElementById("stg-rebalance-threshold");
const maxWeightEl = document.getElementById("stg-max-weight");
const initialCapitalEl = document.getElementById("stg-initial-capital");
const commissionBpsEl = document.getElementById("stg-commission-bps");
const slippageBpsEl = document.getElementById("stg-slippage-bps");
const benchmarkEl = document.getElementById("stg-benchmark");
const minTradeValueEl = document.getElementById("stg-min-trade-value");
const refreshEl = document.getElementById("stg-refresh");

const cagrEl = document.getElementById("stg-cagr");
const totalReturnEl = document.getElementById("stg-total-return");
const volatilityEl = document.getElementById("stg-volatility");
const sharpeEl = document.getElementById("stg-sharpe");
const mddEl = document.getElementById("stg-mdd");
const benchCagrEl = document.getElementById("stg-bench-cagr");
const dataSummaryEl = document.getElementById("stg-data-summary");
const planStatsEl = document.getElementById("stg-plan-stats");
const allocationBodyEl = document.getElementById("stg-allocation-body");
const tradesBodyEl = document.getElementById("stg-trades-body");
const tradesNoteEl = document.getElementById("stg-trades-note");

let running = false;

function closeParamHelpPopovers(exceptDetail = null) {
  const openDetails = document.querySelectorAll("details.param-help[open]");
  openDetails.forEach((detail) => {
    if (detail === exceptDetail) return;
    detail.removeAttribute("open");
  });
}

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function syncButtons() {
  runBtn.disabled = running;
}

function fmtNum(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function fmtPct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${num.toFixed(2)}%`;
}

function fmtMoney(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json().catch(() => ({}));
  return { response, result };
}

function payloadFromForm() {
  return {
    symbols: String(symbolsEl.value || "").trim(),
    method: String(methodEl.value || "inverse_volatility"),
    months: Number(monthsEl.value || 36),
    lookback_days: Number(lookbackEl.value || 126),
    rebalance_frequency: String(rebalanceFreqEl.value || "monthly"),
    rebalance_threshold_pct: Number(rebalanceThresholdEl.value || 5),
    max_weight: Number(maxWeightEl.value || 0.35),
    initial_capital: Number(initialCapitalEl.value || 1000000),
    commission_bps: Number(commissionBpsEl.value || 2),
    slippage_bps: Number(slippageBpsEl.value || 3),
    benchmark_symbol: String(benchmarkEl.value || "SPY").trim(),
    min_trade_value: Number(minTradeValueEl.value || 100),
    refresh: Boolean(refreshEl.checked),
  };
}

function renderSummary(backtest, benchmark, dataSummary) {
  const metrics = backtest?.metrics || {};
  cagrEl.textContent = fmtPct(metrics.cagr_pct);
  totalReturnEl.textContent = fmtPct(metrics.total_return_pct);
  volatilityEl.textContent = fmtPct(metrics.volatility_pct);
  sharpeEl.textContent = fmtNum(metrics.sharpe, 3);
  mddEl.textContent = fmtPct(metrics.max_drawdown_pct);
  benchCagrEl.textContent = fmtPct(benchmark?.cagr_pct);
  dataSummaryEl.textContent = dataSummary
    ? `Data range: ${dataSummary.from || "-"} -> ${dataSummary.to || "-"} | points=${dataSummary.price_points || 0}`
    : "";
}

function renderAllocation(plan) {
  allocationBodyEl.innerHTML = "";
  const symbols = Array.isArray(plan?.symbols) ? plan.symbols : [];
  if (!symbols.length) {
    allocationBodyEl.innerHTML = '<tr><td colspan="2">No allocation</td></tr>';
  } else {
    symbols.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.symbol || "-"}</td>
        <td>${fmtPct((Number(item.weight) || 0) * 100)}</td>
      `;
      allocationBodyEl.appendChild(tr);
    });
  }

  const expectedReturn = fmtPct(plan?.expected_return_pct);
  const expectedVol = fmtPct(plan?.expected_volatility_pct);
  planStatsEl.textContent = `Expected annual return: ${expectedReturn} / Expected annual volatility: ${expectedVol}`;
}

function renderTrades(tradeProposals) {
  tradesBodyEl.innerHTML = "";
  const trades = Array.isArray(tradeProposals?.trades) ? tradeProposals.trades : [];
  if (!trades.length) {
    tradesBodyEl.innerHTML = '<tr><td colspan="6">No trade proposal</td></tr>';
  } else {
    trades.forEach((item) => {
      const side = String(item.side || "-");
      const sideClass = side === "buy" ? "pf-side-buy" : side === "sell" ? "pf-side-sell" : "";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.symbol || "-"}</td>
        <td class="${sideClass}">${side}</td>
        <td>${fmtNum(item.quantity, 4)}</td>
        <td>${fmtMoney(item.price)}</td>
        <td>${fmtMoney(item.delta_value)}</td>
        <td>${fmtPct(item.target_weight_pct)}</td>
      `;
      tradesBodyEl.appendChild(tr);
    });
  }

  const skipped = Array.isArray(tradeProposals?.skipped) ? tradeProposals.skipped : [];
  tradesNoteEl.textContent = skipped.length > 0
    ? `Skipped ${skipped.length} symbols due to unavailable price.`
    : "";
}

async function runStrategy() {
  running = true;
  syncButtons();
  setStatus("Running strategy evaluation...");

  const payload = payloadFromForm();
  const { response, result } = await fetchJson("/api/strategy/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  running = false;
  syncButtons();

  if (!response.ok) {
    setStatus(result.detail || "Strategy evaluation failed.", true);
    return;
  }

  renderSummary(result.backtest, result.benchmark, result.data_summary);
  renderAllocation(result.allocation_plan);
  renderTrades(result.trade_proposals);
  setStatus("Strategy evaluation completed.");
}

formEl?.addEventListener("submit", (event) => {
  event.preventDefault();
  runStrategy().catch((error) => {
    running = false;
    syncButtons();
    setStatus(error instanceof Error ? error.message : "Unexpected error occurred.", true);
  });
});

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const opened = target.closest("details.param-help");
  if (opened) {
    closeParamHelpPopovers(opened);
    return;
  }
  closeParamHelpPopovers();
});
