const portfolioSelectEl = document.getElementById("pa-portfolio-select");
const portfolioNameEl = document.getElementById("pa-portfolio-name");
const lookbackDaysEl = document.getElementById("pa-lookback-days");
const messageEl = document.getElementById("pa-message");
const newBtn = document.getElementById("pa-new-btn");
const saveBtn = document.getElementById("pa-save-btn");
const analyzeBtn = document.getElementById("pa-analyze-btn");
const deleteBtn = document.getElementById("pa-delete-btn");
const addRowButtons = document.querySelectorAll(".pa-add-row-btn");

const REGION_CONFIG = {
  jp: { label: "日本株", currencySymbol: "¥", digits: 0, placeholder: "7203 / 7203.T / NTT" },
  us: { label: "米国株", currencySymbol: "$", digits: 2, placeholder: "AAPL / Apple" },
};
const DEFAULT_LOOKBACK_DAYS = 252;
const DRAFT_SAVE_DEBOUNCE_MS = 500;

const state = {
  currentPortfolioId: "",
  portfolios: [],
  analysis: null,
  busy: false,
};
let draftSaveTimer = null;
let draftSaveInFlight = false;
let draftSaveQueued = false;
let lastDraftSignature = "";

function regionRowsEl(region) {
  return document.getElementById(`pa-${region}-rows`);
}

function metricEl(region, suffix) {
  return document.getElementById(`pa-${region}-${suffix}`);
}

function normalizeSymbolForRegion(raw, region) {
  const symbol = String(raw || "").trim().toUpperCase();
  if (!symbol) return "";
  if (region === "jp" && /^\d{4,5}$/.test(symbol)) {
    return `${symbol}.T`;
  }
  return symbol;
}

function normalizeLookbackDays(rawValue) {
  const parsed = Number.parseInt(String(rawValue ?? ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_LOOKBACK_DAYS;
}

function formatCount(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toLocaleString("en-US") : "-";
}

function formatPercent(value, digits = 2) {
  const num = Number(value);
  return Number.isFinite(num) ? `${num.toFixed(digits)}%` : "-";
}

function formatMoney(region, value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  const config = REGION_CONFIG[region];
  return `${config.currencySymbol}${num.toLocaleString("en-US", {
    minimumFractionDigits: config.digits,
    maximumFractionDigits: config.digits,
  })}`;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setMessage(text, kind = "info") {
  messageEl.textContent = text || "";
  messageEl.classList.remove("error", "pa-success");
  if (kind === "error") {
    messageEl.classList.add("error");
  } else if (kind === "success") {
    messageEl.classList.add("pa-success");
  }
}

function setBusy(isBusy) {
  state.busy = isBusy;
  [portfolioSelectEl, portfolioNameEl, lookbackDaysEl, newBtn, saveBtn, analyzeBtn, deleteBtn, ...addRowButtons].forEach((el) => {
    if (el) el.disabled = isBusy;
  });
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.detail || payload.message || `Request failed (${response.status})`);
  }
  return payload;
}

function buildHoldingRow(region, holding = {}, analysisHolding = null) {
  const symbol = escapeHtml(holding.symbol || "");
  const rawQuantity = String(holding.quantity ?? "").trim();
  const quantity = Number(rawQuantity);
  const quantityValue = rawQuantity || (Number.isFinite(quantity) && quantity > 0 ? quantity : "");
  const lastPrice = analysisHolding ? formatMoney(region, analysisHolding.last_price) : "-";
  const marketValue = analysisHolding ? formatMoney(region, analysisHolding.market_value) : "-";
  const weight = analysisHolding ? formatPercent(analysisHolding.weight) : "-";
  let priceLabel = "saved row";
  if (analysisHolding) {
    if (analysisHolding.last_price_source === "market") {
      priceLabel = "market";
    } else if (analysisHolding.last_price_source === "historical_close") {
      priceLabel = analysisHolding.last_close_date
        ? `historical close ${analysisHolding.last_close_date}`
        : "historical close";
    } else if (analysisHolding.last_price_source === "stale_historical_close") {
      priceLabel = analysisHolding.last_close_date
        ? `stale close ${analysisHolding.last_close_date}`
        : "stale close";
    } else {
      priceLabel = "price unavailable";
    }
  }
  const metaText = analysisHolding
    ? [
      priceLabel,
      analysisHolding.risk_included ? "risk included" : "risk excluded",
    ].join(" / ")
    : "saved row";

  const tr = document.createElement("tr");
  if (
    analysisHolding
    && (
      !analysisHolding.risk_included
      || analysisHolding.last_price_source === "stale_historical_close"
    )
  ) {
    tr.classList.add("pa-row-muted");
  }
  tr.innerHTML = `
    <td class="pa-symbol-cell">
      <input type="text" name="symbol" value="${symbol}" placeholder="${escapeHtml(REGION_CONFIG[region].placeholder)}" />
      <span class="pa-row-meta">${escapeHtml(metaText)}</span>
    </td>
    <td class="pa-qty-cell">
      <input type="number" name="quantity" min="0.0001" step="0.0001" value="${quantityValue}" placeholder="0" />
    </td>
    <td class="pa-value-cell">${escapeHtml(lastPrice)}</td>
    <td class="pa-value-cell">${escapeHtml(marketValue)}</td>
    <td class="pa-value-cell">${escapeHtml(weight)}</td>
    <td class="pa-action-cell">
      <button type="button" class="minor-action pa-remove-row-btn" aria-label="Remove row">✕</button>
    </td>
  `;
  return tr;
}

function renderRegionRows(region, holdings = [], analysisRegion = null) {
  const tbody = regionRowsEl(region);
  const analysisMap = new Map(
    Array.isArray(analysisRegion?.holdings)
      ? analysisRegion.holdings.map((item) => [normalizeSymbolForRegion(item.symbol, region), item])
      : []
  );

  tbody.innerHTML = "";
  const sourceRows = Array.isArray(holdings) && holdings.length > 0 ? holdings : [];
  sourceRows.forEach((holding) => {
    const symbolKey = normalizeSymbolForRegion(holding.symbol, region);
    tbody.appendChild(buildHoldingRow(region, holding, analysisMap.get(symbolKey) || null));
  });

  const minimumRows = Math.max(3, sourceRows.length + 1);
  while (tbody.children.length < minimumRows) {
    tbody.appendChild(buildHoldingRow(region));
  }
}

function clearRegionMetrics(region) {
  [
    "market-value",
    "holdings-count",
    "top-holding",
    "coverage",
    "ann-vol",
    "var95",
    "es95",
    "max-dd",
    "effective",
    "window",
  ].forEach((suffix) => {
    metricEl(region, suffix).textContent = "-";
  });
  metricEl(region, "risk-note").textContent = "Analyze を押すとここにリスク計算の補足を表示します。";
  metricEl(region, "warnings").innerHTML = "";
}

function renderRegionMetrics(region, payload) {
  if (!payload) {
    clearRegionMetrics(region);
    return;
  }

  const summary = payload.summary || {};
  const risk = payload.risk || {};

  metricEl(region, "market-value").textContent = formatMoney(region, summary.market_value);
  metricEl(region, "holdings-count").textContent = `${formatCount(summary.holdings_count)} / priced ${formatCount(summary.priced_holdings_count)}`;
  metricEl(region, "top-holding").textContent = summary.top_holding_symbol
    ? `${summary.top_holding_symbol} (${formatPercent(summary.top_holding_weight_pct)})`
    : "-";
  metricEl(region, "coverage").textContent = formatPercent(summary.risk_coverage_pct);

  metricEl(region, "ann-vol").textContent = formatPercent(risk.annualized_volatility_pct);
  metricEl(region, "var95").textContent = risk.value_at_risk_95_pct != null
    ? `${formatPercent(risk.value_at_risk_95_pct)} / ${formatMoney(region, risk.value_at_risk_95_amount)}`
    : "-";
  metricEl(region, "es95").textContent = risk.expected_shortfall_95_pct != null
    ? `${formatPercent(risk.expected_shortfall_95_pct)} / ${formatMoney(region, risk.expected_shortfall_95_amount)}`
    : "-";
  metricEl(region, "max-dd").textContent = formatPercent(risk.max_drawdown_pct);
  metricEl(region, "effective").textContent = Number.isFinite(Number(summary.effective_holdings))
    ? Number(summary.effective_holdings).toFixed(2)
    : "-";
  metricEl(region, "window").textContent = risk.analysis_start && risk.analysis_end
    ? `${risk.analysis_start} -> ${risk.analysis_end} (${formatCount(risk.observation_count)} obs)`
    : `${formatCount(risk.observation_count)} obs`;
  metricEl(region, "risk-note").textContent = risk.note || "Coverage 100% でリスク指標を算出しました。";

  const warningsEl = metricEl(region, "warnings");
  warningsEl.innerHTML = "";
  (Array.isArray(payload.warnings) ? payload.warnings : []).forEach((warning) => {
    const li = document.createElement("li");
    li.textContent = warning;
    warningsEl.appendChild(li);
  });
}

function renderPortfolioOptions() {
  const previousValue = state.currentPortfolioId || "";
  portfolioSelectEl.innerHTML = '<option value="">新規ポートフォリオ</option>';
  state.portfolios.forEach((portfolio) => {
    const option = document.createElement("option");
    option.value = portfolio.portfolio_id;
    option.textContent = portfolio.name;
    portfolioSelectEl.appendChild(option);
  });
  portfolioSelectEl.value = previousValue && state.portfolios.some((item) => item.portfolio_id === previousValue)
    ? previousValue
    : "";
}

function resetComposer() {
  state.currentPortfolioId = "";
  state.analysis = null;
  portfolioSelectEl.value = "";
  portfolioNameEl.value = "";
  renderRegionRows("jp", []);
  renderRegionRows("us", []);
  clearRegionMetrics("jp");
  clearRegionMetrics("us");
  setMessage("新規ポートフォリオを編集中です。", "info");
}

function loadPortfolio(portfolio) {
  if (!portfolio) {
    resetComposer();
    return;
  }
  state.currentPortfolioId = portfolio.portfolio_id || "";
  state.analysis = null;
  portfolioSelectEl.value = state.currentPortfolioId;
  portfolioNameEl.value = portfolio.name || "";
  renderRegionRows("jp", portfolio.jp_holdings || []);
  renderRegionRows("us", portfolio.us_holdings || []);
  clearRegionMetrics("jp");
  clearRegionMetrics("us");
  setMessage(`"${portfolio.name}" を読み込みました。`, "info");
}

function collectRegionHoldings(region) {
  const rows = Array.from(regionRowsEl(region).querySelectorAll("tr"));
  const holdings = [];

  rows.forEach((row) => {
    const symbolInput = row.querySelector('input[name="symbol"]');
    const quantityInput = row.querySelector('input[name="quantity"]');
    const rawSymbol = String(symbolInput?.value || "").trim();
    const rawQuantity = String(quantityInput?.value || "").trim();

    if (!rawSymbol && !rawQuantity) {
      return;
    }
    if (!rawSymbol) {
      throw new Error(`${REGION_CONFIG[region].label}: symbol is required.`);
    }
    const quantity = Number(rawQuantity);
    if (!Number.isFinite(quantity) || quantity <= 0) {
      throw new Error(`${REGION_CONFIG[region].label}: quantity must be greater than 0 for ${rawSymbol}.`);
    }
    holdings.push({
      symbol: rawSymbol,
      quantity,
    });
  });

  return holdings;
}

function buildEditorPayload() {
  return {
    portfolio_id: state.currentPortfolioId || null,
    name: portfolioNameEl.value.trim(),
    lookback_days: normalizeLookbackDays(lookbackDaysEl.value),
    jp_holdings: collectRegionHoldings("jp"),
    us_holdings: collectRegionHoldings("us"),
  };
}

function collectRegionDraftRows(region) {
  return Array.from(regionRowsEl(region).querySelectorAll("tr"))
    .map((row) => {
      const symbolInput = row.querySelector('input[name="symbol"]');
      const quantityInput = row.querySelector('input[name="quantity"]');
      return {
        symbol: String(symbolInput?.value || "").trim(),
        quantity: String(quantityInput?.value || "").trim(),
      };
    })
    .filter((row) => row.symbol || row.quantity);
}

function buildDraftPayload() {
  return {
    portfolio_id: state.currentPortfolioId || null,
    name: portfolioNameEl.value.trim(),
    lookback_days: normalizeLookbackDays(lookbackDaysEl.value),
    jp_rows: collectRegionDraftRows("jp"),
    us_rows: collectRegionDraftRows("us"),
  };
}

function draftHasContent(draft) {
  if (!draft) return false;
  if (String(draft.portfolio_id || "").trim()) return true;
  if (String(draft.name || "").trim()) return true;
  if (Array.isArray(draft.jp_rows) && draft.jp_rows.length > 0) return true;
  if (Array.isArray(draft.us_rows) && draft.us_rows.length > 0) return true;
  return normalizeLookbackDays(draft.lookback_days) !== DEFAULT_LOOKBACK_DAYS;
}

function draftHasRows(draft) {
  return (Array.isArray(draft?.jp_rows) && draft.jp_rows.length > 0)
    || (Array.isArray(draft?.us_rows) && draft.us_rows.length > 0);
}

function canAnalyzeEditorPayload() {
  try {
    const payload = buildEditorPayload();
    return payload.jp_holdings.length > 0 || payload.us_holdings.length > 0;
  } catch (_error) {
    return false;
  }
}

async function fetchDraft() {
  const payload = await requestJson("/api/portfolio-analysis/draft");
  return payload?.draft || null;
}

async function persistDraftNow() {
  const currentPayload = buildDraftPayload();
  const currentSignature = JSON.stringify(currentPayload);
  if (currentSignature === lastDraftSignature) {
    return;
  }
  if (draftSaveInFlight) {
    draftSaveQueued = true;
    return;
  }

  draftSaveInFlight = true;
  try {
    do {
      draftSaveQueued = false;
      const payload = buildDraftPayload();
      const signature = JSON.stringify(payload);
      if (signature === lastDraftSignature) {
        continue;
      }
      await requestJson("/api/portfolio-analysis/draft", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      lastDraftSignature = signature;
    } while (draftSaveQueued);
  } catch (error) {
    setMessage(error instanceof Error ? error.message : "編集中ドラフトの保存に失敗しました。", "error");
  } finally {
    draftSaveInFlight = false;
  }
}

function scheduleDraftSave() {
  if (draftSaveTimer) {
    window.clearTimeout(draftSaveTimer);
  }
  draftSaveTimer = window.setTimeout(() => {
    draftSaveTimer = null;
    void persistDraftNow();
  }, DRAFT_SAVE_DEBOUNCE_MS);
}

function loadDraft(draft) {
  const draftPortfolioId = String(draft?.portfolio_id || "").trim();
  state.currentPortfolioId = state.portfolios.some((item) => item.portfolio_id === draftPortfolioId)
    ? draftPortfolioId
    : "";
  state.analysis = null;
  portfolioSelectEl.value = state.currentPortfolioId;
  portfolioNameEl.value = String(draft?.name || "").trim();
  lookbackDaysEl.value = String(normalizeLookbackDays(draft?.lookback_days));
  renderRegionRows("jp", Array.isArray(draft?.jp_rows) ? draft.jp_rows : []);
  renderRegionRows("us", Array.isArray(draft?.us_rows) ? draft.us_rows : []);
  clearRegionMetrics("jp");
  clearRegionMetrics("us");
  lastDraftSignature = JSON.stringify(buildDraftPayload());
  setMessage("前回の編集中ドラフトを復元しました。", "info");
}

function applyAnalysis(payload) {
  state.analysis = payload;
  renderRegionRows("jp", payload.portfolio?.jp_holdings || [], payload.regions?.jp || null);
  renderRegionRows("us", payload.portfolio?.us_holdings || [], payload.regions?.us || null);
  renderRegionMetrics("jp", payload.regions?.jp || null);
  renderRegionMetrics("us", payload.regions?.us || null);
}

async function fetchPortfolios() {
  const payload = await requestJson("/api/portfolio-analysis/portfolios");
  state.portfolios = Array.isArray(payload.portfolios) ? payload.portfolios : [];
  renderPortfolioOptions();
}

async function analyzeCurrentPortfolio() {
  const payload = buildEditorPayload();
  setBusy(true);
  try {
    const analysisPayload = await requestJson("/api/portfolio-analysis/analyze", {
      method: "POST",
      body: JSON.stringify({
        jp_holdings: payload.jp_holdings,
        us_holdings: payload.us_holdings,
        lookback_days: payload.lookback_days,
      }),
    });
    applyAnalysis(analysisPayload);
    await persistDraftNow();
    setMessage("リスク分析を更新しました。", "success");
  } finally {
    setBusy(false);
  }
}

async function saveCurrentPortfolio() {
  const payload = buildEditorPayload();
  if (!payload.name) {
    throw new Error("Portfolio name is required.");
  }

  setBusy(true);
  try {
    const savePayload = await requestJson("/api/portfolio-analysis/portfolios", {
      method: "POST",
      body: JSON.stringify({
        portfolio_id: payload.portfolio_id,
        name: payload.name,
        jp_holdings: payload.jp_holdings,
        us_holdings: payload.us_holdings,
      }),
    });
    state.portfolios = Array.isArray(savePayload.portfolios) ? savePayload.portfolios : [];
    state.currentPortfolioId = savePayload.portfolio?.portfolio_id || "";
    renderPortfolioOptions();
    portfolioSelectEl.value = state.currentPortfolioId;
    portfolioNameEl.value = savePayload.portfolio?.name || payload.name;
    renderRegionRows("jp", savePayload.portfolio?.jp_holdings || payload.jp_holdings, state.analysis?.regions?.jp || null);
    renderRegionRows("us", savePayload.portfolio?.us_holdings || payload.us_holdings, state.analysis?.regions?.us || null);
    await persistDraftNow();
    setMessage("ポートフォリオを保存しました。", "success");
  } finally {
    setBusy(false);
  }
}

async function deleteCurrentPortfolio() {
  if (!state.currentPortfolioId) {
    throw new Error("Delete する保存済みポートフォリオがありません。");
  }

  setBusy(true);
  try {
    const payload = await requestJson(`/api/portfolio-analysis/portfolios/${encodeURIComponent(state.currentPortfolioId)}`, {
      method: "DELETE",
    });
    state.portfolios = Array.isArray(payload.portfolios) ? payload.portfolios : [];
    renderPortfolioOptions();
    resetComposer();
    await persistDraftNow();
    setMessage("保存済みポートフォリオを削除しました。", "success");
  } finally {
    setBusy(false);
  }
}

async function initialize() {
  renderRegionRows("jp", []);
  renderRegionRows("us", []);
  try {
    await fetchPortfolios();
    const draft = await fetchDraft();
    if (draftHasContent(draft)) {
      loadDraft(draft);
      if (draftHasRows(draft) && canAnalyzeEditorPayload()) {
        await analyzeCurrentPortfolio();
      }
      return;
    }
    if (state.portfolios.length > 0) {
      loadPortfolio(state.portfolios[0]);
      await analyzeCurrentPortfolio();
      return;
    }
    resetComposer();
    lastDraftSignature = JSON.stringify(buildDraftPayload());
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error), "error");
  }
}

portfolioSelectEl?.addEventListener("change", async (event) => {
  const portfolioId = String(event.target?.value || "");
  if (!portfolioId) {
    resetComposer();
    await persistDraftNow();
    return;
  }
  const portfolio = state.portfolios.find((item) => item.portfolio_id === portfolioId);
  loadPortfolio(portfolio || null);
  await persistDraftNow();
  try {
    await analyzeCurrentPortfolio();
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error), "error");
  }
});

newBtn?.addEventListener("click", () => {
  resetComposer();
  scheduleDraftSave();
});

saveBtn?.addEventListener("click", async () => {
  try {
    await saveCurrentPortfolio();
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error), "error");
  }
});

analyzeBtn?.addEventListener("click", async () => {
  try {
    await analyzeCurrentPortfolio();
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error), "error");
  }
});

deleteBtn?.addEventListener("click", async () => {
  try {
    await deleteCurrentPortfolio();
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error), "error");
  }
});

addRowButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const region = button.getAttribute("data-region");
    if (!region) return;
    regionRowsEl(region).appendChild(buildHoldingRow(region));
    scheduleDraftSave();
  });
});

["jp", "us"].forEach((region) => {
  regionRowsEl(region)?.addEventListener("input", () => {
    scheduleDraftSave();
  });
  regionRowsEl(region)?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element) || !target.closest(".pa-remove-row-btn")) {
      return;
    }
    const row = target.closest("tr");
    if (!row) return;
    row.remove();
    if (regionRowsEl(region).children.length === 0) {
      regionRowsEl(region).appendChild(buildHoldingRow(region));
    }
    scheduleDraftSave();
  });
});

portfolioNameEl?.addEventListener("input", () => {
  scheduleDraftSave();
});

lookbackDaysEl?.addEventListener("change", () => {
  scheduleDraftSave();
});

window.addEventListener("DOMContentLoaded", initialize);
