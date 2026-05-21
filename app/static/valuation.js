const valuationForm = document.getElementById("valuation-form");
const valuationSymbolArea = document.getElementById("valuation-symbol-area");
const valuationSymbolInput = document.getElementById("valuation-symbol");
const valuationSymbolDropdown = document.getElementById("valuation-symbol-dropdown");
const valuationLoadBtn = document.getElementById("valuation-load");
const valuationRefreshBtn = document.getElementById("valuation-refresh");
const valuationMetaEl = document.getElementById("valuation-meta");
const valuationOverviewLink = document.getElementById("valuation-overview-link");
const valuationSymbolLabelEl = document.getElementById("valuation-symbol-label");
const valuationCurrentEl = document.getElementById("valuation-current");
const valuationMedianEl = document.getElementById("valuation-median");
const valuationUpsideEl = document.getElementById("valuation-upside");
const valuationCountEl = document.getElementById("valuation-count");
const valuationBodyEl = document.getElementById("valuation-body");
const valuationFairPerInput = document.getElementById("valuation-fair-per");
const valuationFairPbrInput = document.getElementById("valuation-fair-pbr");
const valuationFairPsrInput = document.getElementById("valuation-fair-psr");
const valuationFairEvEbitdaInput = document.getElementById("valuation-fair-ev-ebitda");
const valuationFairPFcfInput = document.getElementById("valuation-fair-p-fcf");
const valuationRiskFreeInput = document.getElementById("valuation-risk-free-rate");
const valuationFcfGrowthInput = document.getElementById("valuation-fcf-growth-rate");
const valuationTerminalGrowthInput = document.getElementById("valuation-terminal-growth-rate");

const MAX_DROPDOWN_ITEMS = 40;
let symbolCatalog = [];

function normalizeSymbol(value) {
  return String(value || "").trim().toUpperCase();
}

function nullableNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function readValuationNumber(inputEl) {
  if (!inputEl) return null;
  return nullableNumber(inputEl.value);
}

function currencyPrefix(currency) {
  const code = String(currency || "").toUpperCase();
  if (code === "JPY") return "¥";
  if (code === "EUR") return "€";
  if (code === "GBP") return "£";
  return "$";
}

function formatCurrencyValue(value, currency = "USD") {
  const num = nullableNumber(value);
  if (num === null) return "-";
  const code = String(currency || "").toUpperCase();
  const digits = code === "JPY" ? 0 : 2;
  return `${currencyPrefix(code)}${num.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

function formatPercent(value) {
  const num = nullableNumber(value);
  if (num === null) return "-";
  const sign = num > 0 ? "+" : "";
  return `${sign}${num.toFixed(2)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setValuationMeta(message, isError = false) {
  if (!valuationMetaEl) return;
  valuationMetaEl.textContent = message || "";
  valuationMetaEl.classList.toggle("error", Boolean(isError));
}

function setLoading(isLoading) {
  if (valuationLoadBtn) valuationLoadBtn.disabled = isLoading;
  if (valuationRefreshBtn) valuationRefreshBtn.disabled = isLoading;
}

function setMetricTrendClass(el, trend) {
  if (!el) return;
  el.classList.toggle("metric-good", trend === "good");
  el.classList.toggle("metric-bad", trend === "bad");
}

function updateOverviewLink(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!valuationOverviewLink) return;
  if (!normalized) {
    valuationOverviewLink.href = "/historical/AAPL";
    valuationOverviewLink.setAttribute("aria-disabled", "true");
    return;
  }
  valuationOverviewLink.href = `/historical/${encodeURIComponent(normalized)}`;
  valuationOverviewLink.removeAttribute("aria-disabled");
}

function setValuationPlaceholder(message = "Symbolを入力して表示を押してください。") {
  if (valuationSymbolLabelEl) valuationSymbolLabelEl.textContent = "-";
  if (valuationCurrentEl) valuationCurrentEl.textContent = "-";
  if (valuationMedianEl) valuationMedianEl.textContent = "-";
  if (valuationUpsideEl) {
    valuationUpsideEl.textContent = "-";
    setMetricTrendClass(valuationUpsideEl, "neutral");
  }
  if (valuationCountEl) valuationCountEl.textContent = "-";
  if (valuationBodyEl) valuationBodyEl.innerHTML = `<tr><td colspan="4">${escapeHtml(message)}</td></tr>`;
  updateOverviewLink("");
}

function valuationParams(refresh = false, cacheOnly = true) {
  const params = new URLSearchParams();
  params.set("refresh", String(Boolean(refresh)));
  params.set("cache_only", String(Boolean(cacheOnly)));

  [
    ["fair_per", readValuationNumber(valuationFairPerInput)],
    ["fair_pbr", readValuationNumber(valuationFairPbrInput)],
    ["fair_psr", readValuationNumber(valuationFairPsrInput)],
    ["fair_ev_ebitda", readValuationNumber(valuationFairEvEbitdaInput)],
    ["fair_p_fcf", readValuationNumber(valuationFairPFcfInput)],
    ["risk_free_rate", readValuationNumber(valuationRiskFreeInput)],
    ["fcf_growth_rate", readValuationNumber(valuationFcfGrowthInput)],
    ["terminal_growth_rate", readValuationNumber(valuationTerminalGrowthInput)],
  ].forEach(([key, value]) => {
    if (value !== null) params.set(key, String(value));
  });

  return params;
}

function renderValuation(payload) {
  const symbol = normalizeSymbol(payload?.symbol || valuationSymbolInput?.value);
  const name = String(payload?.company_name || payload?.name || "").trim();
  const currency = payload?.currency || "USD";
  const summary = payload?.summary || {};
  const current = nullableNumber(payload?.current_price);
  const medianPrice = nullableNumber(summary.median_price);
  const medianUpside = nullableNumber(summary.median_upside_pct);
  const calculated = nullableNumber(summary.calculated_count);
  const total = nullableNumber(summary.method_count);

  if (valuationSymbolLabelEl) valuationSymbolLabelEl.textContent = name ? `${symbol} / ${name}` : symbol || "-";
  if (valuationCurrentEl) valuationCurrentEl.textContent = formatCurrencyValue(current, currency);
  if (valuationMedianEl) valuationMedianEl.textContent = formatCurrencyValue(medianPrice, currency);
  if (valuationUpsideEl) {
    valuationUpsideEl.textContent = formatPercent(medianUpside);
    setMetricTrendClass(valuationUpsideEl, medianUpside === null ? "neutral" : (medianUpside >= 0 ? "good" : "bad"));
  }
  if (valuationCountEl) {
    valuationCountEl.textContent = calculated !== null && total !== null ? `${calculated} / ${total}` : "-";
  }
  updateOverviewLink(symbol);

  if (!valuationBodyEl) return;
  const rows = Array.isArray(payload?.valuations) ? payload.valuations : [];
  if (rows.length === 0) {
    valuationBodyEl.innerHTML = '<tr><td colspan="4">No valuation methods returned.</td></tr>';
    return;
  }

  valuationBodyEl.innerHTML = rows.map((item) => {
    const theoretical = nullableNumber(item?.theoretical_price);
    const upside = nullableNumber(item?.upside_pct);
    const calculatedLabel = item?.is_calculated ? "calculated" : (item?.unavailable_reason || "unavailable");
    const upsideClass = upside === null ? "" : (upside >= 0 ? "metric-good" : "metric-bad");
    return `
      <tr>
        <td>${escapeHtml(item?.method_name || "-")}</td>
        <td>${formatCurrencyValue(theoretical, currency)}</td>
        <td class="${upsideClass}">${formatPercent(upside)}</td>
        <td>${escapeHtml(calculatedLabel)}</td>
      </tr>
    `;
  }).join("");
}

async function loadValuation(refresh = false, cacheOnly = true) {
  const symbol = normalizeSymbol(valuationSymbolInput?.value);
  if (!symbol) {
    setValuationPlaceholder();
    setValuationMeta("Symbolを入力してください。", true);
    valuationSymbolInput?.focus();
    return;
  }

  valuationSymbolInput.value = symbol;
  setLoading(true);
  setValuationMeta(refresh ? "理論株価を再取得しています..." : "理論株価を読み込んでいます...");

  try {
    let { response, result } = await requestValuation(symbol, refresh, cacheOnly);
    let staleFallbackDetail = "";
    if (!response.ok || !result.ok) {
      if (!refresh && !cacheOnly) {
        staleFallbackDetail = result.detail || "fresh valuation failed";
        ({ response, result } = await requestValuation(symbol, false, true));
      }
    }
    if (!response.ok || !result.ok) {
      setValuationPlaceholder("理論株価を取得できませんでした。");
      setValuationMeta(result.detail || "Failed to load valuation", true);
      return;
    }

    renderValuation(result);
    const inputStatus = result.input_status || {};
    const count = result.summary?.calculated_count ?? 0;
    const total = result.summary?.method_count ?? 0;
    const fundamentals = inputStatus.fundamentals_source || inputStatus.fundamentals_error || "no fundamentals";
    const fallbackNote = staleFallbackDetail ? ` Fresh fetch failed: ${staleFallbackDetail}` : "";
    setValuationMeta(`Loaded valuation (${count}/${total} methods, fundamentals: ${fundamentals}).${fallbackNote}`, Boolean(staleFallbackDetail));
    history.replaceState(null, "", `/valuation?symbol=${encodeURIComponent(symbol)}`);
  } catch (error) {
    setValuationPlaceholder("理論株価を取得できませんでした。");
    setValuationMeta(error instanceof Error ? error.message : "Failed to load valuation", true);
  } finally {
    setLoading(false);
  }
}

async function requestValuation(symbol, refresh, cacheOnly) {
  const params = valuationParams(refresh, cacheOnly);
  const response = await fetch(`/api/valuation/${encodeURIComponent(symbol)}?${params.toString()}`);
  const result = await response.json().catch(() => ({}));
  return { response, result };
}

function symbolOptionLabel(item) {
  const symbol = normalizeSymbol(item?.symbol);
  const name = String(item?.name || "").trim();
  const exchange = String(item?.exchange || "").trim();
  const suffix = [name, exchange].filter(Boolean).join(" / ");
  return suffix ? `${symbol} - ${suffix}` : symbol;
}

function matchingSymbols(query) {
  const normalized = normalizeSymbol(query);
  if (!normalized) return [];
  return symbolCatalog
    .filter((item) => {
      const symbol = normalizeSymbol(item?.symbol);
      const name = String(item?.name || "").toUpperCase();
      return symbol.startsWith(normalized) || name.includes(normalized);
    })
    .slice(0, MAX_DROPDOWN_ITEMS);
}

function hideDropdown() {
  if (!valuationSymbolDropdown) return;
  valuationSymbolDropdown.classList.add("hidden");
}

function renderDropdown() {
  if (!valuationSymbolDropdown || !valuationSymbolInput) return;
  const matches = matchingSymbols(valuationSymbolInput.value);
  if (matches.length === 0) {
    hideDropdown();
    return;
  }
  valuationSymbolDropdown.innerHTML = matches.map((item) => {
    const symbol = normalizeSymbol(item?.symbol);
    return `<button type="button" class="dropdown-item" data-symbol="${escapeHtml(symbol)}">${escapeHtml(symbolOptionLabel(item))}</button>`;
  }).join("");
  valuationSymbolDropdown.classList.remove("hidden");
}

async function loadSymbolCatalog() {
  try {
    const response = await fetch("/api/symbol-catalog?cache_only=true");
    const result = await response.json().catch(() => ({}));
    if (!response.ok || !result.ok || !Array.isArray(result.symbols)) return;
    symbolCatalog = result.symbols
      .map((item) => ({
        symbol: normalizeSymbol(item?.symbol),
        name: String(item?.name || "").trim(),
        exchange: String(item?.exchange || "").trim(),
      }))
      .filter((item) => item.symbol);
  } catch {
    symbolCatalog = [];
  }
}

valuationForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideDropdown();
  await loadValuation(false, false);
});

valuationRefreshBtn?.addEventListener("click", async () => {
  hideDropdown();
  await loadValuation(true, false);
});

valuationSymbolInput?.addEventListener("input", () => {
  valuationSymbolInput.value = normalizeSymbol(valuationSymbolInput.value);
  renderDropdown();
});

valuationSymbolInput?.addEventListener("focus", renderDropdown);

valuationSymbolDropdown?.addEventListener("click", async (event) => {
  const option = event.target instanceof Element ? event.target.closest(".dropdown-item") : null;
  if (!option) return;
  const symbol = normalizeSymbol(option.getAttribute("data-symbol"));
  if (!symbol) return;
  valuationSymbolInput.value = symbol;
  hideDropdown();
  await loadValuation(false, false);
});

document.addEventListener("click", (event) => {
  if (!valuationSymbolArea || valuationSymbolArea.contains(event.target)) return;
  hideDropdown();
});

async function init() {
  setValuationPlaceholder();
  await loadSymbolCatalog();
  const params = new URLSearchParams(window.location.search);
  const symbol = normalizeSymbol(params.get("symbol"));
  if (symbol) {
    valuationSymbolInput.value = symbol;
    await loadValuation(false, false);
  }
}

void init();
