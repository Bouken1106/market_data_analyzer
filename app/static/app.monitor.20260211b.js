const tableBody = document.getElementById("price-table");
const tableHead = tableBody?.closest("table")?.querySelector("thead") || null;
const symbolSearchArea = document.getElementById("symbol-search-area");
const symbolSearchInput = document.getElementById("symbol-search");
const symbolDropdown = document.getElementById("symbol-dropdown");
const catalogMetaEl = document.getElementById("catalog-meta");
const selectionErrorEl = document.getElementById("selection-error");
const refreshCatalogBtn = document.getElementById("refresh-catalog");
const modeEl = document.getElementById("mode");
const wsStateEl = document.getElementById("ws-state");
const fallbackEl = document.getElementById("fallback-interval");
const creditsLeftEl = document.getElementById("credits-left");
const creditsUpdatedEl = document.getElementById("credits-updated");
const refreshCreditsBtn = document.getElementById("refresh-credits");
const clockJstEl = document.getElementById("clock-jst");
const clockEtEl = document.getElementById("clock-et");
const marketHoursEtEl = document.getElementById("market-hours-et");
const marketOpenStateEl = document.getElementById("market-open-state");
const pfInitialCashEl = document.getElementById("pf-initial-cash");
const pfCashEl = document.getElementById("pf-cash");
const pfMarketValueEl = document.getElementById("pf-market-value");
const pfEquityEl = document.getElementById("pf-equity");
const pfUnrealizedPnlEl = document.getElementById("pf-unrealized-pnl");
const pfReturnEl = document.getElementById("pf-return");
const pfTradeForm = document.getElementById("pf-trade-form");
const pfSymbolInput = document.getElementById("pf-symbol");
const pfSideInput = document.getElementById("pf-side");
const pfQuantityInput = document.getElementById("pf-quantity");
const pfPriceInput = document.getElementById("pf-price");
const pfSubmitBtn = document.getElementById("pf-submit");
const pfResetBtn = document.getElementById("pf-reset");
const pfMessageEl = document.getElementById("pf-message");
const pfPositionsBody = document.getElementById("pf-positions-body");
const pfTradesBody = document.getElementById("pf-trades-body");

const MAX_SYMBOLS = 8;
const MAX_DROPDOWN_ITEMS = 120;

const rowsBySymbol = new Map();
const latestRowsBySymbol = new Map();
const symbolInsightsBySymbol = new Map();
const sparklineFetchInFlight = new Set();
let openSymbolsSet = new Set();

let eventSource;
let symbolCatalog = [];
let selectedSymbols = [];
let syncInFlight = false;
let syncQueued = false;
let marketClockTimer = null;
let sortState = { key: "symbol", direction: "asc" };
let portfolioBaseState = null;

const JST_TIME_ZONE = "Asia/Tokyo";
const ET_TIME_ZONE = "America/New_York";
const US_MARKET_OPEN_MINUTE = (9 * 60) + 30;
const US_MARKET_CLOSE_MINUTE = 16 * 60;

const zoneFormatter = (timeZone) => new Intl.DateTimeFormat("en-US", {
  timeZone,
  hour12: false,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  weekday: "short",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

const jstFormatter = zoneFormatter(JST_TIME_ZONE);
const etFormatter = zoneFormatter(ET_TIME_ZONE);

function normalizeSymbol(raw) {
  return String(raw || "").trim().toUpperCase();
}

function uniqueSymbols(values) {
  const out = [];
  const seen = new Set();

  values.forEach((raw) => {
    const symbol = normalizeSymbol(raw);
    if (!symbol) return;
    if (!/^[A-Z0-9.\-]{1,15}$/.test(symbol)) return;
    if (seen.has(symbol)) return;
    seen.add(symbol);
    out.push(symbol);
  });

  return out;
}

function formatPrice(value) {
  if (value === null || value === undefined) return "-";
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleTimeString("ja-JP", { hour12: false });
}

function formatChangePercent(value) {
  if (!Number.isFinite(value)) return "-";
  return `${Math.abs(value).toFixed(2)}%`;
}

function formatMoney(value) {
  if (!Number.isFinite(value)) return "-";
  return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatSignedMoney(value) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  const prefix = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${prefix}${formatMoney(abs)}`;
}

function formatSignedPercent(value) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value).toFixed(2);
  if (value > 0) return `+${abs}%`;
  if (value < 0) return `-${abs}%`;
  return "0.00%";
}

function computeAnnualizedVol(prices) {
  if (!Array.isArray(prices) || prices.length < 3) return null;
  const logReturns = [];
  for (let i = 1; i < prices.length; i++) {
    if (prices[i - 1] > 0 && prices[i] > 0) {
      logReturns.push(Math.log(prices[i] / prices[i - 1]));
    }
  }
  if (logReturns.length < 2) return null;
  const mean = logReturns.reduce((s, v) => s + v, 0) / logReturns.length;
  const variance = logReturns.reduce((s, v) => s + (v - mean) ** 2, 0) / (logReturns.length - 1);
  return Math.sqrt(variance) * Math.sqrt(252) * 100;
}

function computeReturn30d(prices) {
  if (!Array.isArray(prices) || prices.length < 2) return null;
  const first = prices[0];
  const last = prices[prices.length - 1];
  if (!Number.isFinite(first) || !Number.isFinite(last) || first <= 0) return null;
  return ((last - first) / first) * 100;
}

function volLevel(vol) {
  if (!Number.isFinite(vol)) return "";
  if (vol < 20) return "vol-low";
  if (vol < 35) return "vol-mid";
  return "vol-high";
}

function readZoneClockParts(date, formatter) {
  const parts = formatter.formatToParts(date);
  const pick = (type) => parts.find((item) => item.type === type)?.value || "";
  return {
    year: Number(pick("year")),
    month: Number(pick("month")),
    day: Number(pick("day")),
    weekday: pick("weekday"),
    hour: Number(pick("hour")),
    minute: Number(pick("minute")),
    second: Number(pick("second")),
  };
}

function formatZoneClockText(parts) {
  const y = Number.isFinite(parts.year) ? String(parts.year).padStart(4, "0") : "----";
  const m = Number.isFinite(parts.month) ? String(parts.month).padStart(2, "0") : "--";
  const d = Number.isFinite(parts.day) ? String(parts.day).padStart(2, "0") : "--";
  const h = Number.isFinite(parts.hour) ? String(parts.hour).padStart(2, "0") : "--";
  const mm = Number.isFinite(parts.minute) ? String(parts.minute).padStart(2, "0") : "--";
  const s = Number.isFinite(parts.second) ? String(parts.second).padStart(2, "0") : "--";
  const weekday = String(parts.weekday || "").trim();
  return `${y}-${m}-${d} ${h}:${mm}:${s}${weekday ? ` (${weekday})` : ""}`;
}

function isUsRegularSessionOpen(etParts) {
  const weekday = String(etParts.weekday || "");
  const isWeekday = ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(weekday);
  if (!isWeekday) return false;
  if (!Number.isFinite(etParts.hour) || !Number.isFinite(etParts.minute)) return false;

  const minuteOfDay = (etParts.hour * 60) + etParts.minute;
  return minuteOfDay >= US_MARKET_OPEN_MINUTE && minuteOfDay < US_MARKET_CLOSE_MINUTE;
}

function renderMarketClock() {
  if (!clockJstEl || !clockEtEl || !marketHoursEtEl || !marketOpenStateEl) return;

  const now = new Date();
  const jstParts = readZoneClockParts(now, jstFormatter);
  const etParts = readZoneClockParts(now, etFormatter);

  clockJstEl.textContent = formatZoneClockText(jstParts);
  clockEtEl.textContent = formatZoneClockText(etParts);
  marketHoursEtEl.textContent = "09:30-16:00 ET (Mon-Fri)";

  const marketIsOpen = isUsRegularSessionOpen(etParts);
  marketOpenStateEl.textContent = marketIsOpen ? "Open now (regular session)" : "Closed now (regular session)";
  marketOpenStateEl.classList.toggle("open", marketIsOpen);
  marketOpenStateEl.classList.toggle("closed", !marketIsOpen);
}

function startMarketClock() {
  if (!clockJstEl || !clockEtEl || !marketHoursEtEl || !marketOpenStateEl) return;
  renderMarketClock();
  if (marketClockTimer) {
    window.clearInterval(marketClockTimer);
  }
  marketClockTimer = window.setInterval(renderMarketClock, 1000);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json().catch(() => ({}));
  return { response, result };
}

function setSelectionError(message) {
  selectionErrorEl.textContent = message || "";
}

function setCatalogMeta(message) {
  catalogMetaEl.textContent = message || "";
}

function setStatus(status) {
  modeEl.textContent = status?.mode ?? "-";
  const provider = String(status?.provider || "").toLowerCase();
  const wsAvailable = provider !== "fmp";
  wsStateEl.textContent = wsAvailable ? (status?.ws_connected ? "connected" : "disconnected") : "n/a";
  fallbackEl.textContent = `${status?.fallback_poll_interval_sec ?? "-"} sec`;
  const openSymbols = Array.isArray(status?.open_symbols)
    ? status.open_symbols.map((item) => normalizeSymbol(item)).filter((item) => item)
    : [];
  openSymbolsSet = new Set(openSymbols);
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
  if (refreshCreditsBtn) {
    refreshCreditsBtn.disabled = provider === "fmp";
    refreshCreditsBtn.title = provider !== "fmp"
      ? "Refresh via Twelve Data /api_usage"
      : "Unavailable for current provider";
  }
  refreshRowsForInsights(Array.from(rowsBySymbol.keys()));
}

function setPortfolioMessage(message, isError = false) {
  if (!pfMessageEl) return;
  pfMessageEl.textContent = message || "";
  pfMessageEl.classList.toggle("error", Boolean(isError));
}

function buildPortfolioView(baseState) {
  if (!baseState || typeof baseState !== "object") return null;
  const initialCash = Number(baseState.initial_cash);
  const cash = Number(baseState.cash);
  const rawPositions = Array.isArray(baseState.positions) ? baseState.positions : [];
  const positions = [];
  let marketValue = 0;
  let costBasis = 0;

  rawPositions.forEach((item) => {
    const symbol = normalizeSymbol(item?.symbol);
    const quantity = Number(item?.quantity);
    const avgCost = Number(item?.avg_cost);
    if (!symbol || !Number.isFinite(quantity) || quantity <= 0 || !Number.isFinite(avgCost) || avgCost <= 0) {
      return;
    }
    const latestPrice = Number(latestRowsBySymbol.get(symbol)?.price);
    const fallbackPrice = Number(item?.last_price);
    const lastPrice = Number.isFinite(latestPrice) && latestPrice > 0
      ? latestPrice
      : (Number.isFinite(fallbackPrice) && fallbackPrice > 0 ? fallbackPrice : null);
    const rowCostBasis = quantity * avgCost;
    const rowMarketValue = Number.isFinite(lastPrice) ? quantity * lastPrice : null;
    const rowPnl = Number.isFinite(rowMarketValue) ? rowMarketValue - rowCostBasis : null;
    const rowPnlPct = Number.isFinite(rowPnl) && rowCostBasis > 0 ? (rowPnl / rowCostBasis) * 100 : null;
    if (Number.isFinite(rowMarketValue)) {
      marketValue += rowMarketValue;
    }
    costBasis += rowCostBasis;
    positions.push({
      symbol,
      quantity,
      avg_cost: avgCost,
      cost_basis: rowCostBasis,
      last_price: lastPrice,
      market_value: rowMarketValue,
      unrealized_pnl: rowPnl,
      unrealized_pnl_pct: rowPnlPct,
      weight: null,
    });
  });

  if (marketValue > 0) {
    positions.forEach((item) => {
      if (Number.isFinite(item.market_value)) {
        item.weight = (item.market_value / marketValue) * 100;
      }
    });
  }

  const safeCash = Number.isFinite(cash) ? cash : 0;
  const safeInitialCash = Number.isFinite(initialCash) && initialCash > 0 ? initialCash : safeCash;
  const equity = safeCash + marketValue;
  const unrealizedPnl = marketValue - costBasis;
  const totalReturnPct = safeInitialCash > 0 ? ((equity - safeInitialCash) / safeInitialCash) * 100 : null;
  const rawRecentTrades = Array.isArray(baseState.recent_trades) ? baseState.recent_trades : [];
  const openDirectionBySymbol = new Map();
  positions.forEach((item) => {
    if (!item?.symbol || !Number.isFinite(item.quantity) || Math.abs(item.quantity) <= 0) return;
    openDirectionBySymbol.set(item.symbol, item.quantity > 0 ? "long" : "short");
  });
  const recentTrades = rawRecentTrades.filter((item) => {
    const symbol = normalizeSymbol(item?.symbol);
    if (!symbol) return false;
    const side = String(item?.side || "").toLowerCase();
    const openDirection = openDirectionBySymbol.get(symbol);
    if (!openDirection) return true;
    if (openDirection === "long" && side === "buy") return false;
    if (openDirection === "short" && side === "short") return false;
    return true;
  });

  return {
    initial_cash: safeInitialCash,
    cash: safeCash,
    market_value: marketValue,
    equity,
    unrealized_pnl: unrealizedPnl,
    total_return_pct: totalReturnPct,
    positions,
    recent_trades: recentTrades,
  };
}

function renderPortfolio() {
  if (!pfInitialCashEl || !portfolioBaseState) return;
  const view = buildPortfolioView(portfolioBaseState);
  if (!view) return;

  pfInitialCashEl.textContent = `$ ${formatMoney(view.initial_cash)}`;
  pfCashEl.textContent = `$ ${formatMoney(view.cash)}`;
  pfMarketValueEl.textContent = `$ ${formatMoney(view.market_value)}`;
  pfEquityEl.textContent = `$ ${formatMoney(view.equity)}`;
  pfUnrealizedPnlEl.textContent = formatSignedMoney(view.unrealized_pnl);
  pfUnrealizedPnlEl.classList.toggle("pf-positive", Number(view.unrealized_pnl) > 0);
  pfUnrealizedPnlEl.classList.toggle("pf-negative", Number(view.unrealized_pnl) < 0);
  pfReturnEl.textContent = formatSignedPercent(view.total_return_pct);
  pfReturnEl.classList.toggle("pf-positive", Number(view.total_return_pct) > 0);
  pfReturnEl.classList.toggle("pf-negative", Number(view.total_return_pct) < 0);

  if (pfPositionsBody) {
    pfPositionsBody.innerHTML = "";
    if (view.positions.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.className = "pf-empty";
      td.textContent = "No positions";
      tr.appendChild(td);
      pfPositionsBody.appendChild(tr);
    } else {
      view.positions.forEach((item) => {
        const tr = document.createElement("tr");
        const cells = [
          item.symbol,
          Number.isFinite(item.quantity) ? item.quantity.toFixed(4).replace(/\.?0+$/, "") : "-",
          Number.isFinite(item.avg_cost) ? `$ ${formatMoney(item.avg_cost)}` : "-",
          Number.isFinite(item.last_price) ? `$ ${formatMoney(item.last_price)}` : "-",
          Number.isFinite(item.market_value) ? `$ ${formatMoney(item.market_value)}` : "-",
          Number.isFinite(item.unrealized_pnl)
            ? `${formatSignedMoney(item.unrealized_pnl)} (${formatSignedPercent(item.unrealized_pnl_pct)})`
            : "-",
          Number.isFinite(item.weight) ? `${item.weight.toFixed(1)}%` : "-",
        ];
        cells.forEach((text, idx) => {
          const td = document.createElement("td");
          td.textContent = text;
          if (idx === 5) {
            td.classList.toggle("pf-positive", Number(item.unrealized_pnl) > 0);
            td.classList.toggle("pf-negative", Number(item.unrealized_pnl) < 0);
          }
          tr.appendChild(td);
        });
        pfPositionsBody.appendChild(tr);
      });
    }
  }

  if (pfTradesBody) {
    pfTradesBody.innerHTML = "";
    const trades = view.recent_trades.slice(0, 20);
    if (trades.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 6;
      td.className = "pf-empty";
      td.textContent = "No trades";
      tr.appendChild(td);
      pfTradesBody.appendChild(tr);
    } else {
      trades.forEach((item) => {
        const tr = document.createElement("tr");
        const realized = Number(item?.realized_pnl);
        const side = String(item?.side || "").toLowerCase();
        const cells = [
          formatTime(item?.timestamp),
          normalizeSymbol(item?.symbol),
          side || "-",
          Number.isFinite(Number(item?.quantity)) ? Number(item.quantity).toFixed(4).replace(/\.?0+$/, "") : "-",
          Number.isFinite(Number(item?.price)) ? `$ ${formatMoney(Number(item.price))}` : "-",
          Number.isFinite(realized) ? formatSignedMoney(realized) : "-",
        ];
        cells.forEach((text, idx) => {
          const td = document.createElement("td");
          td.textContent = text;
          if (idx === 2) {
            td.classList.add(side === "buy" ? "pf-side-buy" : "pf-side-sell");
          }
          if (idx === 5) {
            td.classList.toggle("pf-positive", realized > 0);
            td.classList.toggle("pf-negative", realized < 0);
          }
          tr.appendChild(td);
        });
        pfTradesBody.appendChild(tr);
      });
    }
  }
}

async function loadPortfolio() {
  try {
    const { response, result } = await fetchJson("/api/portfolio");
    if (!response.ok) {
      setPortfolioMessage(result.detail || "Failed to load portfolio.", true);
      return;
    }
    portfolioBaseState = result;
    renderPortfolio();
  } catch (_error) {
    setPortfolioMessage("Failed to load portfolio.", true);
  }
}

function goHistorical(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;
  window.location.href = `/historical/${encodeURIComponent(normalized)}`;
}

function renderSparklineSvg(values) {
  const points = (Array.isArray(values) ? values : [])
    .map((item) => Number(item))
    .filter((num) => Number.isFinite(num));

  if (points.length < 2) {
    return '<span class="sparkline-empty">-</span>';
  }

  const width = 100;
  const height = 16;
  const pad = 1;
  let min = Math.min(...points);
  let max = Math.max(...points);
  if (min === max) {
    min -= 1;
    max += 1;
  }

  const coords = points
    .map((value, index) => {
      const x = pad + ((index / Math.max(points.length - 1, 1)) * (width - (pad * 2)));
      const ratio = (value - min) / (max - min);
      const y = (height - pad) - (ratio * (height - (pad * 2)));
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return `<svg class="sparkline-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true"><polyline fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" points="${coords}"></polyline></svg>`;
}

function getRowCell(tr, name) {
  return tr.querySelector(`[data-col="${name}"]`);
}

function computeChangePct(symbol, currentPrice, insight = symbolInsightsBySymbol.get(symbol)) {
  const priceNum = Number(currentPrice);
  const latestClose = Number(insight?.latest_close);
  const previousClose = Number(insight?.previous_close);
  const marketIsOpen = openSymbolsSet.has(symbol);
  const referenceClose = marketIsOpen
    ? latestClose
    : (Number.isFinite(previousClose) && previousClose > 0 ? previousClose : latestClose);

  if (!Number.isFinite(priceNum) || !Number.isFinite(referenceClose) || referenceClose <= 0) {
    return null;
  }
  return ((priceNum - referenceClose) / referenceClose) * 100;
}

function getSortValue(symbol, key) {
  const latest = latestRowsBySymbol.get(symbol);
  const insight = symbolInsightsBySymbol.get(symbol);
  const trend = Array.isArray(insight?.trend_30d) ? insight.trend_30d : [];

  if (key === "symbol") {
    return symbol;
  }

  if (key === "price") {
    const price = Number(latest?.price);
    return Number.isFinite(price) ? price : null;
  }

  if (key === "change") {
    return computeChangePct(symbol, latest?.price, insight);
  }

  if (key === "range") {
    if (trend.length < 2) return null;
    const high = Math.max(...trend);
    return Number.isFinite(high) ? high : null;
  }

  if (key === "ret30d") {
    return computeReturn30d(trend);
  }

  if (key === "vol") {
    return computeAnnualizedVol(trend);
  }

  if (key === "updated") {
    const ts = Date.parse(latest?.timestamp || "");
    return Number.isFinite(ts) ? ts : null;
  }

  return null;
}

function compareSortValues(a, b, direction) {
  const aMissing = a === null || a === undefined;
  const bMissing = b === null || b === undefined;
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;

  if (typeof a === "string" || typeof b === "string") {
    const result = String(a).localeCompare(String(b), "en", { sensitivity: "base" });
    return direction === "asc" ? result : -result;
  }

  const diff = Number(a) - Number(b);
  if (diff === 0) return 0;
  return direction === "asc" ? diff : -diff;
}

function updateSortHeaderUI() {
  if (!tableHead) return;
  const sortHeaders = tableHead.querySelectorAll("th[data-sort-key]");
  sortHeaders.forEach((th) => {
    const key = th.dataset.sortKey;
    const isActive = key === sortState.key;
    th.classList.toggle("sorted", isActive);
    th.classList.toggle("sorted-asc", isActive && sortState.direction === "asc");
    th.classList.toggle("sorted-desc", isActive && sortState.direction === "desc");
    th.setAttribute(
      "aria-sort",
      isActive ? (sortState.direction === "asc" ? "ascending" : "descending") : "none",
    );
  });
}

function applySortToTable() {
  const symbols = Array.from(rowsBySymbol.keys());
  if (symbols.length <= 1) {
    updateSortHeaderUI();
    return;
  }

  const { key, direction } = sortState;
  symbols.sort((left, right) => {
    const leftValue = getSortValue(left, key);
    const rightValue = getSortValue(right, key);
    const primary = compareSortValues(leftValue, rightValue, direction);
    if (primary !== 0) return primary;
    return left.localeCompare(right, "en", { sensitivity: "base" });
  });

  const fragment = document.createDocumentFragment();
  symbols.forEach((symbol) => {
    const tr = rowsBySymbol.get(symbol);
    if (tr) fragment.appendChild(tr);
  });
  tableBody.appendChild(fragment);
  updateSortHeaderUI();
}

function applyInsightToRow(tr, symbol, priceValue) {
  const changeEl = getRowCell(tr, "change");
  const sparklineEl = getRowCell(tr, "sparkline");
  const rangeEl = getRowCell(tr, "range");
  const retEl = getRowCell(tr, "return30d");
  const volEl = getRowCell(tr, "vol");
  if (!changeEl || !sparklineEl) return;
  const insight = symbolInsightsBySymbol.get(symbol);

  changeEl.classList.remove("up", "down");

  const currentPrice = Number(priceValue);
  const pct = computeChangePct(symbol, currentPrice, insight);
  if (Number.isFinite(pct)) {
    if (pct > 0) {
      changeEl.classList.add("up");
      changeEl.textContent = `▲ ${formatChangePercent(pct)}`;
    } else if (pct < 0) {
      changeEl.classList.add("down");
      changeEl.textContent = `▼ ${formatChangePercent(pct)}`;
    } else {
      changeEl.textContent = formatChangePercent(0);
    }
  } else {
    changeEl.textContent = "-";
  }

  const trend = insight?.trend_30d ?? [];
  sparklineEl.innerHTML = renderSparklineSvg(trend);

  // 30D High / Low
  if (rangeEl) {
    if (trend.length >= 2) {
      const high = Math.max(...trend);
      const low = Math.min(...trend);
      rangeEl.innerHTML = `<span class="range-high">${formatPrice(high)}</span> <span class="range-sep">/</span> <span class="range-low">${formatPrice(low)}</span>`;
    } else {
      rangeEl.textContent = "-";
    }
  }

  // 30D Return
  if (retEl) {
    retEl.classList.remove("up", "down");
    const ret = computeReturn30d(trend);
    if (Number.isFinite(ret)) {
      if (ret > 0) {
        retEl.classList.add("up");
        retEl.textContent = `▲ ${Math.abs(ret).toFixed(2)}%`;
      } else if (ret < 0) {
        retEl.classList.add("down");
        retEl.textContent = `▼ ${Math.abs(ret).toFixed(2)}%`;
      } else {
        retEl.textContent = "0.00%";
      }
    } else {
      retEl.textContent = "-";
    }
  }

  // Volatility
  if (volEl) {
    volEl.classList.remove("vol-low", "vol-mid", "vol-high");
    const vol = computeAnnualizedVol(trend);
    if (Number.isFinite(vol)) {
      volEl.textContent = `${vol.toFixed(1)}%`;
      const level = volLevel(vol);
      if (level) volEl.classList.add(level);
    } else {
      volEl.textContent = "-";
    }
  }
}

function findCatalogName(symbol) {
  const entry = symbolCatalog.find((item) => item.symbol === symbol);
  return entry?.name || "";
}

function ensureRow(symbol) {
  if (rowsBySymbol.has(symbol)) {
    return rowsBySymbol.get(symbol);
  }

  const tr = document.createElement("tr");
  tr.dataset.symbol = symbol;

  const symTd = document.createElement("td");
  symTd.className = "sym";
  const symCell = document.createElement("div");
  symCell.className = "sym-cell";
  const openBtn = document.createElement("button");
  openBtn.type = "button";
  openBtn.className = "sym-open";
  openBtn.dataset.symbol = symbol;
  openBtn.title = "Open historical chart";
  openBtn.textContent = symbol;
  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "sym-remove";
  removeBtn.dataset.symbol = symbol;
  removeBtn.title = "Remove symbol";
  removeBtn.textContent = "x";
  symCell.appendChild(openBtn);
  symCell.appendChild(removeBtn);
  const nameDiv = document.createElement("div");
  nameDiv.className = "sym-name";
  nameDiv.textContent = findCatalogName(symbol);
  symTd.appendChild(symCell);
  symTd.appendChild(nameDiv);
  tr.appendChild(symTd);

  const priceTd = document.createElement("td");
  priceTd.className = "price";
  priceTd.dataset.col = "price";
  priceTd.textContent = "-";
  tr.appendChild(priceTd);

  const changeTd = document.createElement("td");
  changeTd.className = "change";
  changeTd.dataset.col = "change";
  changeTd.textContent = "-";
  tr.appendChild(changeTd);

  const rangeTd = document.createElement("td");
  rangeTd.className = "range-cell";
  rangeTd.dataset.col = "range";
  rangeTd.textContent = "-";
  tr.appendChild(rangeTd);

  const retTd = document.createElement("td");
  retTd.className = "return-cell";
  retTd.dataset.col = "return30d";
  retTd.textContent = "-";
  tr.appendChild(retTd);

  const volTd = document.createElement("td");
  volTd.className = "vol-cell";
  volTd.dataset.col = "vol";
  volTd.textContent = "-";
  tr.appendChild(volTd);

  const sparkTd = document.createElement("td");
  sparkTd.className = "sparkline-cell";
  const sparkWrap = document.createElement("div");
  sparkWrap.className = "sparkline";
  sparkWrap.dataset.col = "sparkline";
  const sparkEmpty = document.createElement("span");
  sparkEmpty.className = "sparkline-empty";
  sparkEmpty.textContent = "-";
  sparkWrap.appendChild(sparkEmpty);
  sparkTd.appendChild(sparkWrap);
  tr.appendChild(sparkTd);

  const updatedTd = document.createElement("td");
  updatedTd.className = "updated-cell";
  const timeDiv = document.createElement("div");
  timeDiv.className = "time";
  timeDiv.dataset.col = "time";
  timeDiv.textContent = "-";
  const sourceDiv = document.createElement("div");
  sourceDiv.className = "source-meta";
  sourceDiv.dataset.col = "source";
  sourceDiv.textContent = "source: -";
  updatedTd.appendChild(timeDiv);
  updatedTd.appendChild(sourceDiv);
  tr.appendChild(updatedTd);

  rowsBySymbol.set(symbol, tr);
  tableBody.appendChild(tr);
  return tr;
}

function refreshRowActionState() {
  const disableRemove = selectedSymbols.length <= 1;
  for (const [symbol, tr] of rowsBySymbol.entries()) {
    const openBtn = tr.querySelector(".sym-open");
    const removeBtn = tr.querySelector(".sym-remove");
    const nameDiv = tr.querySelector(".sym-name");
    if (openBtn) {
      openBtn.dataset.symbol = symbol;
      openBtn.textContent = symbol;
    }
    if (removeBtn) {
      removeBtn.dataset.symbol = symbol;
      removeBtn.disabled = disableRemove;
      removeBtn.title = disableRemove ? "At least one symbol is required" : "Remove symbol";
    }
    if (nameDiv && !nameDiv.textContent) {
      nameDiv.textContent = findCatalogName(symbol);
    }
  }
}

function renderRow(update, options = {}) {
  const symbol = normalizeSymbol(update?.symbol);
  if (!symbol) return;

  const normalizedUpdate = {
    symbol,
    price: update?.price ?? null,
    timestamp: update?.timestamp ?? null,
    source: update?.source ?? null,
  };

  latestRowsBySymbol.set(symbol, normalizedUpdate);

  const tr = ensureRow(symbol);
  const priceEl = getRowCell(tr, "price");
  const timeEl = getRowCell(tr, "time");
  const sourceEl = getRowCell(tr, "source");
  if (!priceEl || !timeEl || !sourceEl) return;
  priceEl.textContent = formatPrice(normalizedUpdate.price);
  timeEl.textContent = formatTime(normalizedUpdate.timestamp);
  sourceEl.textContent = `source: ${normalizedUpdate.source || "-"}`;
  applyInsightToRow(tr, symbol, normalizedUpdate.price);

  if (options.flash !== false) {
    tr.classList.remove("flash");
    void tr.offsetWidth;
    tr.classList.add("flash");
  }

  refreshRowActionState();
  applySortToTable();
  renderPortfolio();
}

function refreshRowsForInsights(symbols) {
  const targets = uniqueSymbols(symbols);
  targets.forEach((symbol) => {
    const tr = rowsBySymbol.get(symbol);
    if (!tr) return;
    const latest = latestRowsBySymbol.get(symbol);
    if (latest) {
      renderRow(latest, { flash: false });
      return;
    }
    applyInsightToRow(tr, symbol, null);
  });
}

async function loadSymbolInsights(symbols, refresh = false) {
  const targets = uniqueSymbols(symbols)
    .slice(0, MAX_SYMBOLS)
    .filter((symbol) => refresh || !symbolInsightsBySymbol.has(symbol))
    .filter((symbol) => !sparklineFetchInFlight.has(symbol));

  if (targets.length === 0) {
    refreshRowsForInsights(symbols);
    return;
  }

  targets.forEach((symbol) => sparklineFetchInFlight.add(symbol));

  try {
    const params = new URLSearchParams({ symbols: targets.join(",") });
    if (refresh) {
      params.set("refresh", "true");
    }

    const { response, result } = await fetchJson(`/api/sparkline?${params.toString()}`);
    if (!response.ok || !Array.isArray(result.items)) {
      return;
    }

    result.items.forEach((item) => {
      const symbol = normalizeSymbol(item?.symbol);
      if (!symbol) return;

      const latestClose = Number(item?.latest_close);
      const previousClose = Number(item?.previous_close);
      const trend = (Array.isArray(item?.trend_30d) ? item.trend_30d : [])
        .map((point) => Number(point))
        .filter((num) => Number.isFinite(num));

      symbolInsightsBySymbol.set(symbol, {
        latest_close: Number.isFinite(latestClose) ? latestClose : null,
        previous_close: Number.isFinite(previousClose) ? previousClose : null,
        trend_30d: trend,
      });
    });
  } catch (_error) {
    // Keep market rows functional even if sparkline fetch fails.
  } finally {
    targets.forEach((symbol) => sparklineFetchInFlight.delete(symbol));
  }

  refreshRowsForInsights(symbols);
}

function resetRows(symbols) {
  rowsBySymbol.clear();
  latestRowsBySymbol.clear();
  tableBody.innerHTML = "";
  symbols.forEach((symbol) => {
    const tr = ensureRow(symbol);
    applyInsightToRow(tr, symbol, null);
  });
  refreshRowActionState();
  applySortToTable();
  renderPortfolio();
}

function renderRows(rows, fallbackSymbols = []) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const symbols = safeRows.length > 0 ? safeRows.map((row) => row.symbol) : fallbackSymbols;
  resetRows(symbols);
  safeRows.forEach((row) => renderRow(row, { flash: false }));
}

function applySymbolsFromServer(symbols) {
  selectedSymbols = uniqueSymbols(symbols).slice(0, MAX_SYMBOLS);
  refreshRowActionState();
  if (document.activeElement === symbolSearchInput) {
    renderDropdown();
  }
}

function showDropdown() {
  symbolDropdown.classList.remove("hidden");
}

function hideDropdown() {
  symbolDropdown.classList.add("hidden");
}

function pickCandidates(query) {
  const needle = query.trim().toUpperCase();
  const candidates = [];

  if (symbolCatalog.length === 0) {
    return candidates;
  }

  for (const item of symbolCatalog) {
    if (selectedSymbols.includes(item.symbol)) {
      continue;
    }
    if (!needle) {
      candidates.push(item);
    } else if (item.symbol.startsWith(needle)) {
      candidates.push(item);
    }
    if (candidates.length >= MAX_DROPDOWN_ITEMS) {
      break;
    }
  }

  return candidates;
}

function renderDropdown() {
  symbolDropdown.innerHTML = "";

  const candidates = pickCandidates(symbolSearchInput.value);
  if (symbolCatalog.length === 0) {
    const row = document.createElement("div");
    row.className = "dropdown-empty";
    row.textContent = "Symbol list is not loaded yet.";
    symbolDropdown.appendChild(row);
    showDropdown();
    return;
  }

  if (candidates.length === 0) {
    const row = document.createElement("div");
    row.className = "dropdown-empty";
    row.textContent = "No matching symbols";
    symbolDropdown.appendChild(row);
    showDropdown();
    return;
  }

  candidates.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dropdown-item";
    btn.dataset.symbol = item.symbol;
    btn.textContent = `${item.symbol} | ${item.name} (${item.exchange})`;
    symbolDropdown.appendChild(btn);
  });

  showDropdown();
}

async function updateSymbolsOnServer() {
  if (selectedSymbols.length === 0) {
    setSelectionError("At least one symbol is required.");
    return;
  }

  if (syncInFlight) {
    syncQueued = true;
    return;
  }

  syncInFlight = true;
  try {
    do {
      syncQueued = false;
      const payloadSymbols = [...selectedSymbols];
      const { response, result } = await fetchJson("/api/symbols", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ symbols: payloadSymbols.join(",") }),
      });
      if (!response.ok) {
        setSelectionError(result.detail || "Failed to update symbols");
        break;
      }

      setSelectionError("");
      setStatus(result.status || {});
      const serverSymbols = Array.isArray(result.symbols) ? result.symbols : payloadSymbols;
      renderRows(result.rows, serverSymbols);
      applySymbolsFromServer(serverSymbols);
      await loadSymbolInsights(serverSymbols, false);
    } while (syncQueued);
  } finally {
    syncInFlight = false;
  }
}

async function addSymbol(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;

  if (selectedSymbols.includes(normalized)) {
    goHistorical(normalized);
    return;
  }

  if (selectedSymbols.length >= MAX_SYMBOLS) {
    setSelectionError(`You can monitor up to ${MAX_SYMBOLS} symbols.`);
    return;
  }

  selectedSymbols = [...selectedSymbols, normalized];
  setSelectionError("");
  await updateSymbolsOnServer();
}

async function removeSymbol(symbol) {
  if (!selectedSymbols.includes(symbol)) return;
  if (selectedSymbols.length <= 1) {
    setSelectionError("At least one symbol is required.");
    return;
  }

  selectedSymbols = selectedSymbols.filter((item) => item !== symbol);
  setSelectionError("");
  await updateSymbolsOnServer();
}

async function loadSymbolCatalog(refresh = false) {
  refreshCatalogBtn.disabled = true;
  setCatalogMeta(refresh ? "Refreshing symbol catalog..." : "Loading symbol catalog...");

  try {
    const { response, result } = await fetchJson(
      refresh ? "/api/symbol-catalog?refresh=true" : "/api/symbol-catalog",
    );

    if (!response.ok) {
      setCatalogMeta(result.detail || "Failed to load symbol catalog");
      return;
    }

    const rawSymbols = Array.isArray(result.symbols) ? result.symbols : [];
    symbolCatalog = rawSymbols.map((item) => {
      const symbol = normalizeSymbol(item.symbol);
      const name = String(item.name || "").trim();
      const exchange = String(item.exchange || "").trim();
      return {
        symbol,
        name,
        exchange,
      };
    });

    const updatedText = result.updated_at ? `updated ${formatTime(result.updated_at)}` : "updated -";
    setCatalogMeta(`${symbolCatalog.length.toLocaleString()} symbols loaded (${result.source || "unknown"}, ${updatedText})`);

    if (document.activeElement === symbolSearchInput) {
      renderDropdown();
    }
  } finally {
    refreshCatalogBtn.disabled = false;
  }
}

function handleEvent(event) {
  const payload = JSON.parse(event.data);

  if (payload.type === "snapshot") {
    const rows = payload.data?.rows ?? [];
    const symbols = rows.map((row) => row.symbol);
    renderRows(rows, symbols);
    setStatus(payload.data?.status ?? {});
    applySymbolsFromServer(symbols);
    void loadSymbolInsights(symbols, false);
    return;
  }

  if (payload.type === "status") {
    setStatus(payload.data);
    return;
  }

  if (payload.type === "symbols") {
    const symbols = payload.data?.symbols ?? [];
    renderRows(payload.data?.rows ?? [], symbols);
    applySymbolsFromServer(symbols);
    void loadSymbolInsights(symbols, false);
    return;
  }

  if (payload.type === "price") {
    renderRow(payload.data);
  }
}

function connectEventStream() {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource("/api/stream");
  eventSource.onmessage = handleEvent;
  eventSource.onerror = () => {
    wsStateEl.textContent = "reconnecting";
  };
}

if (tableHead) {
  tableHead.addEventListener("click", (event) => {
    const header = event.target.closest("th[data-sort-key]");
    if (!header) return;
    const key = header.dataset.sortKey;
    if (!key) return;

    if (sortState.key === key) {
      sortState = {
        key,
        direction: sortState.direction === "asc" ? "desc" : "asc",
      };
    } else {
      sortState = { key, direction: key === "updated" ? "desc" : "asc" };
    }
    applySortToTable();
  });
}

symbolSearchInput.addEventListener("focus", () => {
  renderDropdown();
});

symbolSearchInput.addEventListener("input", () => {
  renderDropdown();
});

symbolSearchInput.addEventListener("keydown", async (event) => {
  if (event.key === "Escape") {
    hideDropdown();
    return;
  }

  if (event.key === "Enter") {
    event.preventDefault();
    const firstCandidate = symbolDropdown.querySelector(".dropdown-item");
    if (firstCandidate) {
      await addSymbol(firstCandidate.dataset.symbol);
      symbolSearchInput.value = "";
      renderDropdown();
    }
  }
});

tableBody.addEventListener("click", async (event) => {
  const openButton = event.target.closest(".sym-open");
  if (openButton) {
    goHistorical(openButton.dataset.symbol);
    return;
  }

  const removeButton = event.target.closest(".sym-remove");
  if (removeButton) {
    await removeSymbol(removeButton.dataset.symbol);
    renderDropdown();
  }
});

symbolDropdown.addEventListener("mousedown", (event) => {
  event.preventDefault();
});

symbolDropdown.addEventListener("click", async (event) => {
  const button = event.target.closest(".dropdown-item");
  if (!button) return;
  await addSymbol(button.dataset.symbol);
  symbolSearchInput.value = "";
  renderDropdown();
  symbolSearchInput.focus();
});

document.addEventListener("click", (event) => {
  if (!symbolSearchArea.contains(event.target)) {
    hideDropdown();
  }
});

refreshCatalogBtn.addEventListener("click", async () => {
  await loadSymbolCatalog(true);
  symbolSearchInput.focus();
  renderDropdown();
});

refreshCreditsBtn.addEventListener("click", async () => {
  refreshCreditsBtn.disabled = true;
  try {
    const { response, result } = await fetchJson("/api/credits?refresh=true");
    if (!response.ok || !result.status) {
      alert(result.detail || result.note || "Failed to refresh credits");
      return;
    }
    setStatus(result.status);
  } finally {
    refreshCreditsBtn.disabled = false;
  }
});

if (pfTradeForm) {
  pfTradeForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const symbol = normalizeSymbol(pfSymbolInput?.value);
    const side = String(pfSideInput?.value || "buy").toLowerCase();
    const quantity = Number(pfQuantityInput?.value);
    const rawPrice = String(pfPriceInput?.value || "").trim();
    const price = rawPrice ? Number(rawPrice) : null;

    if (!symbol) {
      setPortfolioMessage("Symbol is required.", true);
      return;
    }
    if (!Number.isFinite(quantity) || quantity <= 0) {
      setPortfolioMessage("Quantity must be greater than 0.", true);
      return;
    }
    if (rawPrice && (!Number.isFinite(price) || price <= 0)) {
      setPortfolioMessage("Price must be greater than 0.", true);
      return;
    }

    setPortfolioMessage("");
    if (pfSubmitBtn) pfSubmitBtn.disabled = true;
    try {
      const { response, result } = await fetchJson("/api/portfolio/trades", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          side,
          quantity,
          price: rawPrice ? price : null,
        }),
      });

      if (!response.ok) {
        setPortfolioMessage(result.detail || "Failed to submit trade.", true);
        return;
      }

      portfolioBaseState = result;
      renderPortfolio();
      const executionSource = result.trade?.execution_source || "market";
      setPortfolioMessage(`Trade accepted: ${side.toUpperCase()} ${symbol} (${executionSource} price).`);
      if (pfPriceInput) pfPriceInput.value = "";
      if (pfQuantityInput) pfQuantityInput.value = "";

      if (!selectedSymbols.includes(symbol) && selectedSymbols.length < MAX_SYMBOLS) {
        selectedSymbols = [...selectedSymbols, symbol];
        await updateSymbolsOnServer();
      }
    } catch (_error) {
      setPortfolioMessage("Failed to submit trade.", true);
    } finally {
      if (pfSubmitBtn) pfSubmitBtn.disabled = false;
    }
  });
}

if (pfResetBtn) {
  pfResetBtn.addEventListener("click", async () => {
    const defaultValue = Number(portfolioBaseState?.initial_cash);
    const promptValue = Number.isFinite(defaultValue) && defaultValue > 0 ? String(defaultValue) : "1000000";
    const input = window.prompt("Reset initial cash (blank keeps default).", promptValue);
    if (input === null) return;
    const normalized = String(input).trim();
    const payload = {};
    if (normalized) {
      const parsed = Number(normalized);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        setPortfolioMessage("Initial cash must be greater than 0.", true);
        return;
      }
      payload.initial_cash = parsed;
    }

    setPortfolioMessage("");
    pfResetBtn.disabled = true;
    try {
      const { response, result } = await fetchJson("/api/portfolio/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        setPortfolioMessage(result.detail || "Failed to reset portfolio.", true);
        return;
      }
      portfolioBaseState = result;
      renderPortfolio();
      setPortfolioMessage("Portfolio reset completed.");
    } catch (_error) {
      setPortfolioMessage("Failed to reset portfolio.", true);
    } finally {
      pfResetBtn.disabled = false;
    }
  });
}

updateSortHeaderUI();

loadSymbolCatalog().catch(() => {
  setCatalogMeta("Failed to load symbol catalog");
});
connectEventStream();
startMarketClock();
loadPortfolio();
