const symbolSearchArea = document.getElementById("symbol-search-area");
const symbolSearchInput = document.getElementById("symbol-search");
const symbolDropdown = document.getElementById("symbol-dropdown");
const catalogMetaEl = document.getElementById("catalog-meta");
const selectionErrorEl = document.getElementById("selection-error");
const refreshCatalogBtn = document.getElementById("refresh-catalog");
const refreshCreditsBtn = document.getElementById("refresh-credits");
const modeEl = document.getElementById("mode");
const wsStateEl = document.getElementById("ws-state");
const fallbackEl = document.getElementById("fallback-interval");
const creditsLeftEl = document.getElementById("credits-left");
const creditsUpdatedEl = document.getElementById("credits-updated");
const clockJstEl = document.getElementById("clock-jst");
const clockEtEl = document.getElementById("clock-et");
const marketHoursEtEl = document.getElementById("market-hours-et");
const marketOpenStateEl = document.getElementById("market-open-state");

const watchlistEl = document.getElementById("watchlist");
const watchlistHeadEl = document.getElementById("watchlist-head");
const watchLlmPanelEl = document.getElementById("watch-llm-panel");
const watchLlmTextEl = document.getElementById("watch-llm-text");
const watchLlmMetaEl = document.getElementById("watch-llm-meta");
const watchLlmRefreshBtn = document.getElementById("watch-llm-refresh");
const symbolTabsEl = document.getElementById("symbol-tabs");
const tabEmptyEl = document.getElementById("tab-empty");
const symbolPanelEl = document.getElementById("symbol-panel");
const centerPaneEl = document.querySelector(".center-pane");
const panelSymbolEl = document.getElementById("panel-symbol");
const panelUpdatedEl = document.getElementById("panel-updated");
const panelRefreshBtn = document.getElementById("panel-refresh");
const chartWrapEl = document.getElementById("chart-wrap");
const kpiPriceEl = document.getElementById("kpi-price");
const kpiChangeEl = document.getElementById("kpi-change");
const kpiRangeEl = document.getElementById("kpi-range");
const kpiVolumeEl = document.getElementById("kpi-volume");
const kpiMaEl = document.getElementById("kpi-ma");
const kpiAtrEl = document.getElementById("kpi-atr");
const pfCashEl = document.getElementById("pf-cash");
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

const pageDataset = document.body?.dataset || {};
const parseDatasetBool = (rawValue, fallback) => {
  if (rawValue === undefined) return fallback;
  return String(rawValue).trim().toLowerCase() !== "false";
};
const parseDatasetInt = (rawValue, fallback) => {
  const parsed = Number.parseInt(String(rawValue ?? ""), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};
const normalizeCacheNamespace = (rawValue) => String(rawValue || "").trim().toLowerCase();
const buildCacheKey = (namespace, suffix) => {
  const normalized = normalizeCacheNamespace(namespace);
  return normalized ? `mda.${normalized}.${suffix}` : `mda.${suffix}`;
};

const PAGE_CONFIG = {
  marketPreset: String(pageDataset.marketPreset || "us").trim().toLowerCase(),
  streamEnabled: parseDatasetBool(pageDataset.streamEnabled, true),
  serverSymbolSync: parseDatasetBool(pageDataset.serverSymbolSync, parseDatasetBool(pageDataset.streamEnabled, true)),
  watchlistCommentaryEnabled: parseDatasetBool(pageDataset.watchlistCommentary, true),
  catalogCountry: String(pageDataset.catalogCountry || "").trim(),
  currencySymbol: String(pageDataset.currencySymbol || "$").trim() || "$",
  currencyDigits: Math.max(0, parseDatasetInt(pageDataset.currencyDigits, 2)),
  priceMinDigits: Math.max(0, parseDatasetInt(pageDataset.priceMinDigits, 2)),
  priceMaxDigits: Math.max(0, parseDatasetInt(pageDataset.priceMaxDigits, 4)),
  primaryClockLabel: String(pageDataset.primaryClockLabel || "JST").trim() || "JST",
  primaryTimeZone: String(pageDataset.primaryTimeZone || "Asia/Tokyo").trim() || "Asia/Tokyo",
  secondaryClockLabel: String(pageDataset.secondaryClockLabel || "ET").trim() || "ET",
  secondaryTimeZone: String(pageDataset.secondaryTimeZone || "America/New_York").trim() || "America/New_York",
  sessionTimeZone: String(pageDataset.sessionTimeZone || pageDataset.secondaryTimeZone || "America/New_York").trim() || "America/New_York",
  marketOpenMinute: parseDatasetInt(pageDataset.marketOpenMinute, (9 * 60) + 30),
  marketCloseMinute: parseDatasetInt(pageDataset.marketCloseMinute, 16 * 60),
  marketHoursLabel: String(pageDataset.marketHoursLabel || "09:30-16:00 ET").trim() || "09:30-16:00 ET",
  cacheNamespace: String(pageDataset.cacheNamespace || "").trim(),
  overviewIncludeIntraday: parseDatasetBool(pageDataset.overviewIncludeIntraday, true),
  overviewIncludeMarket: parseDatasetBool(pageDataset.overviewIncludeMarket, true),
  overviewIncludeQqq: parseDatasetBool(pageDataset.overviewIncludeQqq, true),
  staticMode: String(pageDataset.staticMode || "daily-manual").trim() || "daily-manual",
  staticProvider: String(pageDataset.staticProvider || "manual").trim() || "manual",
  staticFallbackLabel: String(pageDataset.staticFallback || "manual").trim() || "manual",
};

const shouldUseEodOnlySparkline = () => PAGE_CONFIG.marketPreset === "us";
const applySparklineModeParams = (params) => {
  if (shouldUseEodOnlySparkline()) {
    params.set("eod_cache_only", "true");
  }
  return params;
};
const isConfiguredMarketOpenNow = () => {
  if (PAGE_CONFIG.marketPreset !== "us") return false;
  return isConfiguredRegularSessionOpen(readZoneClockParts(new Date(), sessionFormatter));
};
const resolveSparklinePriceSource = (item, hasCurrentPrice) => {
  if (String(item?.price_mode || "").trim() === "eod_close") {
    return "eod_close";
  }
  return hasCurrentPrice ? "sparkline_quote" : "sparkline_close";
};

const MAX_SYMBOLS = 8;
const MAX_DROPDOWN_ITEMS = 120;
const PRIMARY_TIME_ZONE = PAGE_CONFIG.primaryTimeZone;
const SECONDARY_TIME_ZONE = PAGE_CONFIG.secondaryTimeZone;
const SESSION_TIME_ZONE = PAGE_CONFIG.sessionTimeZone;
const MARKET_OPEN_MINUTE = PAGE_CONFIG.marketOpenMinute;
const MARKET_CLOSE_MINUTE = PAGE_CONFIG.marketCloseMinute;
const CHART_WIDTH = 900;
const CHART_HEIGHT = 320;
const CHART_PAD_X = 64;
const CHART_PAD_TOP = 16;
const CHART_PAD_BOTTOM = 26;
const CHART_ZOOM_IN_FACTOR = 0.8;
const CHART_ZOOM_OUT_FACTOR = 1.25;
const CHART_MIN_VISIBLE_POINTS = 2;
const CHART_Y_PADDING_RATIO = 0.08;
const CHART_VOLUME_BAND_RATIO = 0.28;
const HISTORICAL_LOAD_YEARS = 100;
const CHART_RANGE_PRESETS = ["1w", "1m", "1y", "5y", "10y", "max"];
const WATCHLIST_MODEL_UNKNOWN = "-";

const watchItemsBySymbol = new Map();
const latestRowsBySymbol = new Map();
const symbolInsightsBySymbol = new Map();
const symbolMetaBySymbol = new Map();
const sparklineFetchInFlight = new Set();
const tabStateBySymbol = new Map();
const chartViewportBySymbol = new Map();
const chartRangePresetBySymbol = new Map();
let openSymbolsSet = new Set();
let selectedSymbols = [];
let watchSortState = { key: "symbol", direction: "asc" };
let openTabs = [];
let activeTabSymbol = "";
let draggingTabSymbol = "";
let draggingTabPlacement = null;
let symbolCatalog = [];
let eventSource;
let syncInFlight = false;
let syncQueued = false;
let marketClockTimer = null;
let watchLlmInFlight = false;
let watchLlmQueuedRefresh = false;
let watchLlmLastSymbolsKey = "";
let watchLlmModelName = WATCHLIST_MODEL_UNKNOWN;
let ignoreFirstEmptySymbolsEvent = false;
let portfolioBaseState = null;
let chartPanState = {
  active: false,
  symbol: "",
  points: [],
  lastClientX: 0,
};
const SYMBOL_INSIGHTS_CACHE_KEY = buildCacheKey(PAGE_CONFIG.cacheNamespace, "symbol_insights.v1");
const WATCHLIST_SYMBOLS_CACHE_KEY = buildCacheKey(PAGE_CONFIG.cacheNamespace, "watchlist_symbols.v1");
const WATCHLIST_ROWS_CACHE_KEY = buildCacheKey(PAGE_CONFIG.cacheNamespace, "watchlist_rows.v1");
const WATCHLIST_NAMESPACE = PAGE_CONFIG.marketPreset === "jp" ? "jp" : "us";

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

const primaryFormatter = zoneFormatter(PRIMARY_TIME_ZONE);
const secondaryFormatter = zoneFormatter(SECONDARY_TIME_ZONE);
const sessionFormatter = zoneFormatter(SESSION_TIME_ZONE);

function normalizeSymbol(raw) {
  const normalized = String(raw || "").trim().toUpperCase();
  if (!normalized) return "";
  if (PAGE_CONFIG.marketPreset === "jp" && /^\d{4,5}$/.test(normalized)) {
    return `${normalized}.T`;
  }
  return normalized;
}

function normalizeSearchText(raw) {
  return String(raw || "").normalize("NFKC").trim().replace(/\s+/g, " ").toUpperCase();
}

function uniqueSymbols(values) {
  const out = [];
  const seen = new Set();
  (Array.isArray(values) ? values : []).forEach((raw) => {
    const symbol = normalizeSymbol(raw);
    if (!symbol) return;
    if (!/^[A-Z0-9.\-]{1,15}$/.test(symbol)) return;
    if (seen.has(symbol)) return;
    seen.add(symbol);
    out.push(symbol);
  });
  return out;
}

function upsertLatestRow(symbol, row, preserveOnInvalid = true) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;
  const existing = latestRowsBySymbol.get(normalized);
  const nextPriceNum = Number(row?.price);
  const nextHasValidPrice = Number.isFinite(nextPriceNum) && nextPriceNum > 0;
  const existingPriceNum = Number(existing?.price);
  const existingHasValidPrice = Number.isFinite(existingPriceNum) && existingPriceNum > 0;

  if (!nextHasValidPrice && preserveOnInvalid && existingHasValidPrice) {
    return;
  }

  latestRowsBySymbol.set(normalized, {
    symbol: normalized,
    price: nextHasValidPrice ? nextPriceNum : null,
    timestamp: row?.timestamp ?? existing?.timestamp ?? null,
    source: row?.source ?? existing?.source ?? null,
  });
}

function formatPrice(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  const maximumFractionDigits = Math.max(PAGE_CONFIG.priceMinDigits, PAGE_CONFIG.priceMaxDigits);
  return num.toLocaleString("en-US", {
    minimumFractionDigits: PAGE_CONFIG.priceMinDigits,
    maximumFractionDigits,
  });
}

function formatMoney(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toLocaleString("en-US", {
    minimumFractionDigits: PAGE_CONFIG.currencyDigits,
    maximumFractionDigits: PAGE_CONFIG.currencyDigits,
  });
}

function formatCurrency(value) {
  const formatted = formatMoney(value);
  if (formatted === "-") return "-";
  return `${PAGE_CONFIG.currencySymbol}${formatted}`;
}

function formatSignedCurrency(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  const sign = num > 0 ? "+" : (num < 0 ? "-" : "");
  return `${sign}${PAGE_CONFIG.currencySymbol}${formatMoney(Math.abs(num))}`;
}

function formatCompact(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(num);
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleTimeString("ja-JP", { hour12: false });
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function upsertSymbolMeta(symbol, payload = {}) {
  const normalizedSymbol = normalizeSymbol(symbol);
  if (!normalizedSymbol) return;
  const existing = symbolMetaBySymbol.get(normalizedSymbol) || {};
  const nextName = String(payload?.name || "").trim();
  const nextExchange = String(payload?.exchange || "").trim();
  symbolMetaBySymbol.set(normalizedSymbol, {
    symbol: normalizedSymbol,
    name: nextName || existing.name || "",
    exchange: nextExchange || existing.exchange || "",
  });
}

function buildSymbolDisplayLabel(symbol) {
  const normalizedSymbol = normalizeSymbol(symbol);
  if (!normalizedSymbol) return "-";
  const name = findCatalogName(normalizedSymbol);
  return name ? `${normalizedSymbol} | ${name}` : normalizedSymbol;
}

function buildSymbolOptionLabel(item) {
  const symbol = normalizeSymbol(item?.symbol);
  const name = String(item?.name || "").trim();
  const exchange = String(item?.exchange || "").trim();
  if (!symbol) return "-";
  if (name && exchange) return `${symbol} | ${name} (${exchange})`;
  if (name) return `${symbol} | ${name}`;
  if (exchange) return `${symbol} (${exchange})`;
  return symbol;
}

function formatSigned(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  if (num > 0) return `+${num.toFixed(digits)}`;
  return num.toFixed(digits);
}

function formatSignedPercent(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${formatSigned(num, 2)}%`;
}

function percentOf(numerator, denominator) {
  const numeratorNum = Number(numerator);
  const denominatorNum = Number(denominator);
  if (!Number.isFinite(numeratorNum) || !Number.isFinite(denominatorNum) || denominatorNum <= 0) {
    return null;
  }
  return (numeratorNum / denominatorNum) * 100;
}

function percentChange(current, previous) {
  const currentNum = Number(current);
  const previousNum = Number(previous);
  if (!Number.isFinite(currentNum) || !Number.isFinite(previousNum) || previousNum <= 0) {
    return null;
  }
  return percentOf(currentNum - previousNum, previousNum);
}

function formatChartDateLabel(value) {
  const raw = String(value || "").trim();
  if (!raw) return "-";
  const dt = new Date(raw);
  if (!Number.isNaN(dt.getTime())) {
    return dt.toLocaleDateString("ja-JP", { year: "2-digit", month: "2-digit", day: "2-digit" });
  }
  return raw.includes(" ") ? raw.split(" ")[0] : raw;
}

function parseChartPointDate(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;
  const datePart = raw.includes(" ") ? raw.split(" ")[0] : raw;
  const dt = new Date(`${datePart}T00:00:00`);
  if (Number.isNaN(dt.getTime())) return null;
  return dt;
}

function findStartIndexByDate(points, targetDate) {
  const rows = Array.isArray(points) ? points : [];
  for (let idx = 0; idx < rows.length; idx += 1) {
    const dt = parseChartPointDate(rows[idx]?.t);
    if (!dt) continue;
    if (dt >= targetDate) return idx;
  }
  return 0;
}

function applyChartRangePreset(symbol, points, preset) {
  const rows = Array.isArray(points) ? points : [];
  if (!symbol || rows.length < 2) return false;
  if (!CHART_RANGE_PRESETS.includes(preset)) return false;

  const lastIndex = rows.length - 1;
  if (preset === "max") {
    setChartViewport(symbol, 0, lastIndex, rows.length);
    chartRangePresetBySymbol.set(symbol, "max");
    return true;
  }

  const lastDate = parseChartPointDate(rows[lastIndex]?.t);
  if (!lastDate) {
    setChartViewport(symbol, 0, lastIndex, rows.length);
    chartRangePresetBySymbol.set(symbol, "max");
    return true;
  }

  const startDate = new Date(lastDate);
  if (preset === "1w") {
    startDate.setDate(startDate.getDate() - 7);
  } else if (preset === "1m") {
    startDate.setMonth(startDate.getMonth() - 1);
  } else if (preset === "1y") {
    startDate.setFullYear(startDate.getFullYear() - 1);
  } else if (preset === "5y") {
    startDate.setFullYear(startDate.getFullYear() - 5);
  } else if (preset === "10y") {
    startDate.setFullYear(startDate.getFullYear() - 10);
  } else {
    return false;
  }

  const startIndex = findStartIndexByDate(rows, startDate);
  setChartViewport(symbol, startIndex, lastIndex, rows.length);
  chartRangePresetBySymbol.set(symbol, preset);
  return true;
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
  return `${y}-${m}-${d} ${h}:${mm}:${s}`;
}

function isConfiguredRegularSessionOpen(sessionParts) {
  const weekday = String(sessionParts.weekday || "");
  const isWeekday = ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(weekday);
  if (!isWeekday || !Number.isFinite(sessionParts.hour) || !Number.isFinite(sessionParts.minute)) return false;
  const minuteOfDay = (sessionParts.hour * 60) + sessionParts.minute;
  if (MARKET_OPEN_MINUTE <= MARKET_CLOSE_MINUTE) {
    return minuteOfDay >= MARKET_OPEN_MINUTE && minuteOfDay < MARKET_CLOSE_MINUTE;
  }
  return minuteOfDay >= MARKET_OPEN_MINUTE || minuteOfDay < MARKET_CLOSE_MINUTE;
}

function renderMarketClock() {
  if (!clockJstEl || !clockEtEl || !marketOpenStateEl || !marketHoursEtEl) return;
  const now = new Date();
  const primary = readZoneClockParts(now, primaryFormatter);
  const secondary = readZoneClockParts(now, secondaryFormatter);
  const session = readZoneClockParts(now, sessionFormatter);
  clockJstEl.textContent = `${PAGE_CONFIG.primaryClockLabel} ${formatZoneClockText(primary)}`;
  clockEtEl.textContent = `${PAGE_CONFIG.secondaryClockLabel} ${formatZoneClockText(secondary)}`;
  marketHoursEtEl.textContent = PAGE_CONFIG.marketHoursLabel;
  const marketIsOpen = isConfiguredRegularSessionOpen(session);
  marketOpenStateEl.textContent = marketIsOpen ? "OPEN" : "CLOSED";
  marketOpenStateEl.classList.toggle("open", marketIsOpen);
  marketOpenStateEl.classList.toggle("closed", !marketIsOpen);
}

function startMarketClock() {
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

function setWatchLlmComment(message, isError = false) {
  if (!watchLlmTextEl) return;
  watchLlmTextEl.textContent = message || "-";
  watchLlmTextEl.classList.toggle("error", Boolean(isError));
}

function setWatchLlmMeta(message) {
  if (!watchLlmMetaEl) return;
  watchLlmMetaEl.textContent = message || "-";
}

function saveSymbolInsightsCache() {
  try {
    const items = [];
    symbolInsightsBySymbol.forEach((value, symbol) => {
      items.push({ symbol, ...value });
    });
    window.localStorage.setItem(
      SYMBOL_INSIGHTS_CACHE_KEY,
      JSON.stringify({ updated_at: new Date().toISOString(), items }),
    );
  } catch (_error) {
    // ignore local cache errors
  }
}

function saveWatchlistSymbolsCache(symbols = selectedSymbols) {
  try {
    const safeSymbols = uniqueSymbols(symbols).slice(0, MAX_SYMBOLS);
    window.localStorage.setItem(
      WATCHLIST_SYMBOLS_CACHE_KEY,
      JSON.stringify({ updated_at: new Date().toISOString(), symbols: safeSymbols }),
    );
  } catch (_error) {
    // ignore local cache errors
  }
}

function saveWatchlistRowsCache() {
  try {
    const items = [];
    latestRowsBySymbol.forEach((row, symbol) => {
      const normalized = normalizeSymbol(symbol);
      if (!normalized) return;
      const priceNum = Number(row?.price);
      if (!(Number.isFinite(priceNum) && priceNum > 0)) return;
      items.push({
        symbol: normalized,
        price: priceNum,
        timestamp: row?.timestamp ?? null,
        source: row?.source ?? null,
      });
    });
    window.localStorage.setItem(
      WATCHLIST_ROWS_CACHE_KEY,
      JSON.stringify({ updated_at: new Date().toISOString(), items }),
    );
  } catch (_error) {
    // ignore local cache errors
  }
}

function restoreWatchlistRowsCache() {
  try {
    const raw = window.localStorage.getItem(WATCHLIST_ROWS_CACHE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    const items = Array.isArray(parsed?.items) ? parsed.items : [];
    items.forEach((item) => {
      const symbol = normalizeSymbol(item?.symbol);
      if (!symbol) return;
      const priceNum = Number(item?.price);
      latestRowsBySymbol.set(symbol, {
        symbol,
        price: Number.isFinite(priceNum) && priceNum > 0 ? priceNum : null,
        timestamp: item?.timestamp ?? null,
        source: item?.source ?? null,
      });
    });
  } catch (_error) {
    // ignore local cache errors
  }
}

function restoreWatchlistSymbolsCache() {
  try {
    const raw = window.localStorage.getItem(WATCHLIST_SYMBOLS_CACHE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return uniqueSymbols(parsed?.symbols).slice(0, MAX_SYMBOLS);
  } catch (_error) {
    return [];
  }
}

function restoreSymbolInsightsCache() {
  try {
    const raw = window.localStorage.getItem(SYMBOL_INSIGHTS_CACHE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    const items = Array.isArray(parsed?.items) ? parsed.items : [];
    items.forEach((item) => {
      const symbol = normalizeSymbol(item?.symbol);
      if (!symbol) return;
      const latestClose = Number(item?.latest_close);
      const previousClose = Number(item?.previous_close);
      const currentPrice = Number(item?.current_price);
      const referenceClose = Number(item?.reference_close);
      const changePct = Number(item?.change_pct);
      const trend = (Array.isArray(item?.trend_30d) ? item.trend_30d : [])
        .map((point) => Number(point))
        .filter((num) => Number.isFinite(num));
      symbolInsightsBySymbol.set(symbol, {
        current_price: Number.isFinite(currentPrice) && currentPrice > 0 ? currentPrice : null,
        latest_close: Number.isFinite(latestClose) ? latestClose : null,
        previous_close: Number.isFinite(previousClose) ? previousClose : null,
        reference_close: Number.isFinite(referenceClose) && referenceClose > 0 ? referenceClose : null,
        change_pct: Number.isFinite(changePct) ? changePct : null,
        trend_30d: trend,
      });
    });
  } catch (_error) {
    // ignore local cache errors
  }
}

function applyInitialSelectedSymbols(symbols, { primeIgnoreEmpty = false } = {}) {
  const normalized = uniqueSymbols(symbols).slice(0, MAX_SYMBOLS);
  if (normalized.length === 0) {
    return false;
  }
  if (primeIgnoreEmpty) {
    ignoreFirstEmptySymbolsEvent = true;
  }
  selectedSymbols = normalized;
  saveWatchlistSymbolsCache(selectedSymbols);
  refreshWatchlist();
  void loadSavedWatchlistCommentary();
  return true;
}

async function loadWatchlistStateFromServer() {
  const params = new URLSearchParams({ namespace: WATCHLIST_NAMESPACE });
  const { response, result } = await fetchJson(`/api/watchlist-state?${params.toString()}`);
  if (!response.ok) {
    throw new Error(result.detail || "Failed to load watchlist state.");
  }
  return uniqueSymbols(result?.symbols).slice(0, MAX_SYMBOLS);
}

async function persistWatchlistStateToServer(symbols = selectedSymbols) {
  const safeSymbols = uniqueSymbols(symbols).slice(0, MAX_SYMBOLS);
  const { response, result } = await fetchJson("/api/watchlist-state", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      namespace: WATCHLIST_NAMESPACE,
      symbols: safeSymbols,
    }),
  });
  if (!response.ok) {
    throw new Error(result.detail || "Failed to persist watchlist state.");
  }
  return uniqueSymbols(result?.symbols || safeSymbols).slice(0, MAX_SYMBOLS);
}

async function refreshWatchlistCommentary(refresh = false) {
  if (!PAGE_CONFIG.watchlistCommentaryEnabled) return;
  if (!watchLlmTextEl || !watchLlmMetaEl) return;
  const symbols = uniqueSymbols(selectedSymbols).slice(0, MAX_SYMBOLS);
  if (symbols.length < 2) {
    watchLlmLastSymbolsKey = "";
    setWatchLlmComment("2銘柄以上でコメントを表示します。");
    setWatchLlmMeta(`モデル: ${watchLlmModelName}`);
    return;
  }

  const symbolsKey = symbols.join(",");
  if (!refresh && !watchLlmInFlight && watchLlmLastSymbolsKey === symbolsKey) {
    return;
  }
  if (watchLlmInFlight) {
    watchLlmQueuedRefresh = watchLlmQueuedRefresh || refresh;
    return;
  }

  watchLlmInFlight = true;
  if (watchLlmRefreshBtn) watchLlmRefreshBtn.disabled = true;
  setWatchLlmMeta("コメント生成中...");

  try {
    const params = new URLSearchParams({ symbols: symbols.join(",") });
    if (refresh) params.set("refresh", "true");
    const { response, result } = await fetchJson(`/api/watchlist-commentary?${params.toString()}`);
    if (!response.ok) {
      throw new Error(result.detail || "LLMコメントの生成に失敗しました。");
    }

    const comment = String(result?.comment || "").trim();
    if (!comment) {
      throw new Error("LLMコメントが空でした。");
    }

    setWatchLlmComment(comment);
    const model = String(result?.model || WATCHLIST_MODEL_UNKNOWN).trim() || WATCHLIST_MODEL_UNKNOWN;
    watchLlmModelName = model;
    const updatedAt = formatDateTime(result?.generated_at);
    setWatchLlmMeta(`${model} / ${updatedAt}`);
    watchLlmLastSymbolsKey = symbolsKey;
  } catch (error) {
    setWatchLlmComment(error instanceof Error ? error.message : "LLMコメントの生成に失敗しました。", true);
    setWatchLlmMeta(`モデル: ${watchLlmModelName}`);
  } finally {
    watchLlmInFlight = false;
    if (watchLlmRefreshBtn) watchLlmRefreshBtn.disabled = false;
    if (watchLlmQueuedRefresh) {
      const queuedNeedsRefresh = watchLlmQueuedRefresh;
      watchLlmQueuedRefresh = false;
      void refreshWatchlistCommentary(queuedNeedsRefresh);
    }
  }
}

async function loadSavedWatchlistCommentary() {
  if (!PAGE_CONFIG.watchlistCommentaryEnabled) return;
  if (!watchLlmTextEl || !watchLlmMetaEl) return;
  try {
    const { response, result } = await fetchJson("/api/watchlist-commentary/latest");
    if (!response.ok) return;
    const comment = String(result?.comment || "").trim();
    if (!comment) return;
    const symbols = Array.isArray(result?.symbols) ? uniqueSymbols(result.symbols) : [];
    if (symbols.length >= 2) {
      const key = symbols.join(",");
      if (key !== uniqueSymbols(selectedSymbols).join(",")) return;
      watchLlmLastSymbolsKey = key;
    }
    setWatchLlmComment(comment);
    const model = String(result?.model || WATCHLIST_MODEL_UNKNOWN).trim() || WATCHLIST_MODEL_UNKNOWN;
    watchLlmModelName = model;
    setWatchLlmMeta(`${model} / ${formatDateTime(result?.generated_at)}`);
  } catch (_error) {
    // ignore
  }
}

function enableContextualScrollbar(target) {
  if (!(target instanceof HTMLElement)) return;
  target.classList.add("auto-scrollbar");
  let hideTimer = null;

  const show = () => {
    target.classList.add("show-scrollbar");
    if (hideTimer) window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => {
      if (!target.matches(":hover")) {
        target.classList.remove("show-scrollbar");
      }
    }, 850);
  };

  const hide = () => {
    if (hideTimer) window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => {
      target.classList.remove("show-scrollbar");
    }, 220);
  };

  target.addEventListener("wheel", show, { passive: true });
  target.addEventListener("scroll", show, { passive: true });
  target.addEventListener("mouseenter", show);
  target.addEventListener("mouseleave", hide);
}

function setStatus(status) {
  if (!modeEl || !wsStateEl || !fallbackEl || !creditsLeftEl || !creditsUpdatedEl) return;
  modeEl.textContent = status?.mode ?? "-";
  const provider = String(status?.provider || "").toLowerCase();
  const wsAvailable = PAGE_CONFIG.streamEnabled && provider !== "fmp";
  wsStateEl.textContent = wsAvailable ? (status?.ws_connected ? "connected" : "disconnected") : "n/a";
  const fallbackValue = status?.fallback_poll_interval_sec;
  fallbackEl.textContent = Number.isFinite(Number(fallbackValue))
    ? `${fallbackValue} sec`
    : (fallbackValue ?? "-");
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
    refreshCreditsBtn.disabled = !PAGE_CONFIG.streamEnabled || provider === "fmp";
  }
  refreshWatchlist();
}

function findCatalogName(symbol) {
  const normalizedSymbol = normalizeSymbol(symbol);
  if (!normalizedSymbol) return "";
  const metaName = String(symbolMetaBySymbol.get(normalizedSymbol)?.name || "").trim();
  if (metaName) return metaName;
  const entry = symbolCatalog.find((item) => item.symbol === normalizedSymbol);
  if (entry?.name) {
    upsertSymbolMeta(normalizedSymbol, entry);
    return entry.name;
  }
  const overviewName = String(tabStateBySymbol.get(normalizedSymbol)?.overview?.name || "").trim();
  if (overviewName) {
    upsertSymbolMeta(normalizedSymbol, {
      name: overviewName,
      exchange: tabStateBySymbol.get(normalizedSymbol)?.overview?.exchange,
    });
    return overviewName;
  }
  return "";
}

function findCatalogExchange(symbol) {
  const normalizedSymbol = normalizeSymbol(symbol);
  if (!normalizedSymbol) return "";
  const metaExchange = String(symbolMetaBySymbol.get(normalizedSymbol)?.exchange || "").trim();
  if (metaExchange) return metaExchange;
  const entry = symbolCatalog.find((item) => item.symbol === normalizedSymbol);
  if (entry?.exchange) {
    upsertSymbolMeta(normalizedSymbol, entry);
    return entry.exchange;
  }
  const overviewExchange = String(tabStateBySymbol.get(normalizedSymbol)?.overview?.exchange || "").trim();
  if (overviewExchange) {
    upsertSymbolMeta(normalizedSymbol, {
      name: tabStateBySymbol.get(normalizedSymbol)?.overview?.name,
      exchange: overviewExchange,
    });
    return overviewExchange;
  }
  return "";
}

function computeChangePct(symbol, currentPrice, insight = symbolInsightsBySymbol.get(symbol)) {
  const priceNum = Number(currentPrice);
  const referenceClose = Number(insight?.reference_close);
  if (Number.isFinite(priceNum) && priceNum > 0 && Number.isFinite(referenceClose) && referenceClose > 0) {
    return percentChange(priceNum, referenceClose);
  }
  const latestClose = Number(insight?.latest_close);
  const previousClose = Number(insight?.previous_close);
  const fallbackReferenceClose = Number.isFinite(previousClose) && previousClose > 0
    ? previousClose
    : latestClose;

  if (!Number.isFinite(priceNum) || priceNum <= 0 || !Number.isFinite(fallbackReferenceClose) || fallbackReferenceClose <= 0) {
    return null;
  }
  return percentChange(priceNum, fallbackReferenceClose);
}

function ensureWatchItem(symbol) {
  if (watchItemsBySymbol.has(symbol)) {
    return watchItemsBySymbol.get(symbol);
  }

  const row = document.createElement("div");
  row.className = "watch-item";
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  row.dataset.symbol = symbol;
  row.innerHTML = `
    <div class="watch-main">
      <div class="watch-symbol">${symbol}</div>
      <div class="watch-name">${findCatalogName(symbol) || "名称未取得"}</div>
    </div>
    <div class="watch-metric" data-col="price">-</div>
    <div class="watch-metric" data-col="change-pct">-</div>
    <button type="button" class="watch-remove" data-symbol="${symbol}" aria-label="remove">×</button>
  `;
  watchItemsBySymbol.set(symbol, row);
  watchlistEl.appendChild(row);
  return row;
}

function computeWatchMetrics(symbol) {
  const state = tabStateBySymbol.get(symbol);
  const overview = state?.overview;
  const overviewPrice = Number(overview?.price?.current);
  const overviewChangePct = Number(overview?.price?.change_pct);
  const latest = latestRowsBySymbol.get(symbol);
  const insight = symbolInsightsBySymbol.get(symbol);
  const cachedPrice = Number(latest?.price);
  const insightPrice = Number(insight?.current_price);
  const insightChangePct = Number(insight?.change_pct);
  const price = (Number.isFinite(overviewPrice) && overviewPrice > 0)
    ? overviewPrice
    : ((Number.isFinite(cachedPrice) && cachedPrice > 0)
      ? cachedPrice
      : (Number.isFinite(insightPrice) && insightPrice > 0 ? insightPrice : null));
  const pct = Number.isFinite(overviewChangePct)
    ? overviewChangePct
    : (Number.isFinite(price) ? computeChangePct(symbol, price, insight) : insightChangePct);
  return {
    price: price > 0 ? price : null,
    changePct: Number.isFinite(pct) ? pct : null,
  };
}

function compareWatchValues(a, b, direction) {
  const aMissing = a === null || a === undefined || Number.isNaN(a);
  const bMissing = b === null || b === undefined || Number.isNaN(b);
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

function sortedWatchSymbols(symbols) {
  const safe = uniqueSymbols(symbols);
  const { key, direction } = watchSortState;
  if (key === "symbol") {
    return [...safe].sort((left, right) => compareWatchValues(left, right, direction));
  }

  return [...safe].sort((left, right) => {
    const leftMetrics = computeWatchMetrics(left);
    const rightMetrics = computeWatchMetrics(right);
    const leftValue = key === "price" ? leftMetrics.price : leftMetrics.changePct;
    const rightValue = key === "price" ? rightMetrics.price : rightMetrics.changePct;
    const primary = compareWatchValues(leftValue, rightValue, direction);
    if (primary !== 0) return primary;
    return left.localeCompare(right, "en", { sensitivity: "base" });
  });
}

function updateWatchSortHeaderUI() {
  if (!watchlistHeadEl) return;
  const buttons = watchlistHeadEl.querySelectorAll(".watch-sort-btn");
  buttons.forEach((button) => {
    const key = button.dataset.sortKey;
    const isActive = key === watchSortState.key;
    const baseLabel = key === "symbol" ? "銘柄" : (key === "price" ? "現在値" : "前日比%");
    button.classList.toggle("active", isActive);
    button.classList.toggle("sorted-asc", isActive && watchSortState.direction === "asc");
    button.classList.toggle("sorted-desc", isActive && watchSortState.direction === "desc");
    button.textContent = isActive
      ? `${baseLabel} ${watchSortState.direction === "asc" ? "↑" : "↓"}`
      : baseLabel;
  });
}

function refreshWatchlist() {
  const symbols = uniqueSymbols(selectedSymbols);
  const sortedSymbols = sortedWatchSymbols(symbols);
  const activeSet = new Set(symbols);

  Array.from(watchItemsBySymbol.keys()).forEach((symbol) => {
    if (!activeSet.has(symbol)) {
      const item = watchItemsBySymbol.get(symbol);
      if (item) item.remove();
      watchItemsBySymbol.delete(symbol);
    }
  });

  updateWatchSortHeaderUI();

  const fragment = document.createDocumentFragment();
  sortedSymbols.forEach((symbol) => {
    const item = ensureWatchItem(symbol);
    item.classList.toggle("active", symbol === activeTabSymbol);
    const nameEl = item.querySelector(".watch-name");
    const priceEl = item.querySelector('[data-col="price"]');
    const changePctEl = item.querySelector('[data-col="change-pct"]');
    item.title = buildSymbolDisplayLabel(symbol);

    if (nameEl) {
      nameEl.textContent = findCatalogName(symbol) || "名称未取得";
    }

    if (priceEl && changePctEl) {
      const metrics = computeWatchMetrics(symbol);
      priceEl.classList.remove("up", "down");
      changePctEl.classList.remove("up", "down");
      if (Number.isFinite(metrics.price)) {
        priceEl.textContent = `${PAGE_CONFIG.currencySymbol}${formatPrice(metrics.price)}`;
      } else {
        priceEl.textContent = "-";
      }
      if (Number.isFinite(metrics.changePct)) {
        changePctEl.textContent = formatSignedPercent(metrics.changePct);
        if (metrics.changePct > 0) {
          priceEl.classList.add("up");
          changePctEl.classList.add("up");
        }
        if (metrics.changePct < 0) {
          priceEl.classList.add("down");
          changePctEl.classList.add("down");
        }
      } else {
        changePctEl.textContent = "-";
      }
    }
    fragment.appendChild(item);
  });

  watchlistEl.innerHTML = "";
  watchlistEl.appendChild(fragment);
}

function applySymbolsFromServer(symbols) {
  const normalized = uniqueSymbols(symbols).slice(0, MAX_SYMBOLS);
  if (normalized.length === 0 && selectedSymbols.length > 0) {
    // 空の更新イベントで既存ウォッチリストを消さない
    return;
  }
  selectedSymbols = normalized;
  saveWatchlistSymbolsCache(selectedSymbols);
  refreshWatchlist();
  void loadSavedWatchlistCommentary();
  if (document.activeElement === symbolSearchInput) {
    renderDropdown();
  }
}

function extractSymbolsFromPayload(data) {
  const fromStatus = Array.isArray(data?.status?.symbols) ? data.status.symbols : [];
  if (fromStatus.length > 0) return uniqueSymbols(fromStatus).slice(0, MAX_SYMBOLS);
  const fromSymbols = Array.isArray(data?.symbols) ? data.symbols : [];
  if (fromSymbols.length > 0) return uniqueSymbols(fromSymbols).slice(0, MAX_SYMBOLS);
  const rows = Array.isArray(data?.rows) ? data.rows : [];
  return uniqueSymbols(rows.map((row) => row?.symbol)).slice(0, MAX_SYMBOLS);
}

function showDropdown() {
  symbolDropdown.classList.remove("hidden");
}

function hideDropdown() {
  symbolDropdown.classList.add("hidden");
}

function pickCandidates(query) {
  const needle = normalizeSearchText(query);
  if (symbolCatalog.length === 0) return [];

  if (!needle) {
    return symbolCatalog
      .filter((item) => !selectedSymbols.includes(item.symbol))
      .slice(0, MAX_DROPDOWN_ITEMS);
  }

  const ranked = [];
  for (const item of symbolCatalog) {
    if (selectedSymbols.includes(item.symbol)) continue;
    const symbolText = normalizeSearchText(item.symbol);
    const nameText = normalizeSearchText(item.name);
    let rank = -1;
    if (symbolText.startsWith(needle)) {
      rank = 0;
    } else if (nameText.startsWith(needle)) {
      rank = 1;
    } else if (symbolText.includes(needle)) {
      rank = 2;
    } else if (nameText.includes(needle)) {
      rank = 3;
    }
    if (rank < 0) continue;
    ranked.push({ item, rank });
  }

  ranked.sort((left, right) => {
    if (left.rank !== right.rank) return left.rank - right.rank;
    return left.item.symbol.localeCompare(right.item.symbol, "en", { sensitivity: "base" });
  });
  return ranked.slice(0, MAX_DROPDOWN_ITEMS).map(({ item }) => item);
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
    btn.textContent = buildSymbolOptionLabel(item);
    btn.title = buildSymbolOptionLabel(item);
    symbolDropdown.appendChild(btn);
  });

  showDropdown();
}

async function updateSymbolsOnServer() {
  if (!PAGE_CONFIG.serverSymbolSync) {
    setSelectionError("");
    saveWatchlistSymbolsCache(selectedSymbols);
    refreshWatchlist();
    try {
      const persistedSymbols = await persistWatchlistStateToServer(selectedSymbols);
      selectedSymbols = persistedSymbols;
      saveWatchlistSymbolsCache(selectedSymbols);
      refreshWatchlist();
    } catch (_error) {
      // keep client-side watchlist even if file persistence fails
    }
    await loadSymbolInsights(selectedSymbols, false);
    return;
  }

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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbols: payloadSymbols.join(",") }),
      });
      if (!response.ok) {
        setSelectionError(result.detail || "Failed to update symbols");
        break;
      }

      setSelectionError("");
      setStatus(result.status || {});
      const serverSymbols = Array.isArray(result.symbols) ? result.symbols : payloadSymbols;
      const rows = Array.isArray(result.rows) ? result.rows : [];
      rows.forEach((row) => {
        const symbol = normalizeSymbol(row?.symbol);
        if (!symbol) return;
        upsertLatestRow(symbol, {
          price: row?.price ?? null,
          timestamp: row?.timestamp ?? null,
          source: row?.source ?? null,
        });
      });
      saveWatchlistRowsCache();
      applySymbolsFromServer(serverSymbols);
    } while (syncQueued);
  } finally {
    syncInFlight = false;
  }
}

async function addSymbol(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;

  if (selectedSymbols.includes(normalized)) {
    openSymbolTab(normalized);
    return;
  }

  if (selectedSymbols.length >= MAX_SYMBOLS) {
    setSelectionError(`You can monitor up to ${MAX_SYMBOLS} symbols.`);
    return;
  }

  selectedSymbols = [...selectedSymbols, normalized];
  setSelectionError("");
  await updateSymbolsOnServer();
  openSymbolTab(normalized);
}

async function removeSymbol(symbol) {
  if (!selectedSymbols.includes(symbol)) return;
  if (PAGE_CONFIG.serverSymbolSync && selectedSymbols.length <= 1) {
    setSelectionError("At least one symbol is required.");
    return;
  }

  selectedSymbols = selectedSymbols.filter((item) => item !== symbol);
  closeSymbolTab(symbol);
  setSelectionError("");
  await updateSymbolsOnServer();
}

async function loadSymbolCatalog(refresh = false, cacheOnly = false) {
  if (!refreshCatalogBtn) {
    return {
      loaded: false,
      count: 0,
      source: "unavailable",
      cacheOnly,
    };
  }
  refreshCatalogBtn.disabled = true;
  if (cacheOnly) {
    setCatalogMeta("Loading saved symbol catalog...");
  } else {
    setCatalogMeta(refresh ? "Refreshing symbol catalog..." : "Loading symbol catalog...");
  }

  try {
    const params = new URLSearchParams();
    if (refresh) {
      params.set("refresh", "true");
    } else if (cacheOnly) {
      params.set("cache_only", "true");
    }
    if (PAGE_CONFIG.catalogCountry) {
      params.set("country", PAGE_CONFIG.catalogCountry);
    }
    const url = params.size > 0 ? `/api/symbol-catalog?${params.toString()}` : "/api/symbol-catalog";
    const { response, result } = await fetchJson(url);
    if (!response.ok) {
      setCatalogMeta(result.detail || "Failed to load symbol catalog");
      return {
        loaded: false,
        count: 0,
        source: "error",
        cacheOnly,
      };
    }

    const rawSymbols = Array.isArray(result.symbols) ? result.symbols : [];
    symbolCatalog = rawSymbols.map((item) => ({
      symbol: normalizeSymbol(item.symbol),
      name: String(item.name || "").trim(),
      exchange: String(item.exchange || "").trim(),
    }));
    symbolCatalog.forEach((item) => {
      upsertSymbolMeta(item.symbol, item);
    });

    const source = String(result.source || "unknown");
    const count = symbolCatalog.length;
    const updatedText = result.updated_at ? `updated ${formatTime(result.updated_at)}` : "updated -";
    setCatalogMeta(`${count.toLocaleString()} symbols loaded (${source}, ${updatedText})`);
    refreshWatchlist();

    if (document.activeElement === symbolSearchInput) {
      renderDropdown();
    }

    return {
      loaded: count > 0,
      count,
      source,
      cacheOnly,
    };
  } finally {
    refreshCatalogBtn.disabled = false;
  }
}

async function bootstrapSymbolCatalog() {
  try {
    const cachedResult = await loadSymbolCatalog(false, true);
    if (cachedResult?.loaded) return;
    const liveResult = await loadSymbolCatalog(false, false);
    if (liveResult?.loaded) return;
    setCatalogMeta("Symbol list is not loaded. Click Refresh Symbols when needed.");
  } catch (_error) {
    setCatalogMeta("Symbol list is not loaded. Click Refresh Symbols when needed.");
  }
}

async function loadSymbolInsights(symbols, refresh = false) {
  const targets = uniqueSymbols(symbols)
    .slice(0, MAX_SYMBOLS)
    .filter((symbol) => refresh || !symbolInsightsBySymbol.has(symbol))
    .filter((symbol) => !sparklineFetchInFlight.has(symbol));

  if (targets.length === 0) {
    refreshWatchlist();
    return;
  }

  targets.forEach((symbol) => sparklineFetchInFlight.add(symbol));

  try {
    const params = applySparklineModeParams(new URLSearchParams({ symbols: targets.join(",") }));
    if (refresh) params.set("refresh", "true");
    const { response, result } = await fetchJson(`/api/sparkline?${params.toString()}`);
    if (!response.ok || !Array.isArray(result.items)) return;

    result.items.forEach((item) => {
      const symbol = normalizeSymbol(item?.symbol);
      if (!symbol) return;
      const latestClose = Number(item?.latest_close);
      const previousClose = Number(item?.previous_close);
      const currentPrice = Number(item?.current_price);
      const referenceClose = Number(item?.reference_close);
      const changePct = Number(item?.change_pct);
      const trend = (Array.isArray(item?.trend_30d) ? item.trend_30d : [])
        .map((point) => Number(point))
        .filter((num) => Number.isFinite(num));

      symbolInsightsBySymbol.set(symbol, {
        current_price: Number.isFinite(currentPrice) && currentPrice > 0 ? currentPrice : null,
        latest_close: Number.isFinite(latestClose) ? latestClose : null,
        previous_close: Number.isFinite(previousClose) ? previousClose : null,
        reference_close: Number.isFinite(referenceClose) && referenceClose > 0 ? referenceClose : null,
        change_pct: Number.isFinite(changePct) ? changePct : null,
        trend_30d: trend,
      });

      // EODモードでは終値、通常モードではquote現在値を優先する。
      const existingRow = latestRowsBySymbol.get(symbol);
      const hasValidPrice = Number(existingRow?.price) > 0;
      const fallbackPrice = (Number.isFinite(currentPrice) && currentPrice > 0) ? currentPrice : latestClose;
      const hasCurrentPrice = Number.isFinite(currentPrice) && currentPrice > 0;
      const resolvedSource = resolveSparklinePriceSource(item, hasCurrentPrice);
      const isEodClose = resolvedSource === "eod_close";
      const marketOpen = isConfiguredMarketOpenNow();
      const shouldReplaceCachedPrice = (
        isEodClose
          ? (!hasValidPrice || !marketOpen || !PAGE_CONFIG.streamEnabled)
          : (!hasValidPrice || refresh || !PAGE_CONFIG.streamEnabled || hasCurrentPrice)
      );
      if (shouldReplaceCachedPrice && Number.isFinite(fallbackPrice) && fallbackPrice > 0) {
        latestRowsBySymbol.set(symbol, {
          ...(existingRow ?? {}),
          symbol,
          price: fallbackPrice,
          timestamp: item?.updated_at ?? item?.latest_close_date ?? existingRow?.timestamp ?? null,
          source: (refresh || resolvedSource === "eod_close") ? resolvedSource : (existingRow?.source || resolvedSource),
        });
        saveWatchlistRowsCache();
      }
    });
    saveSymbolInsightsCache();
  } catch (_error) {
    // Keep UI functional even if insight request fails.
  } finally {
    targets.forEach((symbol) => sparklineFetchInFlight.delete(symbol));
  }

  refreshWatchlist();
}

function renderTabs() {
  symbolTabsEl.innerHTML = "";
  openTabs.forEach((symbol) => {
    const tabBtn = document.createElement("button");
    tabBtn.type = "button";
    tabBtn.className = "symbol-tab";
    tabBtn.dataset.symbol = symbol;
    tabBtn.draggable = true;
    tabBtn.classList.toggle("active", symbol === activeTabSymbol);
    tabBtn.title = buildSymbolDisplayLabel(symbol);
    tabBtn.innerHTML = `<span>${symbol}</span><span class="tab-close" data-close-symbol="${symbol}" aria-label="close">×</span>`;
    symbolTabsEl.appendChild(tabBtn);
  });
}

function captureTabRects() {
  const rects = new Map();
  if (!symbolTabsEl) return rects;
  symbolTabsEl.querySelectorAll(".symbol-tab").forEach((tab) => {
    const symbol = tab.dataset.symbol || "";
    if (!symbol) return;
    rects.set(symbol, tab.getBoundingClientRect());
  });
  return rects;
}

function animateTabReorder(prevRects) {
  if (!symbolTabsEl || !(prevRects instanceof Map) || prevRects.size === 0) return;
  symbolTabsEl.querySelectorAll(".symbol-tab").forEach((tab) => {
    const symbol = tab.dataset.symbol || "";
    if (!symbol) return;
    const prev = prevRects.get(symbol);
    if (!prev) return;
    const next = tab.getBoundingClientRect();
    const dx = prev.left - next.left;
    if (Math.abs(dx) < 1) return;

    tab.style.transition = "none";
    tab.style.transform = `translateX(${dx}px)`;
    tab.classList.add("reordering");
    void tab.offsetWidth;
    tab.style.transition = "";
    tab.style.transform = "";
    const onDone = () => {
      tab.classList.remove("reordering");
      tab.removeEventListener("transitionend", onDone);
    };
    tab.addEventListener("transitionend", onDone);
  });
}

function clearTabDragVisualState() {
  if (!symbolTabsEl) return;
  symbolTabsEl.querySelectorAll(".symbol-tab").forEach((tab) => {
    tab.classList.remove("dragging", "drag-over-left", "drag-over-right");
  });
}

function getDropPlacement(clientX, dragSymbol = "") {
  if (!symbolTabsEl) return null;
  const tabs = Array.from(symbolTabsEl.querySelectorAll(".symbol-tab"));
  const candidates = tabs.filter((tab) => (tab.dataset.symbol || "") !== dragSymbol);
  const effectiveTabs = candidates.length ? candidates : tabs;
  if (!effectiveTabs.length) return null;

  for (const tab of effectiveTabs) {
    const symbol = tab.dataset.symbol || "";
    const rect = tab.getBoundingClientRect();
    const withinTab = clientX >= rect.left && clientX <= rect.right;
    if (!withinTab) continue;
    return {
      targetSymbol: symbol,
      placeBefore: clientX < (rect.left + (rect.width / 2)),
      tab,
    };
  }

  // Handle gaps between tabs by choosing the nearest tab center.
  let nearestTab = null;
  let nearestDistance = Infinity;
  effectiveTabs.forEach((tab) => {
    const rect = tab.getBoundingClientRect();
    const center = rect.left + (rect.width / 2);
    const distance = Math.abs(clientX - center);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestTab = tab;
    }
  });
  if (!nearestTab) return null;
  const targetSymbol = nearestTab.dataset.symbol || dragSymbol || "";
  const rect = nearestTab.getBoundingClientRect();
  return {
    targetSymbol,
    placeBefore: clientX < (rect.left + (rect.width / 2)),
    tab: nearestTab,
  };
}

function reorderOpenTabs(dragSymbol, targetSymbol, placeBefore) {
  const fromIdx = openTabs.indexOf(dragSymbol);
  const targetIdxRaw = openTabs.indexOf(targetSymbol);
  if (fromIdx < 0 || targetIdxRaw < 0) return;
  if (dragSymbol === targetSymbol) return;

  const next = [...openTabs];
  next.splice(fromIdx, 1);
  const targetIdx = next.indexOf(targetSymbol);
  const insertIdx = placeBefore ? targetIdx : targetIdx + 1;
  next.splice(Math.max(0, insertIdx), 0, dragSymbol);
  openTabs = next;
}

function renderEmptyOrPanel() {
  const hasActive = Boolean(activeTabSymbol);
  tabEmptyEl.classList.toggle("hidden", hasActive);
  symbolPanelEl.classList.toggle("hidden", !hasActive);
}

function computeAtr(points, windowSize = 14) {
  const rows = Array.isArray(points) ? points : [];
  if (rows.length < windowSize + 1) return null;
  let prevClose = Number(rows[0]?.c);
  if (!Number.isFinite(prevClose)) return null;
  const trs = [];
  for (let i = 1; i < rows.length; i += 1) {
    const high = Number(rows[i]?.h);
    const low = Number(rows[i]?.l);
    const close = Number(rows[i]?.c);
    if (!Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close)) continue;
    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    if (Number.isFinite(tr)) trs.push(tr);
    prevClose = close;
  }
  if (trs.length < windowSize) return null;
  const sample = trs.slice(-windowSize);
  return sample.reduce((sum, value) => sum + value, 0) / sample.length;
}

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampChartViewport(start, end, totalPoints) {
  const maxIndex = Math.max(0, totalPoints - 1);
  if (maxIndex <= 0) return { start: 0, end: 0 };

  let nextStart = Number(start);
  let nextEnd = Number(end);
  if (!Number.isFinite(nextStart) || !Number.isFinite(nextEnd)) return { start: 0, end: maxIndex };
  if (nextEnd < nextStart) {
    const tmp = nextStart;
    nextStart = nextEnd;
    nextEnd = tmp;
  }

  const fullSpan = maxIndex;
  const minSpan = Math.max(1, Math.min(CHART_MIN_VISIBLE_POINTS - 1, fullSpan));
  let span = nextEnd - nextStart;

  if (span >= fullSpan) return { start: 0, end: maxIndex };
  if (span < minSpan) {
    const center = (nextStart + nextEnd) / 2;
    nextStart = center - (minSpan / 2);
    nextEnd = center + (minSpan / 2);
    span = minSpan;
  }

  if (nextStart < 0) {
    nextEnd -= nextStart;
    nextStart = 0;
  }
  if (nextEnd > maxIndex) {
    nextStart -= (nextEnd - maxIndex);
    nextEnd = maxIndex;
  }

  nextStart = clampNumber(nextStart, 0, maxIndex);
  nextEnd = clampNumber(nextEnd, 0, maxIndex);
  return { start: nextStart, end: nextEnd };
}

function getChartViewport(symbol, totalPoints) {
  const existing = chartViewportBySymbol.get(symbol);
  const viewport = existing
    ? clampChartViewport(existing.start, existing.end, totalPoints)
    : { start: 0, end: Math.max(0, totalPoints - 1) };
  chartViewportBySymbol.set(symbol, viewport);
  return viewport;
}

function setChartViewport(symbol, start, end, totalPoints) {
  const viewport = clampChartViewport(start, end, totalPoints);
  chartViewportBySymbol.set(symbol, viewport);
}

function buildLineChartHtml(symbol, points) {
  const safe = (Array.isArray(points) ? points : [])
    .map((p) => ({ t: p?.t, c: Number(p?.c), v: Number(p?.v) }))
    .filter((p) => Number.isFinite(p.c));

  if (safe.length < 2) {
    chartViewportBySymbol.delete(symbol);
    return '<div class="chart-empty">チャートデータがありません。</div>';
  }

  const viewport = getChartViewport(symbol, safe.length);
  const visibleStart = clampNumber(Math.floor(viewport.start), 0, safe.length - 1);
  const visibleEnd = clampNumber(Math.ceil(viewport.end), 0, safe.length - 1);
  const visible = safe.slice(visibleStart, visibleEnd + 1);

  let min = Math.min(...visible.map((p) => p.c));
  let max = Math.max(...visible.map((p) => p.c));
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const rawRange = max - min || 1;
  const yPadding = rawRange * CHART_Y_PADDING_RATIO;
  min -= yPadding;
  max += yPadding;
  const range = max - min || 1;
  const drawableWidth = CHART_WIDTH - (CHART_PAD_X * 2);
  const drawableHeight = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;
  const axisX = CHART_PAD_X;
  const axisY = CHART_HEIGHT - CHART_PAD_BOTTOM;
  const volumeBandHeight = drawableHeight * CHART_VOLUME_BAND_RATIO;

  const yTickCount = 4;
  const yTicks = [];
  const yGridLines = [];
  const xTickCount = 5;
  const xTicks = [];
  for (let i = 0; i < yTickCount; i += 1) {
    const ratio = i / (yTickCount - 1);
    const y = CHART_PAD_TOP + (ratio * drawableHeight);
    const value = max - (range * ratio);
    yTicks.push(`<text x="${(axisX - 4).toFixed(2)}" y="${(y + 4).toFixed(2)}" class="symbol-chart-axis-label" text-anchor="end">${formatPrice(value)}</text>`);
    yGridLines.push(`<line x1="${axisX}" y1="${y.toFixed(2)}" x2="${(CHART_WIDTH - CHART_PAD_X).toFixed(2)}" y2="${y.toFixed(2)}" class="symbol-chart-grid-line"></line>`);
  }
  for (let i = 0; i < xTickCount; i += 1) {
    const ratio = i / (xTickCount - 1);
    const localIdx = Math.round(ratio * Math.max(visible.length - 1, 1));
    const point = visible[localIdx];
    const x = CHART_PAD_X + (ratio * drawableWidth);
    const anchor = i === 0 ? "start" : (i === (xTickCount - 1) ? "end" : "middle");
    xTicks.push(`<line x1="${x.toFixed(2)}" y1="${axisY.toFixed(2)}" x2="${x.toFixed(2)}" y2="${(axisY + 5).toFixed(2)}" class="symbol-chart-axis-tick"></line>`);
    xTicks.push(`<text x="${x.toFixed(2)}" y="${(CHART_HEIGHT - 8).toFixed(2)}" class="symbol-chart-axis-label" text-anchor="${anchor}">${formatChartDateLabel(point?.t)}</text>`);
  }

  const polyline = visible.map((p, idx) => {
    const x = CHART_PAD_X + ((idx / Math.max(visible.length - 1, 1)) * drawableWidth);
    const y = CHART_PAD_TOP + (1 - ((p.c - min) / range)) * drawableHeight;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  const maxVolume = Math.max(
    0,
    ...visible.map((p) => (Number.isFinite(p.v) && p.v > 0 ? p.v : 0))
  );
  const clipId = `chart-volume-clip-${String(symbol || "sym").replace(/[^a-zA-Z0-9_-]/g, "_")}`;
  const barWidth = Math.max(1, (drawableWidth / Math.max(visible.length, 1)) * 0.68);
  const volumeBars = maxVolume > 0
    ? visible.map((p, idx) => {
      if (!Number.isFinite(p.v) || p.v <= 0) return "";
      const x = CHART_PAD_X + ((idx / Math.max(visible.length - 1, 1)) * drawableWidth);
      const barHeight = (p.v / maxVolume) * volumeBandHeight;
      const y = axisY - barHeight;
      return `<rect x="${(x - (barWidth / 2)).toFixed(2)}" y="${y.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${barHeight.toFixed(2)}" class="symbol-chart-volume-bar"></rect>`;
    }).join("")
    : "";
  const lastPoint = polyline.split(" ").pop() || `${CHART_PAD_X},${CHART_HEIGHT - CHART_PAD_BOTTOM}`;
  const [lastX, lastY] = lastPoint.split(",");

  const firstVisible = visible[0];
  const latestVisible = visible[visible.length - 1];
  const diffPct = percentChange(latestVisible.c, firstVisible.c) ?? 0;

  return `
    <div class="chart-controls">
      <div class="chart-toolbar">
        <div class="chart-zoom-actions">
          <button type="button" class="minor-action" data-chart-action="zoom-in">+</button>
          <button type="button" class="minor-action" data-chart-action="zoom-out">-</button>
          <button type="button" class="minor-action" data-chart-action="zoom-reset">Reset</button>
        </div>
        <div class="chart-range-actions">
          <button type="button" class="minor-action" data-chart-range="1w">1W</button>
          <button type="button" class="minor-action" data-chart-range="1m">1M</button>
          <button type="button" class="minor-action" data-chart-range="1y">1Y</button>
          <button type="button" class="minor-action" data-chart-range="5y">5Y</button>
          <button type="button" class="minor-action" data-chart-range="10y">10Y</button>
          <button type="button" class="minor-action" data-chart-range="max">Max</button>
        </div>
        <span class="chart-hint">Wheel: zoom / Drag: pan / Hover: price & volume</span>
      </div>
      <div class="chart-caption">${visible.length} points / ${firstVisible.t || "-"} - ${latestVisible.t || "-"} / ${formatSignedPercent(diffPct)}</div>
    </div>
    <div class="chart-scroll-shell">
      <div class="chart-canvas-host">
        <svg class="symbol-chart interactive" viewBox="0 0 ${CHART_WIDTH} ${CHART_HEIGHT}" preserveAspectRatio="none" role="img" aria-label="price chart" data-symbol="${symbol}">
          <defs>
            <linearGradient id="chartAreaFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="rgba(41, 210, 193, 0.34)" />
              <stop offset="100%" stop-color="rgba(41, 210, 193, 0.03)" />
            </linearGradient>
            <clipPath id="${clipId}">
              <rect x="${axisX}" y="${CHART_PAD_TOP}" width="${drawableWidth.toFixed(2)}" height="${drawableHeight.toFixed(2)}"></rect>
            </clipPath>
          </defs>
          <rect x="0" y="0" width="${CHART_WIDTH}" height="${CHART_HEIGHT}" class="symbol-chart-bg"></rect>
          ${yGridLines.join("")}
          <line x1="${axisX}" y1="${CHART_PAD_TOP}" x2="${axisX}" y2="${axisY}" class="symbol-chart-axis-line"></line>
          <line x1="${axisX}" y1="${axisY}" x2="${(CHART_WIDTH - CHART_PAD_X).toFixed(2)}" y2="${axisY}" class="symbol-chart-axis-line"></line>
          <g clip-path="url(#${clipId})">${volumeBars}</g>
          <polyline class="symbol-chart-line" points="${polyline}"></polyline>
          <circle class="symbol-chart-point" cx="${lastX}" cy="${lastY}" r="4"></circle>
          <line x1="0" y1="0" x2="0" y2="0" class="symbol-chart-hover-line hidden"></line>
          <circle cx="0" cy="0" r="4" class="symbol-chart-hover-point hidden"></circle>
          ${yTicks.join("")}
          ${xTicks.join("")}
        </svg>
        <div class="chart-tooltip hidden"></div>
      </div>
    </div>
  `;
}

function renderLineChart(symbol, points) {
  chartWrapEl.innerHTML = buildLineChartHtml(symbol, points);
  const safe = (Array.isArray(points) ? points : [])
    .map((p) => ({ t: p?.t, c: Number(p?.c), v: Number(p?.v) }))
    .filter((p) => Number.isFinite(p.c));
  if (safe.length < 2) return;

  const svg = chartWrapEl.querySelector(".symbol-chart.interactive");
  if (!svg) return;
  if (chartPanState.active && chartPanState.symbol === symbol) {
    svg.classList.add("dragging");
  }
  const zoomInBtn = chartWrapEl.querySelector('[data-chart-action="zoom-in"]');
  const zoomOutBtn = chartWrapEl.querySelector('[data-chart-action="zoom-out"]');
  const zoomResetBtn = chartWrapEl.querySelector('[data-chart-action="zoom-reset"]');
  const rangeButtons = chartWrapEl.querySelectorAll("[data-chart-range]");
  const tooltipEl = chartWrapEl.querySelector(".chart-tooltip");
  const hoverLineEl = svg.querySelector(".symbol-chart-hover-line");
  const hoverPointEl = svg.querySelector(".symbol-chart-hover-point");
  const viewport = getChartViewport(symbol, safe.length);
  const visibleStart = clampNumber(Math.floor(viewport.start), 0, safe.length - 1);
  const visibleEnd = clampNumber(Math.ceil(viewport.end), 0, safe.length - 1);
  const visibleCount = Math.max(1, visibleEnd - visibleStart + 1);
  const drawableWidth = CHART_WIDTH - (CHART_PAD_X * 2);
  const drawableHeight = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;
  const visible = safe.slice(visibleStart, visibleEnd + 1);
  let min = Math.min(...visible.map((p) => p.c));
  let max = Math.max(...visible.map((p) => p.c));
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const rawRange = max - min || 1;
  const yPadding = rawRange * CHART_Y_PADDING_RATIO;
  min -= yPadding;
  max += yPadding;
  const range = max - min || 1;
  const yFromPrice = (price) => CHART_PAD_TOP + (1 - ((price - min) / range)) * drawableHeight;

  const hideHover = () => {
    if (tooltipEl) tooltipEl.classList.add("hidden");
    if (hoverLineEl) hoverLineEl.classList.add("hidden");
    if (hoverPointEl) hoverPointEl.classList.add("hidden");
  };

  const getIndexFromClientX = (clientX) => {
    const rect = svg.getBoundingClientRect();
    const ratio = clampNumber((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    return ratio * Math.max(1, safe.length - 1);
  };

  const zoomAtClientX = (clientX, factor) => {
    const current = getChartViewport(symbol, safe.length);
    const maxIndex = Math.max(1, safe.length - 1);
    const fullSpan = maxIndex;
    const minSpan = Math.max(1, Math.min(CHART_MIN_VISIBLE_POINTS - 1, fullSpan));
    const currentSpan = Math.max(1, current.end - current.start);
    let targetSpan = clampNumber(currentSpan * factor, minSpan, fullSpan);
    if (Math.abs(targetSpan - currentSpan) < 0.001) return;
    const centerIndex = clampNumber(getIndexFromClientX(clientX), 0, maxIndex);
    const ratio = (centerIndex - current.start) / currentSpan;
    const nextStart = centerIndex - (targetSpan * ratio);
    const nextEnd = nextStart + targetSpan;
    setChartViewport(symbol, nextStart, nextEnd, safe.length);
    chartRangePresetBySymbol.set(symbol, "");
    renderLineChart(symbol, points);
  };

  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => {
      const rect = svg.getBoundingClientRect();
      zoomAtClientX(rect.left + (rect.width / 2), CHART_ZOOM_IN_FACTOR);
    });
  }
  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => {
      const rect = svg.getBoundingClientRect();
      zoomAtClientX(rect.left + (rect.width / 2), CHART_ZOOM_OUT_FACTOR);
    });
  }
  if (zoomResetBtn) {
    zoomResetBtn.addEventListener("click", () => {
      setChartViewport(symbol, 0, safe.length - 1, safe.length);
      chartRangePresetBySymbol.set(symbol, "max");
      renderLineChart(symbol, points);
    });
  }
  rangeButtons.forEach((btn) => {
    const preset = String(btn.getAttribute("data-chart-range") || "").toLowerCase();
    const isActive = chartRangePresetBySymbol.get(symbol) === preset;
    btn.classList.toggle("active", isActive);
    btn.addEventListener("click", () => {
      if (!CHART_RANGE_PRESETS.includes(preset)) return;
      applyChartRangePreset(symbol, points, preset);
      renderLineChart(symbol, points);
    });
  });

  svg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? CHART_ZOOM_IN_FACTOR : CHART_ZOOM_OUT_FACTOR;
      zoomAtClientX(event.clientX, factor);
    },
    { passive: false }
  );
  svg.addEventListener("mousemove", (event) => {
    if (chartPanState.active) {
      hideHover();
      return;
    }
    const rect = svg.getBoundingClientRect();
    const chartX = ((event.clientX - rect.left) / Math.max(1, rect.width)) * CHART_WIDTH;
    const chartY = ((event.clientY - rect.top) / Math.max(1, rect.height)) * CHART_HEIGHT;
    if (
      chartX < CHART_PAD_X
      || chartX > (CHART_WIDTH - CHART_PAD_X)
      || chartY < CHART_PAD_TOP
      || chartY > (CHART_HEIGHT - CHART_PAD_BOTTOM)
    ) {
      hideHover();
      return;
    }

    const ratio = clampNumber((chartX - CHART_PAD_X) / Math.max(1, drawableWidth), 0, 1);
    const localIdx = Math.round(ratio * Math.max(visibleCount - 1, 1));
    const idx = clampNumber(visibleStart + localIdx, 0, safe.length - 1);
    const point = safe[idx];
    if (!point) {
      hideHover();
      return;
    }

    const x = CHART_PAD_X + (((idx - visibleStart) / Math.max(visibleCount - 1, 1)) * drawableWidth);
    const y = yFromPrice(point.c);

    if (hoverLineEl) {
      hoverLineEl.setAttribute("x1", x.toFixed(2));
      hoverLineEl.setAttribute("x2", x.toFixed(2));
      hoverLineEl.setAttribute("y1", CHART_PAD_TOP.toFixed(2));
      hoverLineEl.setAttribute("y2", (CHART_HEIGHT - CHART_PAD_BOTTOM).toFixed(2));
      hoverLineEl.classList.remove("hidden");
    }
    if (hoverPointEl) {
      hoverPointEl.setAttribute("cx", x.toFixed(2));
      hoverPointEl.setAttribute("cy", y.toFixed(2));
      hoverPointEl.classList.remove("hidden");
    }

    if (tooltipEl) {
      const volumeText = Number.isFinite(point.v) ? formatCompact(point.v) : "-";
      tooltipEl.innerHTML = [
        `<div>${point.t || "-"}</div>`,
        `<div>Price: ${PAGE_CONFIG.currencySymbol}${formatPrice(point.c)}</div>`,
        `<div>Vol: ${volumeText}</div>`,
      ].join("");
      tooltipEl.classList.remove("hidden");
      const tooltipW = Math.min(320, Math.max(140, tooltipEl.offsetWidth));
      const tooltipH = Math.max(30, tooltipEl.offsetHeight);
      let left = (event.clientX - rect.left) + 12;
      if ((left + tooltipW) > rect.width) left = (event.clientX - rect.left) - tooltipW - 12;
      left = clampNumber(left, 6, Math.max(6, rect.width - tooltipW - 6));
      let top = (event.clientY - rect.top) - tooltipH - 10;
      if (top < 6) top = (event.clientY - rect.top) + 10;
      top = clampNumber(top, 6, Math.max(6, rect.height - tooltipH - 6));
      tooltipEl.style.left = `${left}px`;
      tooltipEl.style.top = `${top}px`;
    }
  });
  svg.addEventListener("mouseleave", () => {
    hideHover();
  });
  svg.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    hideHover();
    chartPanState = {
      active: true,
      symbol,
      points,
      lastClientX: event.clientX,
    };
    svg.classList.add("dragging");
    document.body.classList.add("chart-panning");
  });
  svg.addEventListener("dblclick", () => {
    setChartViewport(symbol, 0, safe.length - 1, safe.length);
    chartRangePresetBySymbol.set(symbol, "max");
    renderLineChart(symbol, points);
  });
}

function renderActiveTab() {
  renderTabs();
  renderEmptyOrPanel();
  refreshWatchlist();

  if (!activeTabSymbol) return;

  const state = tabStateBySymbol.get(activeTabSymbol);
  if (!state) return;

  panelSymbolEl.textContent = buildSymbolDisplayLabel(activeTabSymbol);

  if (state.loading) {
    panelUpdatedEl.textContent = "Loading...";
    chartWrapEl.innerHTML = '<div class="chart-empty">データ取得中...</div>';
    return;
  }

  if (state.error) {
    panelUpdatedEl.textContent = state.error;
    chartWrapEl.innerHTML = '<div class="chart-empty">取得に失敗しました。</div>';
    kpiPriceEl.textContent = "-";
    kpiChangeEl.textContent = "-";
    kpiRangeEl.textContent = "-";
    kpiVolumeEl.textContent = "-";
    kpiMaEl.textContent = "-";
    kpiAtrEl.textContent = "-";
    return;
  }

  const overview = state.overview;
  const historical = state.historical;
  const price = Number(overview?.price?.current);
  const changeAbs = Number(overview?.price?.change_abs);
  const changePct = Number(overview?.price?.change_pct);
  const dayHigh = Number(overview?.price?.day_high);
  const dayLow = Number(overview?.price?.day_low);
  const volume = Number(overview?.volume?.today);
  const ma20 = Number(overview?.technical?.ma_short_20);
  const ma50 = Number(overview?.technical?.ma_mid_50);
  const atr14 = Number(overview?.technical?.atr_14) || computeAtr(historical?.points || []);

  panelUpdatedEl.textContent = `updated ${formatTime(overview?.price?.updated_at)} / source ${overview?.source || "-"}`;
  kpiPriceEl.textContent = Number.isFinite(price) ? `${PAGE_CONFIG.currencySymbol}${formatPrice(price)}` : "-";
  kpiChangeEl.textContent = Number.isFinite(changePct)
    ? `${formatSignedCurrency(changeAbs)} (${formatSignedPercent(changePct)})`
    : "-";
  kpiChangeEl.classList.toggle("up", Number.isFinite(changePct) && changePct > 0);
  kpiChangeEl.classList.toggle("down", Number.isFinite(changePct) && changePct < 0);
  kpiRangeEl.textContent = (Number.isFinite(dayHigh) && Number.isFinite(dayLow))
    ? `${PAGE_CONFIG.currencySymbol}${formatPrice(dayHigh)} / ${PAGE_CONFIG.currencySymbol}${formatPrice(dayLow)}`
    : "-";
  kpiVolumeEl.textContent = Number.isFinite(volume) ? formatCompact(volume) : "-";
  kpiMaEl.textContent = (Number.isFinite(ma20) && Number.isFinite(ma50))
    ? `${PAGE_CONFIG.currencySymbol}${formatPrice(ma20)} / ${PAGE_CONFIG.currencySymbol}${formatPrice(ma50)}`
    : "-";
  kpiAtrEl.textContent = Number.isFinite(atr14) ? `${PAGE_CONFIG.currencySymbol}${formatPrice(atr14)}` : "-";

  renderLineChart(activeTabSymbol, historical?.points || []);
}

async function loadTabData(symbol, forceRefresh = false) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;

  const prev = tabStateBySymbol.get(normalized) || {};
  tabStateBySymbol.set(normalized, {
    ...prev,
    loading: true,
    error: "",
  });
  renderActiveTab();

  try {
    const overviewParams = new URLSearchParams();
    if (forceRefresh) overviewParams.set("refresh", "true");
    if (!PAGE_CONFIG.overviewIncludeIntraday) overviewParams.set("include_intraday", "false");
    if (!PAGE_CONFIG.overviewIncludeMarket) overviewParams.set("include_market", "false");
    if (!PAGE_CONFIG.overviewIncludeQqq) overviewParams.set("include_qqq", "false");
    const overviewUrl = overviewParams.size > 0
      ? `/api/security-overview/${encodeURIComponent(normalized)}?${overviewParams.toString()}`
      : `/api/security-overview/${encodeURIComponent(normalized)}`;
    const historicalUrl = forceRefresh
      ? `/api/historical/${encodeURIComponent(normalized)}?years=${HISTORICAL_LOAD_YEARS}&refresh=true`
      : `/api/historical/${encodeURIComponent(normalized)}?years=${HISTORICAL_LOAD_YEARS}`;

    const [overviewResp, historicalResp] = await Promise.all([
      fetchJson(overviewUrl),
      fetchJson(historicalUrl),
    ]);

    if (!overviewResp.response.ok) {
      throw new Error(overviewResp.result?.detail || "Failed to load security overview.");
    }
    if (!historicalResp.response.ok) {
      throw new Error(historicalResp.result?.detail || "Failed to load historical data.");
    }

    tabStateBySymbol.set(normalized, {
      loading: false,
      error: "",
      overview: overviewResp.result,
      historical: historicalResp.result,
      loadedAt: Date.now(),
    });
    upsertSymbolMeta(normalized, {
      name: overviewResp.result?.name,
      exchange: overviewResp.result?.exchange,
    });
    const overviewPrice = Number(overviewResp.result?.price?.current);
    if (Number.isFinite(overviewPrice) && overviewPrice > 0) {
      upsertLatestRow(normalized, {
        price: overviewPrice,
        timestamp: overviewResp.result?.price?.updated_at ?? null,
        source: overviewResp.result?.source ?? "security_overview",
      }, false);
      saveWatchlistRowsCache();
    }
    const existingInsight = symbolInsightsBySymbol.get(normalized) || {};
    const overviewPreviousClose = Number(overviewResp.result?.price?.previous_close);
    const overviewChangePct = Number(overviewResp.result?.price?.change_pct);
    symbolInsightsBySymbol.set(normalized, {
      ...existingInsight,
      current_price: Number.isFinite(overviewPrice) && overviewPrice > 0 ? overviewPrice : (existingInsight.current_price ?? null),
      reference_close: Number.isFinite(overviewPreviousClose) && overviewPreviousClose > 0
        ? overviewPreviousClose
        : (existingInsight.reference_close ?? null),
      change_pct: Number.isFinite(overviewChangePct) ? overviewChangePct : (existingInsight.change_pct ?? null),
    });
    saveSymbolInsightsCache();
    const loadedPoints = Array.isArray(historicalResp.result?.points) ? historicalResp.result.points : [];
    if (loadedPoints.length >= 2) {
      applyChartRangePreset(normalized, loadedPoints, "1y");
    }
    refreshWatchlist();
    renderTabs();
  } catch (error) {
    tabStateBySymbol.set(normalized, {
      ...prev,
      loading: false,
      error: error instanceof Error ? error.message : "Failed to load data.",
    });
  }

  if (activeTabSymbol === normalized) {
    renderActiveTab();
  }
}

function openSymbolTab(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;

  if (!openTabs.includes(normalized)) {
    openTabs = [...openTabs, normalized];
  }
  activeTabSymbol = normalized;
  renderActiveTab();

  const state = tabStateBySymbol.get(normalized);
  if (!state || (!state.overview && !state.loading)) {
    void loadTabData(normalized, false);
  }
}

function closeSymbolTab(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;

  if (chartPanState.symbol === normalized) {
    chartPanState.active = false;
    chartPanState.symbol = "";
    chartPanState.points = [];
    document.body.classList.remove("chart-panning");
  }
  openTabs = openTabs.filter((item) => item !== normalized);
  chartViewportBySymbol.delete(normalized);
  chartRangePresetBySymbol.delete(normalized);

  if (activeTabSymbol === normalized) {
    activeTabSymbol = openTabs.length ? openTabs[openTabs.length - 1] : "";
  }

  renderActiveTab();
}

function handleEvent(event) {
  const payload = JSON.parse(event.data);

  if (payload.type === "snapshot") {
    const rows = Array.isArray(payload.data?.rows) ? payload.data.rows : [];
    rows.forEach((row) => {
      const symbol = normalizeSymbol(row?.symbol);
      if (!symbol) return;
      upsertLatestRow(symbol, {
        price: row?.price ?? null,
        timestamp: row?.timestamp ?? null,
        source: row?.source ?? null,
      });
    });
    saveWatchlistRowsCache();
    const symbols = extractSymbolsFromPayload(payload.data);
    if (symbols.length === 0 && ignoreFirstEmptySymbolsEvent && selectedSymbols.length > 0) {
      ignoreFirstEmptySymbolsEvent = false;
      return;
    }
    ignoreFirstEmptySymbolsEvent = false;
    applySymbolsFromServer(symbols);
    setStatus(payload.data?.status || {});
    renderPortfolio();
    return;
  }

  if (payload.type === "status") {
    setStatus(payload.data || {});
    return;
  }

  if (payload.type === "symbols") {
    const symbols = extractSymbolsFromPayload(payload.data);
    const rows = Array.isArray(payload.data?.rows) ? payload.data.rows : [];
    rows.forEach((row) => {
      const symbol = normalizeSymbol(row?.symbol);
      if (!symbol) return;
      upsertLatestRow(symbol, {
        price: row?.price ?? null,
        timestamp: row?.timestamp ?? null,
        source: row?.source ?? null,
      });
    });
    saveWatchlistRowsCache();
    if (symbols.length === 0 && ignoreFirstEmptySymbolsEvent && selectedSymbols.length > 0) {
      ignoreFirstEmptySymbolsEvent = false;
      return;
    }
    ignoreFirstEmptySymbolsEvent = false;
    applySymbolsFromServer(symbols);
    renderPortfolio();
    return;
  }

  if (payload.type === "price") {
    const symbol = normalizeSymbol(payload.data?.symbol);
    if (!symbol) return;
    upsertLatestRow(symbol, {
      price: payload.data?.price ?? null,
      timestamp: payload.data?.timestamp ?? null,
      source: payload.data?.source ?? null,
    });
    saveWatchlistRowsCache();
    refreshWatchlist();
    renderPortfolio();
  }
}

function connectEventStream() {
  if (!PAGE_CONFIG.streamEnabled) {
    return;
  }
  if (eventSource) eventSource.close();
  eventSource = new EventSource("/api/stream");
  eventSource.onmessage = handleEvent;
  eventSource.onerror = () => {
    wsStateEl.textContent = "reconnecting";
  };
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
    if (!symbol || !Number.isFinite(quantity) || Math.abs(quantity) <= 0 || !Number.isFinite(avgCost) || avgCost <= 0) return;

    const latestPrice = Number(latestRowsBySymbol.get(symbol)?.price);
    const fallbackPrice = Number(item?.last_price);
    const lastPrice = Number.isFinite(latestPrice) && latestPrice > 0
      ? latestPrice
      : (Number.isFinite(fallbackPrice) && fallbackPrice > 0 ? fallbackPrice : null);

    const absQty = Math.abs(quantity);
    const rowCostBasis = absQty * avgCost;
    const rowMarketValue = Number.isFinite(lastPrice) ? quantity * lastPrice : null;
    const rowPnl = Number.isFinite(lastPrice)
      ? (quantity > 0 ? (lastPrice - avgCost) * absQty : (avgCost - lastPrice) * absQty)
      : null;
    const rowPnlPct = percentOf(rowPnl, rowCostBasis);

    if (Number.isFinite(rowMarketValue)) marketValue += rowMarketValue;
    costBasis += rowCostBasis;

    positions.push({
      symbol,
      quantity,
      avg_cost: avgCost,
      last_price: lastPrice,
      unrealized_pnl: rowPnl,
      unrealized_pnl_pct: rowPnlPct,
    });
  });

  const safeCash = Number.isFinite(cash) ? cash : 0;
  const safeInitialCash = Number.isFinite(initialCash) && initialCash > 0 ? initialCash : safeCash;
  const equity = safeCash + marketValue;
  const unrealizedPnl = marketValue - costBasis;
  const totalReturnPct = percentChange(equity, safeInitialCash);
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
    cash: safeCash,
    equity,
    unrealized_pnl: unrealizedPnl,
    total_return_pct: totalReturnPct,
    positions,
    recent_trades: recentTrades,
    initial_cash: safeInitialCash,
  };
}

function renderPortfolio() {
  if (!pfCashEl || !pfEquityEl || !pfUnrealizedPnlEl || !pfReturnEl || !pfPositionsBody || !pfTradesBody) return;
  const view = buildPortfolioView(portfolioBaseState);
  if (!view) return;

  pfCashEl.textContent = formatCurrency(view.cash);
  pfEquityEl.textContent = formatCurrency(view.equity);
  pfUnrealizedPnlEl.textContent = formatSignedCurrency(view.unrealized_pnl);
  pfUnrealizedPnlEl.classList.toggle("pf-positive", Number(view.unrealized_pnl) > 0);
  pfUnrealizedPnlEl.classList.toggle("pf-negative", Number(view.unrealized_pnl) < 0);
  pfReturnEl.textContent = formatSignedPercent(view.total_return_pct);
  pfReturnEl.classList.toggle("pf-positive", Number(view.total_return_pct) > 0);
  pfReturnEl.classList.toggle("pf-negative", Number(view.total_return_pct) < 0);

  pfPositionsBody.innerHTML = "";
  if (view.positions.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "pf-empty";
    td.textContent = "No positions";
    tr.appendChild(td);
    pfPositionsBody.appendChild(tr);
  } else {
    view.positions.forEach((item) => {
      const tr = document.createElement("tr");
      const pnl = Number(item.unrealized_pnl);
      const pnlPct = Number(item.unrealized_pnl_pct);
      if (Number.isFinite(pnl)) {
        tr.classList.toggle("pf-row-positive", pnl > 0);
        tr.classList.toggle("pf-row-negative", pnl < 0);
      }
      let unrealizedText = "-";
      if (Number.isFinite(pnl)) {
        unrealizedText = formatSignedCurrency(pnl);
        if (Number.isFinite(pnlPct)) {
          unrealizedText += ` (${formatSignedPercent(pnlPct)})`;
        }
      }
      const cells = [
        item.symbol,
        Number(item.quantity).toFixed(4).replace(/\.?0+$/, ""),
        Number.isFinite(item.avg_cost) ? formatCurrency(item.avg_cost) : "-",
        Number.isFinite(item.last_price) ? formatCurrency(item.last_price) : "-",
        unrealizedText,
      ];
      cells.forEach((text, idx) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (idx === 4) {
          td.classList.toggle("pf-positive", pnl > 0);
          td.classList.toggle("pf-negative", pnl < 0);
        }
        tr.appendChild(td);
      });
      pfPositionsBody.appendChild(tr);
    });
  }

  pfTradesBody.innerHTML = "";
  const trades = view.recent_trades;
  if (trades.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "pf-empty";
    td.textContent = "No trades";
    tr.appendChild(td);
    pfTradesBody.appendChild(tr);
  } else {
    trades.forEach((item) => {
      const tr = document.createElement("tr");
      const side = String(item?.side || "").toLowerCase();
      const cells = [
        formatDateTime(item?.timestamp),
        normalizeSymbol(item?.symbol),
        side || "-",
        Number.isFinite(Number(item?.price)) ? formatCurrency(Number(item.price)) : "-",
      ];
      cells.forEach((text, idx) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (idx === 2) {
          td.classList.add((side === "buy" || side === "cover") ? "pf-side-buy" : "pf-side-sell");
        }
        tr.appendChild(td);
      });
      pfTradesBody.appendChild(tr);
    });
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

watchlistEl.addEventListener("click", async (event) => {
  const removeBtn = event.target.closest(".watch-remove");
  if (removeBtn) {
    event.preventDefault();
    event.stopPropagation();
    await removeSymbol(removeBtn.dataset.symbol);
    renderDropdown();
    return;
  }

  const watchItem = event.target.closest(".watch-item");
  if (watchItem) {
    openSymbolTab(watchItem.dataset.symbol);
  }
});

watchlistEl.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const watchItem = event.target.closest(".watch-item");
  if (!watchItem) return;
  event.preventDefault();
  openSymbolTab(watchItem.dataset.symbol);
});

if (watchlistHeadEl) {
  watchlistHeadEl.addEventListener("click", (event) => {
    const button = event.target.closest(".watch-sort-btn");
    if (!button) return;
    const key = button.dataset.sortKey;
    if (!key) return;

    if (watchSortState.key === key) {
      watchSortState = {
        key,
        direction: watchSortState.direction === "asc" ? "desc" : "asc",
      };
    } else {
      watchSortState = {
        key,
        direction: key === "symbol" ? "asc" : "desc",
      };
    }
    refreshWatchlist();
  });
}

symbolTabsEl.addEventListener("click", (event) => {
  const close = event.target.closest(".tab-close");
  if (close) {
    event.stopPropagation();
    closeSymbolTab(close.dataset.closeSymbol);
    return;
  }
  const tab = event.target.closest(".symbol-tab");
  if (!tab) return;
  activeTabSymbol = tab.dataset.symbol;
  renderActiveTab();
});

symbolTabsEl.addEventListener("dragstart", (event) => {
  const target = event.target instanceof Element ? event.target : null;
  const tab = target ? target.closest(".symbol-tab") : null;
  if (!tab) return;
  draggingTabSymbol = tab.dataset.symbol || "";
  draggingTabPlacement = null;
  tab.classList.add("dragging");
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", draggingTabSymbol);
  }
});

symbolTabsEl.addEventListener("dragover", (event) => {
  if (!draggingTabSymbol) return;
  event.preventDefault();
  const placement = getDropPlacement(event.clientX, draggingTabSymbol);
  if (!placement) return;
  draggingTabPlacement = placement;
  clearTabDragVisualState();
  placement.tab.classList.add(placement.placeBefore ? "drag-over-left" : "drag-over-right");
  const draggingTab = symbolTabsEl.querySelector(`.symbol-tab[data-symbol="${draggingTabSymbol}"]`);
  if (draggingTab) draggingTab.classList.add("dragging");
});

symbolTabsEl.addEventListener("drop", (event) => {
  if (!draggingTabSymbol) return;
  event.preventDefault();
  const placement = draggingTabPlacement || getDropPlacement(event.clientX, draggingTabSymbol);
  if (placement) {
    const prevRects = captureTabRects();
    reorderOpenTabs(draggingTabSymbol, placement.targetSymbol, placement.placeBefore);
    renderTabs();
    animateTabReorder(prevRects);
  }
  clearTabDragVisualState();
  draggingTabSymbol = "";
  draggingTabPlacement = null;
});

symbolTabsEl.addEventListener("dragend", () => {
  clearTabDragVisualState();
  draggingTabSymbol = "";
  draggingTabPlacement = null;
});

window.addEventListener("mousemove", (event) => {
  if (!chartPanState.active || !chartPanState.symbol) return;
  const symbol = chartPanState.symbol;
  const points = chartPanState.points;
  const safe = (Array.isArray(points) ? points : [])
    .map((p) => ({ t: p?.t, c: Number(p?.c) }))
    .filter((p) => Number.isFinite(p.c));
  if (safe.length < 2) return;
  const svg = chartWrapEl.querySelector(".symbol-chart.interactive");
  if (!svg) return;

  const rect = svg.getBoundingClientRect();
  const drawableWidth = rect.width * ((CHART_WIDTH - (CHART_PAD_X * 2)) / CHART_WIDTH);
  const deltaX = event.clientX - chartPanState.lastClientX;
  chartPanState.lastClientX = event.clientX;

  const current = getChartViewport(symbol, safe.length);
  const span = Math.max(1, current.end - current.start);
  const deltaIndex = (deltaX / Math.max(1, drawableWidth)) * span;
  setChartViewport(symbol, current.start - deltaIndex, current.end - deltaIndex, safe.length);
  chartRangePresetBySymbol.set(symbol, "");
  renderLineChart(symbol, points);
});

window.addEventListener("mouseup", () => {
  if (!chartPanState.active) return;
  chartPanState.active = false;
  document.body.classList.remove("chart-panning");
  const svg = chartWrapEl.querySelector(".symbol-chart.interactive");
  if (svg) {
    svg.classList.remove("dragging");
  }
});

panelRefreshBtn.addEventListener("click", () => {
  if (!activeTabSymbol) return;
  void loadTabData(activeTabSymbol, true);
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
      window.alert(result.detail || result.note || "Failed to refresh credits");
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
      setPortfolioMessage(`Trade accepted: ${side.toUpperCase()} ${symbol}.`);
      if (pfPriceInput) pfPriceInput.value = "";
      if (pfQuantityInput) pfQuantityInput.value = "";

      if (!selectedSymbols.includes(symbol) && selectedSymbols.length < MAX_SYMBOLS) {
        selectedSymbols = [...selectedSymbols, symbol];
        await updateSymbolsOnServer();
      }
      openSymbolTab(symbol);
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

setCatalogMeta("Loading saved symbol catalog...");
if (!PAGE_CONFIG.watchlistCommentaryEnabled && watchLlmPanelEl) {
  watchLlmPanelEl.hidden = true;
}
if (!PAGE_CONFIG.streamEnabled && refreshCreditsBtn) {
  refreshCreditsBtn.hidden = true;
}
restoreSymbolInsightsCache();
restoreWatchlistRowsCache();
const restoredWatchSymbols = restoreWatchlistSymbolsCache();

async function hydrateInitialWatchlist() {
  try {
    const persistedWatchSymbols = await loadWatchlistStateFromServer();
    if (persistedWatchSymbols.length > 0) {
      applyInitialSelectedSymbols(persistedWatchSymbols, { primeIgnoreEmpty: PAGE_CONFIG.serverSymbolSync });
      if (PAGE_CONFIG.serverSymbolSync) {
        await loadSymbolInsights(persistedWatchSymbols, false);
        await updateSymbolsOnServer();
      } else {
        await loadSymbolInsights(persistedWatchSymbols, true);
      }
      return;
    }
  } catch (_error) {
    // fall through to local/browser fallback
  }

  if (restoredWatchSymbols.length > 0) {
    applyInitialSelectedSymbols(restoredWatchSymbols, { primeIgnoreEmpty: PAGE_CONFIG.serverSymbolSync });
    if (!PAGE_CONFIG.serverSymbolSync) {
      await persistWatchlistStateToServer(restoredWatchSymbols).catch(() => {
        // ignore startup sync errors
      });
      await loadSymbolInsights(restoredWatchSymbols, true);
      return;
    }
    await loadSymbolInsights(restoredWatchSymbols, false);
    await updateSymbolsOnServer().catch(() => {
      // ignore startup sync errors
    });
    return;
  }

  if (!PAGE_CONFIG.serverSymbolSync) {
    saveWatchlistSymbolsCache(selectedSymbols);
    if (selectedSymbols.length > 0) {
      await persistWatchlistStateToServer(selectedSymbols).catch(() => {
        // ignore startup sync errors
      });
      await loadSymbolInsights(selectedSymbols, true);
    }
    return;
  }

  try {
    const { response, result } = await fetchJson("/api/snapshot");
    if (!response.ok || result?.type !== "snapshot") return;
    const rows = Array.isArray(result?.data?.rows) ? result.data.rows : [];
    rows.forEach((row) => {
      const symbol = normalizeSymbol(row?.symbol);
      if (!symbol) return;
      upsertLatestRow(symbol, {
        price: row?.price ?? null,
        timestamp: row?.timestamp ?? null,
        source: row?.source ?? null,
      });
    });
    saveWatchlistRowsCache();
    const symbols = extractSymbolsFromPayload(result?.data);
    if (symbols.length > 0) {
      applySymbolsFromServer(symbols);
      void loadSymbolInsights(symbols, false);
    }
    setStatus(result?.data?.status || {});
  } catch (_error) {
    // ignore startup sync errors
  }
}
void bootstrapSymbolCatalog();
if (!PAGE_CONFIG.streamEnabled) {
  setStatus({
    mode: PAGE_CONFIG.staticMode,
    provider: PAGE_CONFIG.staticProvider,
    ws_connected: false,
    fallback_poll_interval_sec: PAGE_CONFIG.staticFallbackLabel,
    open_symbols: [],
    daily_credits_left: null,
    daily_credits_limit: null,
    daily_credits_updated_at: null,
    daily_credits_is_estimated: false,
  });
}
void hydrateInitialWatchlist().finally(() => {
  connectEventStream();
  if (PAGE_CONFIG.serverSymbolSync) {
    window.setTimeout(() => {
      if (selectedSymbols.length > 0) return;
      void hydrateInitialWatchlist();
    }, 1500);
  }
});
startMarketClock();
loadPortfolio();
enableContextualScrollbar(watchlistEl);
enableContextualScrollbar(centerPaneEl);
renderActiveTab();

// ウォッチリスト更新ボタン
async function refreshWatchlistData() {
  const btn = document.getElementById("watchlist-refresh");
  if (!btn || btn.disabled) return;

  const symbols = uniqueSymbols(selectedSymbols);
  if (symbols.length === 0) return;

  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = "更新中…";

  try {
    const refreshed = new Set();
    const applyItems = (items) => {
      if (!Array.isArray(items)) return;
      items.forEach((item) => {
        const symbol = normalizeSymbol(item?.symbol);
        if (!symbol) return;
        refreshed.add(symbol);
        const latestClose = Number(item?.latest_close);
        const previousClose = Number(item?.previous_close);
        const currentPrice = Number(item?.current_price);
        const referenceClose = Number(item?.reference_close);
        const changePct = Number(item?.change_pct);
        const trend = (Array.isArray(item?.trend_30d) ? item.trend_30d : [])
          .map((point) => Number(point))
          .filter((num) => Number.isFinite(num));

        symbolInsightsBySymbol.set(symbol, {
          current_price: Number.isFinite(currentPrice) && currentPrice > 0 ? currentPrice : null,
          latest_close: Number.isFinite(latestClose) ? latestClose : null,
          previous_close: Number.isFinite(previousClose) ? previousClose : null,
          reference_close: Number.isFinite(referenceClose) && referenceClose > 0 ? referenceClose : null,
          change_pct: Number.isFinite(changePct) ? changePct : null,
          trend_30d: trend,
        });
        saveSymbolInsightsCache();

        // EODモードでは終値、通常モードではquote現在値を優先する。
        const existingRow = latestRowsBySymbol.get(symbol);
        const hasValidPrice = existingRow && Number(existingRow.price) > 0;
        const fallbackPrice = (Number.isFinite(currentPrice) && currentPrice > 0) ? currentPrice : latestClose;
        const hasCurrentPrice = Number.isFinite(currentPrice) && currentPrice > 0;
        const resolvedSource = resolveSparklinePriceSource(item, hasCurrentPrice);
        const isEodClose = resolvedSource === "eod_close";
        const marketOpen = isConfiguredMarketOpenNow();
        const shouldReplaceCachedPrice = (
          isEodClose
            ? (!hasValidPrice || !marketOpen || !PAGE_CONFIG.streamEnabled)
            : (!hasValidPrice || !PAGE_CONFIG.streamEnabled || hasCurrentPrice)
        );
        if (shouldReplaceCachedPrice && Number.isFinite(fallbackPrice) && fallbackPrice > 0) {
          latestRowsBySymbol.set(symbol, {
            ...(existingRow ?? {}),
            symbol,
            price: fallbackPrice,
            timestamp: item.updated_at ?? item.latest_close_date ?? null,
            source: resolvedSource,
          });
          saveWatchlistRowsCache();
        }
      });
    };

    // まずは一括再取得。取りこぼした銘柄のみ個別再取得する。
    const batchParams = applySparklineModeParams(new URLSearchParams({ symbols: symbols.join(","), refresh: "true" }));
    const batch = await fetchJson(`/api/sparkline?${batchParams.toString()}`);
    if (batch.response.ok) {
      applyItems(batch.result?.items);
    }

    const missing = symbols.filter((symbol) => !refreshed.has(symbol));
    for (const symbol of missing) {
      const params = applySparklineModeParams(new URLSearchParams({ symbols: symbol, refresh: "true" }));
      const single = await fetchJson(`/api/sparkline?${params.toString()}`);
      if (!single.response.ok) continue;
      applyItems(single.result?.items);
    }
  } catch (_err) {
    // ネットワークエラー等 - 失敗しても UI は壊さない
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }

  refreshWatchlist();
  await refreshWatchlistCommentary(true);
}

document.getElementById("watchlist-refresh")?.addEventListener("click", () => {
  void refreshWatchlistData();
});

watchLlmRefreshBtn?.addEventListener("click", () => {
  void refreshWatchlistCommentary(true);
});
