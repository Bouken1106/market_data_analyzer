const MAX_SYMBOLS = 8;
const MAX_DROPDOWN_ITEMS = 100;
const CHART_WIDTH = 960;
const CHART_HEIGHT = 300;
const CHART_PAD_X = 58;
const CHART_PAD_TOP = 18;
const CHART_PAD_BOTTOM = 30;

const noticeEl = document.getElementById("mdl-global-notice");
const guideCardEl = document.getElementById("mdl-guide-card");
const guideCloseBtn = document.getElementById("mdl-guide-close");
const guideReopenBtn = document.getElementById("mdl-guide-reopen");
const guideReopenWrapEl = document.getElementById("mdl-guide-reopen-wrap");

const refreshAllBtn = document.getElementById("mdl-refresh-all");
const refreshCreditsBtn = document.getElementById("mdl-refresh-credits");
const refreshWatchBtn = document.getElementById("mdl-refresh-watch");
const refreshDetailBtn = document.getElementById("mdl-refresh-detail");

const bannerProviderEl = document.getElementById("mdl-banner-provider");
const bannerCountEl = document.getElementById("mdl-banner-count");
const bannerPollingEl = document.getElementById("mdl-banner-polling");
const bannerUnsupportedEl = document.getElementById("mdl-banner-unsupported");

const providerStatusEl = document.getElementById("mdl-provider-status");
const providerConfigEl = document.getElementById("mdl-provider-config");
const updateModeEl = document.getElementById("mdl-update-mode");
const updateNoteEl = document.getElementById("mdl-update-note");
const cacheStateEl = document.getElementById("mdl-cache-state");
const cacheNoteEl = document.getElementById("mdl-cache-note");
const creditsStateEl = document.getElementById("mdl-credits-state");
const creditsUpdatedEl = document.getElementById("mdl-credits-updated");
const fallbackStateEl = document.getElementById("mdl-fallback-state");
const fallbackNoteEl = document.getElementById("mdl-fallback-note");

const searchInputEl = document.getElementById("mdl-search-input");
const searchDropdownEl = document.getElementById("mdl-search-dropdown");
const catalogMetaEl = document.getElementById("mdl-catalog-meta");
const searchErrorEl = document.getElementById("mdl-search-error");

const watchCountPillEl = document.getElementById("mdl-watch-count");
const watchlistEl = document.getElementById("mdl-watchlist");
const watchEmptyEl = document.getElementById("mdl-watch-empty");

const unsupportedListEl = document.getElementById("mdl-unsupported-list");

const detailSubtitleEl = document.getElementById("mdl-detail-subtitle");
const detailEmptyEl = document.getElementById("mdl-detail-empty");
const detailContentEl = document.getElementById("mdl-detail-content");
const detailSymbolEl = document.getElementById("mdl-detail-symbol");
const detailExchangeEl = document.getElementById("mdl-detail-exchange");
const detailNameEl = document.getElementById("mdl-detail-name");
const detailPriceEl = document.getElementById("mdl-detail-price");
const detailChangeEl = document.getElementById("mdl-detail-change");
const detailUpdatedEl = document.getElementById("mdl-detail-updated");

const kpiChangeEl = document.getElementById("mdl-kpi-change");
const kpiRangeEl = document.getElementById("mdl-kpi-range");
const kpiVolumeEl = document.getElementById("mdl-kpi-volume");
const kpiTurnoverEl = document.getElementById("mdl-kpi-turnover");
const kpiVwapEl = document.getElementById("mdl-kpi-vwap");
const kpiMaEl = document.getElementById("mdl-kpi-ma");
const kpiAtrEl = document.getElementById("mdl-kpi-atr");
const kpiGapEl = document.getElementById("mdl-kpi-gap");
const kpiBetaEl = document.getElementById("mdl-kpi-beta");
const kpiSourceEl = document.getElementById("mdl-kpi-source");

const chartMetaEl = document.getElementById("mdl-chart-meta");
const chartEl = document.getElementById("mdl-chart");
const supportStatusEl = document.getElementById("mdl-support-status");
const marketContextEl = document.getElementById("mdl-market-context");

const intervalButtons = Array.from(document.querySelectorAll(".mdl-interval-btn"));
const rangeButtons = Array.from(document.querySelectorAll(".mdl-range-btn"));
const sortButtons = Array.from(document.querySelectorAll(".mdl-watch-head .watch-sort-btn"));

const state = {
  bootstrap: null,
  status: null,
  helpTexts: {},
  catalog: [],
  watchlist: [],
  watchRows: new Map(),
  details: new Map(),
  selectedSymbol: "",
  chartInterval: "1day",
  chartRange: "1M",
  sort: { key: "symbol", direction: "asc" },
  pollIntervalSec: 60,
  pollTimer: 0,
  quotesLoading: false,
  detailLoading: false,
  persistInFlight: false,
  persistQueued: false,
};

function normalizeSymbol(raw) {
  return String(raw || "").trim().toUpperCase();
}

function uniqueSymbols(values) {
  const out = [];
  const seen = new Set();
  const source = Array.isArray(values) ? values : String(values || "").split(",");
  source.forEach((value) => {
    const symbol = normalizeSymbol(value);
    if (!symbol) return;
    if (!/^[A-Z0-9.\-]{1,15}$/.test(symbol)) return;
    if (seen.has(symbol)) return;
    seen.add(symbol);
    out.push(symbol);
  });
  return out;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json().catch(() => ({}));
  return { response, result };
}

function formatPrice(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
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

function formatCompact(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(num);
}

function formatMaybeCurrency(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  if (Math.abs(num) >= 1_000_000) {
    return `$${formatCompact(num)}`;
  }
  return `$${formatPrice(num)}`;
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

function closeParamHelpPopovers(exceptDetail = null) {
  const openDetails = document.querySelectorAll("details.param-help[open]");
  openDetails.forEach((detail) => {
    if (detail === exceptDetail) return;
    detail.removeAttribute("open");
  });
}

function setSearchError(message) {
  if (!searchErrorEl) return;
  searchErrorEl.textContent = message || "";
}

function setCatalogMeta(message) {
  if (!catalogMetaEl) return;
  catalogMetaEl.textContent = message || "";
}

function setNotice(message = "", level = "info") {
  if (!noticeEl) return;
  noticeEl.textContent = message || "";
  noticeEl.classList.toggle("is-hidden", !message);
  noticeEl.classList.toggle("is-info", Boolean(message) && level === "info");
  noticeEl.classList.toggle("is-warn", Boolean(message) && level === "warn");
  noticeEl.classList.toggle("is-error", Boolean(message) && level === "error");
}

function fillHelpTexts() {
  document.querySelectorAll(".mdl-help").forEach((element) => {
    const key = element.getAttribute("data-help-key") || "";
    const text = state.helpTexts[key] || "説明はまだ用意されていません。";
    const paragraph = element.querySelector("p");
    if (paragraph) paragraph.textContent = text;
  });
}

function renderGuide(isVisible) {
  const show = Boolean(isVisible);
  if (guideCardEl) guideCardEl.classList.toggle("hidden", !show);
  if (guideReopenWrapEl) guideReopenWrapEl.classList.toggle("hidden", show);
}

function renderUnsupportedList() {
  if (!unsupportedListEl) return;
  const items = Array.isArray(state.bootstrap?.unsupported_features) ? state.bootstrap.unsupported_features : [];
  unsupportedListEl.innerHTML = "";
  if (items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "未対応項目はありません。";
    unsupportedListEl.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${escapeHtml(item?.label || "-")}</strong><span>${escapeHtml(item?.reason || "-")}</span>`;
    unsupportedListEl.appendChild(li);
  });
}

function renderWatchCount() {
  const text = `${state.watchlist.length} / ${MAX_SYMBOLS}`;
  if (watchCountPillEl) watchCountPillEl.textContent = text;
  if (bannerCountEl) bannerCountEl.textContent = text;
}

function renderBannerAndStatus() {
  const bootstrap = state.bootstrap || {};
  const providerLabel = bootstrap.provider_mode_label || "-";
  const configured = bootstrap.configured_sources || {};
  const fmpConfigured = configured.fmp ? "FMP" : null;
  const tdConfigured = configured.twelvedata ? "Twelve Data" : null;
  const configLabels = [fmpConfigured, tdConfigured].filter((item) => item).join(" / ");

  if (bannerProviderEl) bannerProviderEl.textContent = providerLabel;
  if (bannerPollingEl) bannerPollingEl.textContent = `${Number(state.pollIntervalSec || 0) || "-"} sec`;
  if (bannerUnsupportedEl) {
    const unsupportedCount = Array.isArray(bootstrap.unsupported_features) ? bootstrap.unsupported_features.length : 0;
    bannerUnsupportedEl.textContent = unsupportedCount > 0 ? `${unsupportedCount} items` : "0 items";
  }

  if (providerStatusEl) providerStatusEl.textContent = providerLabel;
  if (providerConfigEl) providerConfigEl.textContent = configLabels || "API 設定の詳細は bootstrap 情報待ち";
  if (updateModeEl) updateModeEl.textContent = bootstrap.update_mode_label || "REST polling";
  if (updateNoteEl) updateNoteEl.textContent = `目安 ${Number(state.pollIntervalSec || 0) || "-"} sec / ${state.watchlist.length || 1} symbols`;

  const selectedDetail = state.details.get(state.selectedSymbol);
  const source = selectedDetail?.overview?.source || "";
  if (cacheStateEl) cacheStateEl.textContent = source === "cache" ? "Cache hit" : "Enabled";
  if (cacheNoteEl) {
    cacheNoteEl.textContent = state.selectedSymbol && source
      ? `${state.selectedSymbol}: ${source}`
      : "symbol catalog と詳細 API のキャッシュを再利用します。";
  }

  const creditsLeft = state.status?.daily_credits_left;
  const creditsLimit = state.status?.daily_credits_limit;
  const estimated = Boolean(state.status?.daily_credits_is_estimated);
  if (creditsStateEl) {
    if (creditsLeft === null || creditsLeft === undefined) {
      creditsStateEl.textContent = "-";
    } else if (creditsLimit === null || creditsLimit === undefined) {
      creditsStateEl.textContent = `${creditsLeft}${estimated ? " (est)" : ""}`;
    } else {
      creditsStateEl.textContent = `${creditsLeft} / ${creditsLimit}${estimated ? " (est)" : ""}`;
    }
  }
  if (creditsUpdatedEl) {
    creditsUpdatedEl.textContent = `updated ${formatDateTime(state.status?.daily_credits_updated_at)}`;
  }

  const supportsWebsocket = Boolean(bootstrap.supports_websocket);
  if (fallbackStateEl) fallbackStateEl.textContent = supportsWebsocket ? "REST first / WS capable" : "REST only";
  if (fallbackNoteEl) {
    fallbackNoteEl.textContent = supportsWebsocket
      ? "このページは独立ウォッチリスト維持のため REST 定期更新を優先します。"
      : "リアルタイム未対応時も定期更新で画面を継続します。";
  }

  renderWatchCount();
}

function pickCatalogCandidates(query) {
  const needle = String(query || "").trim().toLowerCase();
  if (state.catalog.length === 0) return [];
  if (!needle) {
    return state.catalog
      .filter((item) => !state.watchlist.includes(item.symbol))
      .slice(0, MAX_DROPDOWN_ITEMS);
  }

  const direct = [];
  const partial = [];
  state.catalog.forEach((item) => {
    if (state.watchlist.includes(item.symbol)) return;
    const symbol = String(item.symbol || "").toLowerCase();
    const name = String(item.name || "").toLowerCase();
    if (symbol.startsWith(needle)) {
      direct.push(item);
      return;
    }
    if (name.includes(needle)) {
      partial.push(item);
    }
  });
  return [...direct, ...partial].slice(0, MAX_DROPDOWN_ITEMS);
}

function showDropdown() {
  if (searchDropdownEl) searchDropdownEl.classList.remove("hidden");
}

function hideDropdown() {
  if (searchDropdownEl) searchDropdownEl.classList.add("hidden");
}

function renderDropdown() {
  if (!searchDropdownEl) return;
  searchDropdownEl.innerHTML = "";
  const candidates = pickCatalogCandidates(searchInputEl?.value || "");
  if (state.catalog.length === 0) {
    const row = document.createElement("div");
    row.className = "dropdown-empty";
    row.textContent = "Symbol catalog is not loaded yet.";
    searchDropdownEl.appendChild(row);
    showDropdown();
    return;
  }
  if (candidates.length === 0) {
    const row = document.createElement("div");
    row.className = "dropdown-empty";
    row.textContent = "No matching symbols";
    searchDropdownEl.appendChild(row);
    showDropdown();
    return;
  }

  const providerLabel = state.bootstrap?.provider_mode_label || "Catalog";
  candidates.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dropdown-item";
    btn.dataset.symbol = item.symbol;
    btn.innerHTML = `
      <span>${escapeHtml(item.symbol)} | ${escapeHtml(item.name || "Company name unavailable")}</span>
      <small>${escapeHtml(item.exchange || "-")} / United States / ${escapeHtml(providerLabel)}</small>
    `;
    searchDropdownEl.appendChild(btn);
  });
  showDropdown();
}

function compareValues(left, right, direction) {
  const leftMissing = left === null || left === undefined || Number.isNaN(left);
  const rightMissing = right === null || right === undefined || Number.isNaN(right);
  if (leftMissing && rightMissing) return 0;
  if (leftMissing) return 1;
  if (rightMissing) return -1;

  if (typeof left === "string" || typeof right === "string") {
    const result = String(left).localeCompare(String(right), "en", { sensitivity: "base" });
    return direction === "asc" ? result : -result;
  }
  const diff = Number(left) - Number(right);
  if (diff === 0) return 0;
  return direction === "asc" ? diff : -diff;
}

function sortedWatchlist() {
  const symbols = [...state.watchlist];
  const { key, direction } = state.sort;
  if (key === "symbol") {
    return symbols.sort((left, right) => compareValues(left, right, direction));
  }
  return symbols.sort((left, right) => {
    const leftRow = state.watchRows.get(left);
    const rightRow = state.watchRows.get(right);
    const leftValue = key === "price" ? Number(leftRow?.price) : Number(leftRow?.change_pct);
    const rightValue = key === "price" ? Number(rightRow?.price) : Number(rightRow?.change_pct);
    const primary = compareValues(leftValue, rightValue, direction);
    if (primary !== 0) return primary;
    return left.localeCompare(right, "en", { sensitivity: "base" });
  });
}

function updateSortButtons() {
  sortButtons.forEach((button) => {
    const key = button.getAttribute("data-sort-key") || "";
    const active = key === state.sort.key;
    const baseLabel = key === "symbol" ? "銘柄" : (key === "price" ? "現在値" : "前日比%");
    button.classList.toggle("active", active);
    button.textContent = active
      ? `${baseLabel} ${state.sort.direction === "asc" ? "↑" : "↓"}`
      : baseLabel;
  });
}

function renderWatchlist() {
  if (!watchlistEl || !watchEmptyEl) return;
  updateSortButtons();
  watchlistEl.innerHTML = "";
  const symbols = sortedWatchlist();
  watchEmptyEl.classList.toggle("hidden", symbols.length > 0);

  symbols.forEach((symbol) => {
    const row = state.watchRows.get(symbol) || {};
    const price = Number(row?.price);
    const changePct = Number(row?.change_pct);
    const article = document.createElement("article");
    article.className = "mdl-watch-row";
    article.classList.toggle("active", symbol === state.selectedSymbol);
    article.innerHTML = `
      <button type="button" class="mdl-watch-select" data-select-symbol="${escapeHtml(symbol)}">
        <span class="mdl-watch-symbol">${escapeHtml(symbol)}</span>
        <span class="mdl-watch-price">${Number.isFinite(price) ? `$${formatPrice(price)}` : "-"}</span>
        <span class="mdl-watch-change ${changePct > 0 ? "up" : (changePct < 0 ? "down" : "")}">
          ${Number.isFinite(changePct) ? formatSignedPercent(changePct) : "-"}
        </span>
        <span class="mdl-watch-updated">${formatTime(row?.updated_at)}</span>
        <span class="mdl-watch-source">${escapeHtml(row?.source || "-")}</span>
      </button>
      <button type="button" class="watch-remove" data-remove-symbol="${escapeHtml(symbol)}" aria-label="remove">×</button>
    `;
    watchlistEl.appendChild(article);
  });
}

async function persistState() {
  if (state.persistInFlight) {
    state.persistQueued = true;
    return;
  }
  state.persistInFlight = true;
  try {
    do {
      state.persistQueued = false;
      const payload = {
        watchlist_symbols: state.watchlist.join(","),
        last_viewed_symbol: state.selectedSymbol,
        chart_interval: state.chartInterval,
      };
      const { response, result } = await fetchJson("/api/market-data-lab/state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        setNotice(result.detail || "状態保存に失敗しました。ページを更新して再度お試しください。", "warn");
        break;
      }
      state.pollIntervalSec = Number(result?.recommended_poll_interval_sec) || state.pollIntervalSec;
      renderBannerAndStatus();
      schedulePolling();
    } while (state.persistQueued);
  } finally {
    state.persistInFlight = false;
  }
}

async function setOnboardingDismissed(dismissed) {
  const { response, result } = await fetchJson("/api/market-data-lab/onboarding", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dismissed: Boolean(dismissed) }),
  });
  if (!response.ok) {
    setNotice(result.detail || "ガイド状態の保存に失敗しました。", "warn");
    return;
  }
  renderGuide(Boolean(result.enabled));
}

async function loadCatalog(refresh = false) {
  setCatalogMeta(refresh ? "Refreshing symbol catalog..." : "Loading symbol catalog...");
  try {
    const url = refresh ? "/api/symbol-catalog?refresh=true" : "/api/symbol-catalog";
    const { response, result } = await fetchJson(url);
    if (!response.ok) {
      setCatalogMeta(result.detail || "Failed to load symbol catalog");
      return;
    }
    const rows = Array.isArray(result?.symbols) ? result.symbols : [];
    state.catalog = rows.map((item) => ({
      symbol: normalizeSymbol(item?.symbol),
      name: String(item?.name || "").trim(),
      exchange: String(item?.exchange || "").trim(),
    })).filter((item) => item.symbol);
    setCatalogMeta(`${state.catalog.length.toLocaleString()} symbols loaded (${result.source || "unknown"})`);
    if (document.activeElement === searchInputEl) {
      renderDropdown();
    }
  } catch (_error) {
    setCatalogMeta("Failed to load symbol catalog");
  }
}

async function refreshCredits(refresh = false) {
  try {
    const { response, result } = await fetchJson(refresh ? "/api/credits?refresh=true" : "/api/credits");
    if (!response.ok) {
      setNotice(result.detail || "クレジット情報の取得に失敗しました。", "warn");
      return;
    }
    state.status = result?.status || state.status;
    renderBannerAndStatus();
  } catch (_error) {
    setNotice("クレジット情報の取得に失敗しました。少し待ってから再試行してください。", "warn");
  }
}

async function refreshQuotes(showMessage = false) {
  if (state.watchlist.length === 0) {
    renderWatchlist();
    return;
  }
  if (state.quotesLoading) return;
  state.quotesLoading = true;
  if (refreshWatchBtn) refreshWatchBtn.disabled = true;
  try {
    const params = new URLSearchParams({ symbols: state.watchlist.join(",") });
    const { response, result } = await fetchJson(`/api/market-data-lab/quotes?${params.toString()}`);
    if (!response.ok) {
      setNotice(result.detail || "ウォッチリスト更新に失敗しました。監視銘柄を減らすか、時間を空けて再試行してください。", "warn");
      return;
    }
    const items = Array.isArray(result?.items) ? result.items : [];
    items.forEach((item) => {
      const symbol = normalizeSymbol(item?.symbol);
      if (!symbol) return;
      state.watchRows.set(symbol, {
        symbol,
        name: String(item?.name || "").trim(),
        exchange: String(item?.exchange || "").trim(),
        price: Number(item?.price),
        change_abs: Number(item?.change_abs),
        change_pct: Number(item?.change_pct),
        updated_at: item?.updated_at || null,
        source: item?.source || null,
      });
    });
    renderWatchlist();
    if (showMessage) {
      setNotice(`ウォッチリストを更新しました。(${state.watchlist.length} symbols)`, "info");
    }
  } catch (_error) {
    setNotice("ウォッチリスト更新に失敗しました。通信状態を確認して再試行してください。", "warn");
  } finally {
    state.quotesLoading = false;
    if (refreshWatchBtn) refreshWatchBtn.disabled = false;
  }
}

function normalizeChartPoints(points) {
  return (Array.isArray(points) ? points : [])
    .map((item) => ({
      t: item?.t,
      c: Number(item?.c),
      v: Number(item?.v),
    }))
    .filter((item) => item.t && Number.isFinite(item.c));
}

function sliceDailyPoints(points, range) {
  const safe = normalizeChartPoints(points);
  if (range === "MAX") return safe;
  const sizeByRange = {
    "1D": 2,
    "5D": 5,
    "1M": 22,
    "6M": 132,
    "1Y": 252,
  };
  const size = sizeByRange[range] || 22;
  return safe.slice(-size);
}

function resolveChartView(overview) {
  const charts = overview?.charts || {};
  const requestedInterval = state.chartInterval;
  const requestedRange = state.chartRange;
  if (requestedRange === "1D") {
    if (requestedInterval === "1min") {
      const intraday1m = normalizeChartPoints(charts["1min"]);
      if (intraday1m.length >= 2) {
        return { points: intraday1m, actualInterval: "1min", note: "" };
      }
    }
    if (requestedInterval === "5min") {
      const intraday5m = normalizeChartPoints(charts["5min"]);
      if (intraday5m.length >= 2) {
        return { points: intraday5m, actualInterval: "5min", note: "" };
      }
    }
  }

  const dailyPoints = sliceDailyPoints(charts["1day"], requestedRange);
  const note = requestedInterval !== "1day" && requestedRange !== "1D"
    ? "1分足 / 5分足は 1D レンジのみ対応のため日足へフォールバックしました。"
    : "";
  return { points: dailyPoints, actualInterval: "1day", note };
}

function buildChartSvg(points) {
  const safe = normalizeChartPoints(points);
  if (safe.length < 2) {
    return '<div class="chart-empty">チャートデータが不足しています。</div>';
  }

  let min = Math.min(...safe.map((item) => item.c));
  let max = Math.max(...safe.map((item) => item.c));
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const yRange = max - min || 1;
  const paddedMin = min - (yRange * 0.08);
  const paddedMax = max + (yRange * 0.08);
  const fullRange = paddedMax - paddedMin || 1;
  const drawableWidth = CHART_WIDTH - (CHART_PAD_X * 2);
  const drawableHeight = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;
  const xForIndex = (index) => CHART_PAD_X + ((index / Math.max(1, safe.length - 1)) * drawableWidth);
  const yForPrice = (price) => CHART_PAD_TOP + (1 - ((price - paddedMin) / fullRange)) * drawableHeight;

  const polyline = safe.map((item, index) => `${xForIndex(index).toFixed(2)},${yForPrice(item.c).toFixed(2)}`).join(" ");
  const area = [
    `${CHART_PAD_X},${CHART_HEIGHT - CHART_PAD_BOTTOM}`,
    ...safe.map((item, index) => `${xForIndex(index).toFixed(2)},${yForPrice(item.c).toFixed(2)}`),
    `${CHART_WIDTH - CHART_PAD_X},${CHART_HEIGHT - CHART_PAD_BOTTOM}`,
  ].join(" ");

  const yTicks = Array.from({ length: 4 }, (_, index) => {
    const ratio = index / 3;
    const y = CHART_PAD_TOP + (ratio * drawableHeight);
    const value = paddedMax - (fullRange * ratio);
    return `
      <line x1="${CHART_PAD_X}" y1="${y.toFixed(2)}" x2="${(CHART_WIDTH - CHART_PAD_X).toFixed(2)}" y2="${y.toFixed(2)}" class="mdl-chart-grid"></line>
      <text x="${(CHART_PAD_X - 8).toFixed(2)}" y="${(y + 4).toFixed(2)}" class="mdl-chart-axis" text-anchor="end">${formatPrice(value)}</text>
    `;
  }).join("");

  const xTicks = Array.from({ length: Math.min(5, safe.length) }, (_, index) => {
    const ratio = (Math.min(4, safe.length - 1) === 0) ? 0 : index / Math.min(4, safe.length - 1);
    const pointIndex = Math.round(ratio * (safe.length - 1));
    const x = xForIndex(pointIndex);
    const label = String(safe[pointIndex]?.t || "").split(" ")[0];
    return `
      <line x1="${x.toFixed(2)}" y1="${(CHART_HEIGHT - CHART_PAD_BOTTOM).toFixed(2)}" x2="${x.toFixed(2)}" y2="${(CHART_HEIGHT - CHART_PAD_BOTTOM + 6).toFixed(2)}" class="mdl-chart-grid"></line>
      <text x="${x.toFixed(2)}" y="${(CHART_HEIGHT - 8).toFixed(2)}" class="mdl-chart-axis" text-anchor="middle">${escapeHtml(label)}</text>
    `;
  }).join("");

  const last = safe[safe.length - 1];
  const lastX = xForIndex(safe.length - 1);
  const lastY = yForPrice(last.c);
  return `
    <svg class="mdl-chart-svg" viewBox="0 0 ${CHART_WIDTH} ${CHART_HEIGHT}" preserveAspectRatio="none" role="img" aria-label="price chart">
      <defs>
        <linearGradient id="mdl-chart-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="rgba(255, 198, 109, 0.34)"></stop>
          <stop offset="100%" stop-color="rgba(255, 198, 109, 0.02)"></stop>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="${CHART_WIDTH}" height="${CHART_HEIGHT}" class="mdl-chart-bg"></rect>
      ${yTicks}
      ${xTicks}
      <polyline points="${area}" class="mdl-chart-area"></polyline>
      <polyline points="${polyline}" class="mdl-chart-line"></polyline>
      <circle cx="${lastX.toFixed(2)}" cy="${lastY.toFixed(2)}" r="4.2" class="mdl-chart-point"></circle>
    </svg>
  `;
}

function renderChart(overview) {
  if (!chartEl || !chartMetaEl) return;
  const view = resolveChartView(overview);
  chartEl.innerHTML = buildChartSvg(view.points);
  const points = normalizeChartPoints(view.points);
  const fromLabel = points[0]?.t || "-";
  const toLabel = points[points.length - 1]?.t || "-";
  chartMetaEl.textContent = `${view.actualInterval} / ${points.length} points / ${fromLabel} -> ${toLabel}${view.note ? ` / ${view.note}` : ""}`;

  intervalButtons.forEach((button) => {
    const interval = button.getAttribute("data-interval") || "";
    button.classList.toggle("active", interval === state.chartInterval);
  });
  rangeButtons.forEach((button) => {
    const range = button.getAttribute("data-range") || "";
    button.classList.toggle("active", range === state.chartRange);
  });
}

function renderSupportStatus(overview) {
  if (!supportStatusEl) return;
  const items = overview?.support_status || {};
  const fallbackItems = Array.isArray(state.bootstrap?.unsupported_features) ? state.bootstrap.unsupported_features : [];
  const fragments = fallbackItems.map((item) => {
    const key = String(item?.key || "");
    const status = items[key] ? "未対応" : "未対応";
    const reason = item?.reason || "無料モードでは未対応です。";
    return `
      <div class="mdl-support-item">
        <span>${escapeHtml(item?.label || key || "-")}</span>
        <strong>${status}</strong>
        <p>${escapeHtml(reason)}</p>
      </div>
    `;
  });
  supportStatusEl.innerHTML = fragments.join("");
}

function renderMarketContext(overview) {
  if (!marketContextEl) return;
  const market = overview?.market || {};
  const spy = market?.sp500_proxy || {};
  const qqq = market?.nasdaq_proxy || {};
  const rows = [
    {
      label: "SPY",
      value: Number.isFinite(Number(spy?.price)) ? `$${formatPrice(spy.price)}` : "未対応",
      sub: Number.isFinite(Number(spy?.change_pct)) ? formatSignedPercent(spy.change_pct) : "-",
    },
    {
      label: "QQQ",
      value: Number.isFinite(Number(qqq?.price)) ? `$${formatPrice(qqq.price)}` : "未対応",
      sub: Number.isFinite(Number(qqq?.change_pct)) ? formatSignedPercent(qqq.change_pct) : "-",
    },
    {
      label: "Beta vs SPY",
      value: Number.isFinite(Number(market?.beta_60d_vs_spy)) ? Number(market.beta_60d_vs_spy).toFixed(2) : "未対応",
      sub: "60d",
    },
    {
      label: "Corr vs SPY",
      value: Number.isFinite(Number(market?.corr_60d_vs_spy)) ? Number(market.corr_60d_vs_spy).toFixed(2) : "未対応",
      sub: "60d",
    },
  ];
  marketContextEl.innerHTML = rows.map((row) => `
    <div class="mdl-support-item">
      <span>${escapeHtml(row.label)}</span>
      <strong>${escapeHtml(row.value)}</strong>
      <p>${escapeHtml(row.sub)}</p>
    </div>
  `).join("");
}

function renderDetail() {
  const symbol = state.selectedSymbol;
  if (!symbol) {
    detailContentEl.classList.add("hidden");
    detailEmptyEl.classList.remove("hidden");
    detailSubtitleEl.textContent = "ウォッチリストから銘柄を選ぶと、ここに詳細を表示します。";
    return;
  }

  const entry = state.details.get(symbol);
  if (!entry || entry.loading) {
    detailContentEl.classList.add("hidden");
    detailEmptyEl.classList.remove("hidden");
    detailEmptyEl.innerHTML = "<p>詳細データを取得中です...</p>";
    detailSubtitleEl.textContent = `${symbol} の詳細データを取得しています。`;
    return;
  }

  if (entry.error || !entry.overview) {
    detailContentEl.classList.add("hidden");
    detailEmptyEl.classList.remove("hidden");
    detailEmptyEl.innerHTML = `<p>${escapeHtml(entry?.error || "詳細データの取得に失敗しました。")}</p>`;
    detailSubtitleEl.textContent = "詳細取得に失敗しました。右上の詳細更新をお試しください。";
    return;
  }

  const overview = entry.overview;
  const currentPrice = Number(overview?.price?.current);
  const changeAbs = Number(overview?.price?.change_abs);
  const changePct = Number(overview?.price?.change_pct);
  const volumeToday = Number(overview?.volume?.today);
  const turnover = Number(overview?.volume?.turnover);
  const vwap = Number(overview?.technical?.vwap_1m);
  const vwap5m = Number(overview?.technical?.vwap_5m);
  const ma20 = Number(overview?.technical?.ma_short_20);
  const ma50 = Number(overview?.technical?.ma_mid_50);
  const atr14 = Number(overview?.technical?.atr_14);
  const gapAbs = Number(overview?.price?.gap_abs);
  const gapPct = Number(overview?.price?.gap_pct);
  const beta = Number(overview?.market?.beta_60d_vs_spy);
  const corr = Number(overview?.market?.corr_60d_vs_spy);

  detailContentEl.classList.remove("hidden");
  detailEmptyEl.classList.add("hidden");
  detailSubtitleEl.textContent = "価格、指標、補足表示を 1 ページで確認できます。";

  detailSymbolEl.textContent = symbol;
  detailExchangeEl.textContent = overview?.exchange || "-";
  detailNameEl.textContent = overview?.name || "Company name unavailable";
  detailPriceEl.textContent = Number.isFinite(currentPrice) ? `$${formatPrice(currentPrice)}` : "-";
  detailChangeEl.textContent = Number.isFinite(changePct)
    ? `${formatSigned(changeAbs, 2)} / ${formatSignedPercent(changePct)}`
    : "-";
  detailChangeEl.classList.toggle("up", Number.isFinite(changePct) && changePct > 0);
  detailChangeEl.classList.toggle("down", Number.isFinite(changePct) && changePct < 0);
  detailUpdatedEl.textContent = `${formatDateTime(overview?.price?.updated_at)} / ${overview?.source || "-"}`;

  kpiChangeEl.textContent = Number.isFinite(changePct)
    ? `${formatSigned(changeAbs, 2)} (${formatSignedPercent(changePct)})`
    : "-";
  kpiChangeEl.classList.toggle("up", Number.isFinite(changePct) && changePct > 0);
  kpiChangeEl.classList.toggle("down", Number.isFinite(changePct) && changePct < 0);
  kpiRangeEl.textContent = (Number.isFinite(Number(overview?.price?.day_high)) && Number.isFinite(Number(overview?.price?.day_low)))
    ? `${formatPrice(overview.price.day_high)} / ${formatPrice(overview.price.day_low)}`
    : "-";
  kpiVolumeEl.textContent = Number.isFinite(volumeToday) ? formatCompact(volumeToday) : "未対応";
  kpiTurnoverEl.textContent = Number.isFinite(turnover) ? formatMaybeCurrency(turnover) : "未対応";
  kpiVwapEl.textContent = Number.isFinite(vwap)
    ? formatPrice(vwap)
    : (Number.isFinite(vwap5m) ? `${formatPrice(vwap5m)} (5m)` : "未対応");
  kpiMaEl.textContent = (Number.isFinite(ma20) && Number.isFinite(ma50))
    ? `${formatPrice(ma20)} / ${formatPrice(ma50)}`
    : "未対応";
  kpiAtrEl.textContent = Number.isFinite(atr14) ? formatPrice(atr14) : "未対応";
  kpiGapEl.textContent = Number.isFinite(gapPct) ? `${formatSigned(gapAbs, 2)} / ${formatSignedPercent(gapPct)}` : "未対応";
  kpiBetaEl.textContent = (Number.isFinite(beta) && Number.isFinite(corr))
    ? `${beta.toFixed(2)} / ${corr.toFixed(2)}`
    : "未対応";
  kpiSourceEl.textContent = overview?.source_detail?.mode || overview?.source || "-";

  renderChart(overview);
  renderSupportStatus(overview);
  renderMarketContext(overview);
  renderBannerAndStatus();
}

async function loadDetail(symbol, refresh = false) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;
  const existing = state.details.get(normalized) || {};
  state.details.set(normalized, {
    ...existing,
    loading: true,
    error: "",
  });
  renderDetail();
  if (refreshDetailBtn) refreshDetailBtn.disabled = true;
  try {
    const url = refresh
      ? `/api/security-overview/${encodeURIComponent(normalized)}?refresh=true`
      : `/api/security-overview/${encodeURIComponent(normalized)}`;
    const { response, result } = await fetchJson(url);
    if (!response.ok) {
      throw new Error(result.detail || "詳細データの取得に失敗しました。");
    }
    state.details.set(normalized, {
      loading: false,
      error: "",
      overview: result,
    });
    renderDetail();
  } catch (error) {
    state.details.set(normalized, {
      loading: false,
      error: error instanceof Error ? error.message : "詳細データの取得に失敗しました。",
      overview: existing?.overview || null,
    });
    renderDetail();
    setNotice("詳細取得に失敗しました。無料枠上限や通信状態を確認してください。", "warn");
  } finally {
    if (refreshDetailBtn) refreshDetailBtn.disabled = false;
  }
}

async function selectSymbol(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized || !state.watchlist.includes(normalized)) return;
  state.selectedSymbol = normalized;
  renderWatchlist();
  renderDetail();
  void persistState();
  const cached = state.details.get(normalized);
  if (!cached || !cached.overview) {
    await loadDetail(normalized, false);
  }
}

async function addSymbol(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) return;
  if (state.watchlist.includes(normalized)) {
    await selectSymbol(normalized);
    hideDropdown();
    if (searchInputEl) searchInputEl.value = "";
    return;
  }
  if (state.watchlist.length >= MAX_SYMBOLS) {
    setSearchError(`無料モードでは最大 ${MAX_SYMBOLS} 銘柄です。監視銘柄を減らしてから追加してください。`);
    setNotice("無料枠上限に達しました。監視銘柄を減らすか、更新間隔を延ばしてください。", "warn");
    return;
  }
  state.watchlist = [...state.watchlist, normalized];
  if (!state.selectedSymbol) state.selectedSymbol = normalized;
  setSearchError("");
  renderWatchCount();
  renderWatchlist();
  renderBannerAndStatus();
  hideDropdown();
  if (searchInputEl) searchInputEl.value = "";
  void persistState();
  schedulePolling();
  await refreshQuotes();
  await selectSymbol(normalized);
}

async function removeSymbol(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized || !state.watchlist.includes(normalized)) return;
  state.watchlist = state.watchlist.filter((item) => item !== normalized);
  state.watchRows.delete(normalized);
  state.details.delete(normalized);
  if (state.selectedSymbol === normalized) {
    state.selectedSymbol = state.watchlist[0] || "";
  }
  renderWatchlist();
  renderBannerAndStatus();
  renderDetail();
  void persistState();
  schedulePolling();
  if (state.selectedSymbol) {
    await loadDetail(state.selectedSymbol, false);
  }
}

function schedulePolling() {
  if (state.pollTimer) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = 0;
  }
  if (state.watchlist.length === 0) return;
  const delayMs = Math.max(30_000, (Number(state.pollIntervalSec) || 60) * 1000);
  state.pollTimer = window.setTimeout(async () => {
    await refreshQuotes();
    schedulePolling();
  }, delayMs);
}

async function bootstrapPage() {
  setNotice("ページ初期化中です...", "info");
  const { response, result } = await fetchJson("/api/market-data-lab/bootstrap");
  if (!response.ok) {
    setNotice(result.detail || "初期設定の取得に失敗しました。", "error");
    return;
  }

  state.bootstrap = result;
  state.status = result?.status || null;
  state.helpTexts = result?.help_texts || {};
  state.watchlist = uniqueSymbols(result?.state?.watchlist_symbols).slice(0, MAX_SYMBOLS);
  state.selectedSymbol = normalizeSymbol(result?.state?.last_viewed_symbol);
  if (!state.selectedSymbol || !state.watchlist.includes(state.selectedSymbol)) {
    state.selectedSymbol = state.watchlist[0] || "";
  }
  const chartInterval = String(result?.state?.chart_interval || "").toLowerCase();
  state.chartInterval = ["1min", "5min", "1day"].includes(chartInterval) ? chartInterval : "1day";
  state.chartRange = "1M";
  state.pollIntervalSec = Number(result?.recommended_poll_interval_sec) || 60;

  fillHelpTexts();
  renderGuide(Boolean(result?.onboarding_enabled));
  renderUnsupportedList();
  renderWatchlist();
  renderDetail();
  renderBannerAndStatus();
  setNotice(state.watchlist.length > 0 ? "" : "左上の検索欄から銘柄を追加してください。", "info");

  await Promise.all([
    loadCatalog(false),
    state.watchlist.length > 0 ? refreshQuotes() : Promise.resolve(),
  ]);

  if (state.selectedSymbol) {
    await loadDetail(state.selectedSymbol, false);
  }
  schedulePolling();
}

if (guideCloseBtn) {
  guideCloseBtn.addEventListener("click", async () => {
    await setOnboardingDismissed(true);
  });
}

if (guideReopenBtn) {
  guideReopenBtn.addEventListener("click", async () => {
    await setOnboardingDismissed(false);
  });
}

if (refreshAllBtn) {
  refreshAllBtn.addEventListener("click", async () => {
    await Promise.all([
      refreshQuotes(true),
      state.selectedSymbol ? loadDetail(state.selectedSymbol, true) : Promise.resolve(),
      refreshCredits(true),
    ]);
  });
}

if (refreshCreditsBtn) {
  refreshCreditsBtn.addEventListener("click", async () => {
    refreshCreditsBtn.disabled = true;
    await refreshCredits(true);
    refreshCreditsBtn.disabled = false;
  });
}

if (refreshWatchBtn) {
  refreshWatchBtn.addEventListener("click", async () => {
    await refreshQuotes(true);
  });
}

if (refreshDetailBtn) {
  refreshDetailBtn.addEventListener("click", async () => {
    if (!state.selectedSymbol) return;
    await loadDetail(state.selectedSymbol, true);
  });
}

if (searchInputEl) {
  searchInputEl.addEventListener("focus", () => {
    renderDropdown();
  });
  searchInputEl.addEventListener("input", () => {
    setSearchError("");
    renderDropdown();
  });
  searchInputEl.addEventListener("keydown", async (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      const first = pickCatalogCandidates(searchInputEl.value)[0];
      if (first?.symbol) {
        await addSymbol(first.symbol);
      }
    }
    if (event.key === "Escape") {
      hideDropdown();
    }
  });
}

if (searchDropdownEl) {
  searchDropdownEl.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest("[data-symbol]");
    if (!(button instanceof HTMLElement)) return;
    const symbol = button.getAttribute("data-symbol");
    await addSymbol(symbol);
  });
}

if (watchlistEl) {
  watchlistEl.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const removeButton = target.closest("[data-remove-symbol]");
    if (removeButton instanceof HTMLElement) {
      await removeSymbol(removeButton.getAttribute("data-remove-symbol"));
      return;
    }
    const selectButton = target.closest("[data-select-symbol]");
    if (selectButton instanceof HTMLElement) {
      await selectSymbol(selectButton.getAttribute("data-select-symbol"));
    }
  });
}

document.querySelectorAll(".mdl-example-btn").forEach((button) => {
  button.addEventListener("click", async () => {
    const symbol = button.getAttribute("data-symbol");
    await addSymbol(symbol);
  });
});

sortButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.getAttribute("data-sort-key") || "symbol";
    if (state.sort.key === key) {
      state.sort.direction = state.sort.direction === "asc" ? "desc" : "asc";
    } else {
      state.sort.key = key;
      state.sort.direction = key === "symbol" ? "asc" : "desc";
    }
    renderWatchlist();
  });
});

intervalButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const interval = button.getAttribute("data-interval") || "1day";
    if (!["1min", "5min", "1day"].includes(interval)) return;
    state.chartInterval = interval;
    renderDetail();
    void persistState();
  });
});

rangeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const range = button.getAttribute("data-range") || "1M";
    if (!["1D", "5D", "1M", "6M", "1Y", "MAX"].includes(range)) return;
    state.chartRange = range;
    renderDetail();
  });
});

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) {
    hideDropdown();
    closeParamHelpPopovers();
    return;
  }
  if (!target.closest("#mdl-search-area")) {
    hideDropdown();
  }
  const clickedHelp = target.closest("details.param-help");
  closeParamHelpPopovers(clickedHelp);
});

window.addEventListener("beforeunload", () => {
  if (state.pollTimer) {
    window.clearTimeout(state.pollTimer);
  }
});

bootstrapPage();
