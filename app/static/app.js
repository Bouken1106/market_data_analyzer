const tableBody = document.getElementById("price-table");
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

function applyInsightToRow(tr, symbol, priceValue) {
  const changeEl = getRowCell(tr, "change");
  const sparklineEl = getRowCell(tr, "sparkline");
  if (!changeEl || !sparklineEl) return;
  const insight = symbolInsightsBySymbol.get(symbol);

  changeEl.classList.remove("up", "down");

  const currentPrice = Number(priceValue);
  const latestClose = Number(insight?.latest_close);
  const previousClose = Number(insight?.previous_close);
  const marketIsOpen = openSymbolsSet.has(symbol);
  const referenceClose = marketIsOpen
    ? latestClose
    : (Number.isFinite(previousClose) && previousClose > 0 ? previousClose : latestClose);

  if (Number.isFinite(currentPrice) && Number.isFinite(referenceClose) && referenceClose > 0) {
    const pct = ((currentPrice - referenceClose) / referenceClose) * 100;
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

  sparklineEl.innerHTML = renderSparklineSvg(insight?.trend_30d ?? []);
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
  symTd.appendChild(symCell);
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
    if (openBtn) {
      openBtn.dataset.symbol = symbol;
      openBtn.textContent = symbol;
    }
    if (removeBtn) {
      removeBtn.dataset.symbol = symbol;
      removeBtn.disabled = disableRemove;
      removeBtn.title = disableRemove ? "At least one symbol is required" : "Remove symbol";
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

loadSymbolCatalog().catch(() => {
  setCatalogMeta("Failed to load symbol catalog");
});
connectEventStream();
startMarketClock();
