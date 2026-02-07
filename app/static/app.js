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

const MAX_SYMBOLS = 8;
const MAX_DROPDOWN_ITEMS = 120;

const rowsBySymbol = new Map();
let eventSource;
let symbolCatalog = [];
let selectedSymbols = [];
let syncInFlight = false;
let syncQueued = false;

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

function setSelectionError(message) {
  selectionErrorEl.textContent = message || "";
}

function setCatalogMeta(message) {
  catalogMetaEl.textContent = message || "";
}

function setStatus(status) {
  modeEl.textContent = status?.mode ?? "-";
  wsStateEl.textContent = status?.ws_connected ? "connected" : "disconnected";
  fallbackEl.textContent = `${status?.fallback_poll_interval_sec ?? "-"} sec`;
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
}

function goHistorical(symbol) {
  window.location.href = `/historical/${encodeURIComponent(symbol)}`;
}

function ensureRow(symbol) {
  if (rowsBySymbol.has(symbol)) {
    return rowsBySymbol.get(symbol);
  }

  const tr = document.createElement("tr");
  tr.dataset.symbol = symbol;
  tr.innerHTML = `
    <td class="sym">
      <div class="sym-cell">
        <button type="button" class="sym-open" data-symbol="${symbol}">${symbol}</button>
        <button type="button" class="sym-remove" data-symbol="${symbol}" title="Remove symbol">x</button>
      </div>
    </td>
    <td class="price">-</td>
    <td class="updated-cell">
      <div class="time">-</div>
      <div class="source-meta">source: -</div>
    </td>
  `;
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

function renderRow(update) {
  const tr = ensureRow(update.symbol);
  tr.querySelector(".price").textContent = formatPrice(update.price);
  tr.querySelector(".time").textContent = formatTime(update.timestamp);
  tr.querySelector(".source-meta").textContent = `source: ${update.source || "-"}`;

  tr.classList.remove("flash");
  void tr.offsetWidth;
  tr.classList.add("flash");

  refreshRowActionState();
}

function resetRows(symbols) {
  rowsBySymbol.clear();
  tableBody.innerHTML = "";
  symbols.forEach((symbol) => ensureRow(symbol));
  refreshRowActionState();
}

function renderRows(rows, fallbackSymbols = []) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const symbols = safeRows.length > 0 ? safeRows.map((row) => row.symbol) : fallbackSymbols;
  resetRows(symbols);
  safeRows.forEach((row) => renderRow(row));
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
      const response = await fetch("/api/symbols", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ symbols: payloadSymbols.join(",") }),
      });

      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        setSelectionError(result.detail || "Failed to update symbols");
        break;
      }

      setSelectionError("");
      setStatus(result.status || {});
      const serverSymbols = Array.isArray(result.symbols) ? result.symbols : payloadSymbols;
      renderRows(result.rows, serverSymbols);
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
    const response = await fetch(refresh ? "/api/symbol-catalog?refresh=true" : "/api/symbol-catalog");
    const result = await response.json().catch(() => ({}));

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
    const response = await fetch("/api/credits?refresh=true");
    const result = await response.json().catch(() => ({}));
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
