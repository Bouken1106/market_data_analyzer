const statusEl = document.getElementById("llg-status");
const runBtn = document.getElementById("llg-run");
const resetDefaultsBtn = document.getElementById("llg-reset-defaults");

const usSymbolsEl = document.getElementById("llg-us-symbols");
const jpSymbolsEl = document.getElementById("llg-jp-symbols");
const cyclicalSymbolsEl = document.getElementById("llg-cyclical-symbols");
const defensiveSymbolsEl = document.getElementById("llg-defensive-symbols");
const usSummaryEl = document.getElementById("llg-us-summary");
const jpSummaryEl = document.getElementById("llg-jp-summary");
const cyclicalSummaryEl = document.getElementById("llg-cyclical-summary");
const defensiveSummaryEl = document.getElementById("llg-defensive-summary");
const helpUsEl = document.getElementById("llg-help-us");
const helpJpEl = document.getElementById("llg-help-jp");
const helpCyclicalEl = document.getElementById("llg-help-cyclical");
const helpDefensiveEl = document.getElementById("llg-help-defensive");
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
const selectorState = {
  universe: {
    us: [],
    jp: [],
  },
  selected: {
    us: new Set(),
    jp: new Set(),
    cyclical: new Set(),
    defensive: new Set(),
  },
};

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

function uniqueOrderedSymbols(values) {
  const seen = new Set();
  const ordered = [];
  (Array.isArray(values) ? values : []).forEach((value) => {
    const symbol = String(value || "").trim();
    if (!symbol || seen.has(symbol)) return;
    seen.add(symbol);
    ordered.push(symbol);
  });
  return ordered;
}

function closeParamHelpPopovers(exceptDetail = null) {
  const openDetails = document.querySelectorAll("details.param-help[open]");
  openDetails.forEach((detail) => {
    if (detail === exceptDetail) return;
    detail.removeAttribute("open");
  });
}

function summarizeSelection(values, limit = 6) {
  const symbols = uniqueOrderedSymbols(values);
  if (!symbols.length) return "未選択";
  if (symbols.length <= limit) return symbols.join(", ");
  return `${symbols.slice(0, limit).join(", ")} ほか ${symbols.length - limit} 銘柄`;
}

function orderedSelection(universeKey) {
  const universe = universeKey === "labels"
    ? activeLabelCandidates()
    : uniqueOrderedSymbols(selectorState.universe[universeKey]);
  const selectedSet = universeKey === "labels"
    ? null
    : selectorState.selected[universeKey];
  if (universeKey === "labels") {
    return universe;
  }
  return universe.filter((symbol) => selectedSet.has(symbol));
}

function activeLabelCandidates() {
  const activeSet = new Set([
    ...orderedSelection("us"),
    ...orderedSelection("jp"),
  ]);
  return uniqueOrderedSymbols([
    ...selectorState.universe.us,
    ...selectorState.universe.jp,
  ]).filter((symbol) => activeSet.has(symbol));
}

function syncLabelSelections() {
  const activeCandidates = new Set(activeLabelCandidates());
  selectorState.selected.cyclical = new Set(
    [...selectorState.selected.cyclical].filter((symbol) => activeCandidates.has(symbol))
  );
  selectorState.selected.defensive = new Set(
    [...selectorState.selected.defensive].filter(
      (symbol) => activeCandidates.has(symbol) && !selectorState.selected.cyclical.has(symbol)
    )
  );
}

function applyDefaults(defaults) {
  selectorState.universe.us = uniqueOrderedSymbols(defaults?.universe?.us || defaults?.us_symbols || []);
  selectorState.universe.jp = uniqueOrderedSymbols(defaults?.universe?.jp || defaults?.jp_symbols || []);
  selectorState.selected.us = new Set(uniqueOrderedSymbols(defaults?.us_symbols || []));
  selectorState.selected.jp = new Set(uniqueOrderedSymbols(defaults?.jp_symbols || []));
  selectorState.selected.cyclical = new Set(uniqueOrderedSymbols(defaults?.cyclical_symbols || []));
  selectorState.selected.defensive = new Set(uniqueOrderedSymbols(defaults?.defensive_symbols || []));
  syncLabelSelections();
  renderSelectors();
}

function renderOptionGrid(targetEl, symbols, selectedSet, kind) {
  targetEl.innerHTML = "";
  const values = uniqueOrderedSymbols(symbols);
  if (!values.length) {
    targetEl.innerHTML = '<div class="llg-option-empty">候補がありません。</div>';
    return;
  }
  values.forEach((symbol) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "llg-option-btn";
    button.dataset.selectKind = kind;
    button.dataset.symbol = symbol;
    if (selectedSet.has(symbol)) {
      button.classList.add("is-active");
    }
    button.textContent = symbol;
    targetEl.appendChild(button);
  });
}

function renderLabelGrid(targetEl, symbols, selectedSet, kind, blockedSet) {
  targetEl.innerHTML = "";
  const values = uniqueOrderedSymbols(symbols);
  if (!values.length) {
    targetEl.innerHTML = '<div class="llg-option-empty">先に US / JP Symbols を選択してください。</div>';
    return;
  }
  values.forEach((symbol) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "llg-option-btn";
    button.dataset.selectKind = kind;
    button.dataset.symbol = symbol;
    if (selectedSet.has(symbol)) {
      button.classList.add("is-active");
    }
    if (blockedSet.has(symbol) && !selectedSet.has(symbol)) {
      button.disabled = true;
      button.classList.add("is-disabled");
    }
    button.textContent = symbol;
    targetEl.appendChild(button);
  });
}

function renderHelpTexts() {
  const usSelected = orderedSelection("us");
  const jpSelected = orderedSelection("jp");
  const cyclicalSelected = selectorSymbolsFor("cyclical");
  const defensiveSelected = selectorSymbolsFor("defensive");

  if (helpUsEl) {
    helpUsEl.textContent = `現在選択中の US Symbols は、各 signal date の米国側 close-to-close リターンを標準化した入力として因子抽出に入ります。ここで選んだ銘柄群が、翌営業日の日本側予測を作る情報集合です。現在選択中: ${summarizeSelection(usSelected)}`;
  }
  if (helpJpEl) {
    helpJpEl.textContent = `現在選択中の JP Symbols は、モデルが翌営業日に open-to-close を予測する対象銘柄です。US 側と合わせて相関構造も作りますが、最終的に signal が出るのはこの日本側銘柄です。現在選択中: ${summarizeSelection(jpSelected)}`;
  }
  if (helpCyclicalEl) {
    helpCyclicalEl.textContent = `Cyclical Labels は prior subspace の第3方向で +1 側に置かれ、景気敏感グループとして C0 / D0 の正則化ターゲットを作るために使います。単なる表示ラベルではなく、相関構造の事前仮説に効きます。現在選択中: ${summarizeSelection(cyclicalSelected)}`;
  }
  if (helpDefensiveEl) {
    helpDefensiveEl.textContent = `Defensive Labels は prior subspace の第3方向で -1 側に置かれ、ディフェンシブグループとして cyclical と対になる方向を作ります。同じ銘柄を両方に入れないことで、景気敏感 vs 防御的の軸を明確にします。現在選択中: ${summarizeSelection(defensiveSelected)}`;
  }
}

function renderSelectors() {
  const usUniverse = uniqueOrderedSymbols(selectorState.universe.us);
  const jpUniverse = uniqueOrderedSymbols(selectorState.universe.jp);
  const labelUniverse = activeLabelCandidates();
  const usSelected = orderedSelection("us");
  const jpSelected = orderedSelection("jp");
  const cyclicalSelected = labelUniverse.filter((symbol) => selectorState.selected.cyclical.has(symbol));
  const defensiveSelected = labelUniverse.filter((symbol) => selectorState.selected.defensive.has(symbol));

  renderOptionGrid(usSymbolsEl, usUniverse, selectorState.selected.us, "us");
  renderOptionGrid(jpSymbolsEl, jpUniverse, selectorState.selected.jp, "jp");
  renderLabelGrid(
    cyclicalSymbolsEl,
    labelUniverse,
    selectorState.selected.cyclical,
    "cyclical",
    selectorState.selected.defensive
  );
  renderLabelGrid(
    defensiveSymbolsEl,
    labelUniverse,
    selectorState.selected.defensive,
    "defensive",
    selectorState.selected.cyclical
  );

  usSummaryEl.textContent = `${usSelected.length} / ${usUniverse.length} selected`;
  jpSummaryEl.textContent = `${jpSelected.length} / ${jpUniverse.length} selected`;
  cyclicalSummaryEl.textContent = `${cyclicalSelected.length} selected from active universe (${labelUniverse.length})`;
  defensiveSummaryEl.textContent = `${defensiveSelected.length} selected from active universe (${labelUniverse.length})`;
  renderHelpTexts();
}

function toggleUniverseSelection(kind, symbol) {
  const target = selectorState.selected[kind];
  if (!target) return;
  if (target.has(symbol)) {
    target.delete(symbol);
  } else {
    target.add(symbol);
  }
  syncLabelSelections();
  renderSelectors();
}

function toggleLabelSelection(kind, symbol) {
  const target = selectorState.selected[kind];
  if (!target) return;
  const activeCandidates = new Set(activeLabelCandidates());
  if (!activeCandidates.has(symbol)) return;
  if (target.has(symbol)) {
    target.delete(symbol);
  } else {
    target.add(symbol);
    if (kind === "cyclical") {
      selectorState.selected.defensive.delete(symbol);
    } else if (kind === "defensive") {
      selectorState.selected.cyclical.delete(symbol);
    }
  }
  renderSelectors();
}

function selectorSymbolsFor(kind) {
  if (kind === "cyclical" || kind === "defensive") {
    const labelUniverse = activeLabelCandidates();
    return labelUniverse.filter((symbol) => selectorState.selected[kind].has(symbol));
  }
  return orderedSelection(kind);
}

function setSelectorSelection(kind, action) {
  if (kind === "us" || kind === "jp") {
    if (action === "all") {
      selectorState.selected[kind] = new Set(uniqueOrderedSymbols(selectorState.universe[kind]));
    } else if (action === "clear") {
      selectorState.selected[kind] = new Set();
    }
    syncLabelSelections();
    renderSelectors();
    return;
  }
  if ((kind === "cyclical" || kind === "defensive") && action === "clear") {
    selectorState.selected[kind] = new Set();
    renderSelectors();
  }
}

function payloadFromForm() {
  return {
    us_symbols: selectorSymbolsFor("us").join(","),
    jp_symbols: selectorSymbolsFor("jp").join(","),
    cyclical_symbols: selectorSymbolsFor("cyclical").join(","),
    defensive_symbols: selectorSymbolsFor("defensive").join(","),
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
  applyDefaults(defaults);
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

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;

  const openedHelp = target.closest("details.param-help");
  if (openedHelp) {
    closeParamHelpPopovers(openedHelp);
    return;
  }

  const optionButton = event.target.closest("[data-select-kind]");
  if (optionButton) {
    const kind = optionButton.dataset.selectKind;
    const symbol = optionButton.dataset.symbol;
    if (!kind || !symbol) return;
    if (kind === "us" || kind === "jp") {
      toggleUniverseSelection(kind, symbol);
      return;
    }
    if (kind === "cyclical" || kind === "defensive") {
      toggleLabelSelection(kind, symbol);
    }
    return;
  }

  const actionButton = event.target.closest("[data-selector-action]");
  if (actionButton) {
    const kind = actionButton.dataset.selectorTarget;
    const action = actionButton.dataset.selectorAction;
    if (!kind || !action) return;
    setSelectorSelection(kind, action);
    return;
  }

  closeParamHelpPopovers();
});

runBtn?.addEventListener("click", () => {
  runAnalysis().catch((error) => {
    running = false;
    runBtn.disabled = false;
    setStatus(error instanceof Error ? error.message : "Unexpected error.", true);
  });
});

resetDefaultsBtn?.addEventListener("click", async () => {
  const { response, result } = await fetchJson("/api/leadlag/config");
  if (!response.ok || !result.ok) {
    setStatus(result.detail || "Failed to reset defaults.", true);
    return;
  }
  setDefaults(result.defaults || {});
  setStatus("既定の候補選択へ戻しました。");
});

window.addEventListener("DOMContentLoaded", () => {
  bootstrap().catch((error) => {
    setStatus(error instanceof Error ? error.message : "Failed to initialize.", true);
  });
});
