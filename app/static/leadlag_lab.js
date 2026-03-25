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
const strategyChartMetaEl = document.getElementById("llg-strategy-chart-meta");
const strategyChartEl = document.getElementById("llg-strategy-chart");
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

const summaryMetricHelp = {
  "Included US": "今回の計算で実際に使えた US 銘柄数です。選択していても履歴不足などで除外された銘柄は含みません。",
  "Included JP": "今回の計算で実際に使えた JP 銘柄数です。最終的な予測対象になった日本 ETF の本数を示します。",
  Signals: "生成できた signal date の本数です。十分な履歴がある日だけシグナルが作られます。",
  Range: "今回の分析で実際に使ったデータ期間です。前処理後に使えた最初の日から最後の日までを表します。",
};

const strategyMetricHelp = {
  "Annual Return": "観測できた signal day の平均日次 long-short リターンを年率換算した値です。期間全体の CAGR ではありません。",
  "Annual Volatility": "戦略リターンの年率換算の値動きの大きさです。高いほど成績のブレが大きいです。",
  "Return / Risk": "年率リターンを年率ボラティリティで割った比率です。大きいほど、同じリスクに対して効率よく稼げています。",
  "Max Drawdown": "運用期間中の最大の落ち込み幅です。ピークからどれだけ深く下げたかを示します。",
  "Signal Days": "実際に売買シグナルを出せた日数です。",
  "Average Breadth": "1 回の signal date あたりに long/short へ配分された平均銘柄数です。広いほど分散が効きます。",
};

const priorDirectionHelp = {
  global_equal_weight: "全銘柄が同じ向きに動く、市場全体の共通因子を表す prior 方向です。",
  country_spread: "US 群と JP 群の強弱差を表す prior 方向です。国ごとの相対優位を捉えます。",
  cyclical_defensive: "景気敏感群と defensive 群の差を表す prior 方向です。",
};

const SYMBOL_DISPLAY_NAMES = {
  XLB: "Materials",
  XLC: "Communication Services",
  XLE: "Energy",
  XLF: "Financials",
  XLI: "Industrials",
  XLK: "Technology",
  XLP: "Consumer Staples",
  XLRE: "Real Estate",
  XLU: "Utilities",
  XLV: "Health Care",
  XLY: "Consumer Discretionary",
  "1617.T": "食品",
  "1618.T": "エネルギー資源",
  "1619.T": "建設・資材",
  "1620.T": "素材・化学",
  "1621.T": "医薬品",
  "1622.T": "自動車・輸送機",
  "1623.T": "鉄鋼・非鉄",
  "1624.T": "機械",
  "1625.T": "電機・精密",
  "1626.T": "情報通信・サービスその他",
  "1627.T": "電力・ガス",
  "1628.T": "運輸・物流",
  "1629.T": "商社・卸売",
  "1630.T": "小売",
  "1631.T": "銀行",
  "1632.T": "金融（除く銀行）",
  "1633.T": "不動産",
};

const STRATEGY_CHART_WIDTH = 880;
const STRATEGY_CHART_HEIGHT = 320;
const STRATEGY_CHART_PAD_X = 62;
const STRATEGY_CHART_PAD_TOP = 18;
const STRATEGY_CHART_PAD_BOTTOM = 42;
const STRATEGY_CHART_Y_PADDING_RATIO = 0.14;
const STRATEGY_CHART_ZOOM_IN_FACTOR = 0.72;
const STRATEGY_CHART_ZOOM_OUT_FACTOR = 1 / STRATEGY_CHART_ZOOM_IN_FACTOR;
const STRATEGY_CHART_MIN_VISIBLE_POINTS = 12;

const strategyChartState = {
  points: [],
  viewportStart: 0,
  viewportEnd: 0,
  seriesKey: "",
};
let strategyChartPanState = {
  active: false,
  lastClientX: 0,
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

function fmtSigned(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  if (num > 0) return `+${num.toFixed(digits)}`;
  return num.toFixed(digits);
}

function fmtSignedPct(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${fmtSigned(num, digits)}%`;
}

function clampNumber(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatChartDateLabel(value) {
  const raw = String(value || "").trim();
  if (!raw) return "-";
  const dt = new Date(raw.includes("T") ? raw : `${raw}T00:00:00`);
  if (!Number.isNaN(dt.getTime())) {
    return dt.toLocaleDateString("ja-JP", { year: "2-digit", month: "2-digit", day: "2-digit" });
  }
  return raw.includes(" ") ? raw.split(" ")[0] : raw;
}

function formatSymbolDisplay(symbol) {
  const normalized = String(symbol || "").trim().toUpperCase();
  if (!normalized) return "-";
  const name = SYMBOL_DISPLAY_NAMES[normalized];
  return name ? `${normalized} (${name})` : normalized;
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

function buildHelpDetails(text) {
  if (!text) return null;
  const details = document.createElement("details");
  details.className = "param-help llg-param-help";
  const summary = document.createElement("summary");
  summary.textContent = "?";
  const body = document.createElement("p");
  body.textContent = text;
  details.append(summary, body);
  return details;
}

function buildLabeledHelpRow(label, helpText, className = "") {
  const row = document.createElement("span");
  row.className = ["llg-heading-row", className].filter(Boolean).join(" ");
  const labelEl = document.createElement("span");
  labelEl.textContent = label;
  row.appendChild(labelEl);
  const details = buildHelpDetails(helpText);
  if (details) {
    row.appendChild(details);
  }
  return row;
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
    button.textContent = formatSymbolDisplay(symbol);
    button.title = symbol;
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
    const labelEl = document.createElement("span");
    labelEl.className = "label";
    labelEl.appendChild(buildLabeledHelpRow(item.label, item.help || null));
    const valueEl = document.createElement("strong");
    valueEl.textContent = String(item.value ?? "-");
    card.append(labelEl, valueEl);
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
    chip.textContent = formatSymbolDisplay(symbol);
    chip.title = symbol;
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
    const nameTd = document.createElement("td");
    nameTd.appendChild(buildLabeledHelpRow(name, priorDirectionHelp[name] || null, "llg-th-help"));
    const valueTd = document.createElement("td");
    valueTd.textContent = fmtNum(d0[index], 6);
    tr.append(nameTd, valueTd);
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

function buildStrategyAnnualReturnSeries(strategy) {
  const rows = Array.isArray(strategy?.daily_rows) ? strategy.daily_rows : [];
  const orderedRows = rows
    .map((row) => ({
      targetDate: String(row?.target_date || "").trim(),
      grossReturn: Number(row?.gross_return),
      breadth: Number(row?.breadth),
      bucketSize: Number(row?.bucket_size),
    }))
    .filter((row) => row.targetDate && Number.isFinite(row.grossReturn))
    .sort((left, right) => left.targetDate.localeCompare(right.targetDate));

  let cumulativeGrossReturn = 0;
  return orderedRows.map((row, index) => {
    cumulativeGrossReturn += row.grossReturn;
    return {
      t: row.targetDate,
      c: (cumulativeGrossReturn / (index + 1)) * 252.0 * 100.0,
      gross_return_pct: row.grossReturn * 100.0,
      annualized_daily_return_pct: row.grossReturn * 252.0 * 100.0,
      breadth: Number.isFinite(row.breadth) ? row.breadth : null,
      bucket_size: Number.isFinite(row.bucketSize) ? row.bucketSize : null,
      observation_index: index + 1,
    };
  });
}

function buildStrategySeriesKey(points) {
  const safe = Array.isArray(points) ? points : [];
  const first = safe[0]?.t || "";
  const last = safe[safe.length - 1]?.t || "";
  return `${safe.length}:${first}:${last}`;
}

function clampStrategyViewport(start, end, totalPoints) {
  const maxIndex = Math.max(0, totalPoints - 1);
  if (maxIndex <= 0) {
    return { start: 0, end: 0 };
  }

  let nextStart = Number.isFinite(start) ? start : 0;
  let nextEnd = Number.isFinite(end) ? end : maxIndex;
  if (nextEnd < nextStart) {
    [nextStart, nextEnd] = [nextEnd, nextStart];
  }

  const fullSpan = maxIndex;
  const minSpan = Math.max(1, Math.min(STRATEGY_CHART_MIN_VISIBLE_POINTS - 1, fullSpan));
  let span = clampNumber(nextEnd - nextStart, minSpan, fullSpan);
  nextEnd = nextStart + span;

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
  span = clampNumber(nextEnd - nextStart, minSpan, fullSpan);
  if ((nextEnd - nextStart) < span) {
    nextStart = Math.max(0, nextEnd - span);
  }
  return {
    start: clampNumber(nextStart, 0, maxIndex),
    end: clampNumber(nextEnd, 0, maxIndex),
  };
}

function resetStrategyViewport(totalPoints) {
  strategyChartState.viewportStart = 0;
  strategyChartState.viewportEnd = Math.max(0, totalPoints - 1);
}

function setStrategyViewport(start, end, totalPoints) {
  const viewport = clampStrategyViewport(start, end, totalPoints);
  strategyChartState.viewportStart = viewport.start;
  strategyChartState.viewportEnd = viewport.end;
}

function buildStrategyChartHtml(points) {
  const safe = (Array.isArray(points) ? points : [])
    .map((point) => ({
      t: String(point?.t || "").trim(),
      c: Number(point?.c),
      gross_return_pct: Number(point?.gross_return_pct),
      annualized_daily_return_pct: Number(point?.annualized_daily_return_pct),
      breadth: Number(point?.breadth),
      bucket_size: Number(point?.bucket_size),
      observation_index: Number(point?.observation_index),
    }))
    .filter((point) => point.t && Number.isFinite(point.c));

  if (!safe.length) {
    return '<div class="chart-empty">バックテスト結果がないため、グラフを表示できません。</div>';
  }

  const maxIndex = Math.max(0, safe.length - 1);
  const viewport = clampStrategyViewport(
    strategyChartState.viewportStart,
    strategyChartState.viewportEnd,
    safe.length
  );
  strategyChartState.viewportStart = viewport.start;
  strategyChartState.viewportEnd = viewport.end;
  const visibleStart = clampNumber(Math.floor(viewport.start), 0, maxIndex);
  const visibleEnd = clampNumber(Math.ceil(viewport.end), 0, maxIndex);
  const visible = safe.slice(visibleStart, visibleEnd + 1);
  const visibleCount = visible.length;

  let min = Math.min(...visible.map((point) => point.c));
  let max = Math.max(...visible.map((point) => point.c));
  if (min === max) {
    const pad = Math.max(1, Math.abs(min) * 0.12 || 1);
    min -= pad;
    max += pad;
  }
  const rawRange = max - min || 1;
  const yPadding = rawRange * STRATEGY_CHART_Y_PADDING_RATIO;
  min -= yPadding;
  max += yPadding;
  const range = max - min || 1;

  const drawableWidth = STRATEGY_CHART_WIDTH - (STRATEGY_CHART_PAD_X * 2);
  const drawableHeight = STRATEGY_CHART_HEIGHT - STRATEGY_CHART_PAD_TOP - STRATEGY_CHART_PAD_BOTTOM;
  const axisX = STRATEGY_CHART_PAD_X;
  const axisY = STRATEGY_CHART_HEIGHT - STRATEGY_CHART_PAD_BOTTOM;
  const xDenom = Math.max(visibleCount - 1, 1);

  const xForIndex = (index) => STRATEGY_CHART_PAD_X + ((index / xDenom) * drawableWidth);
  const yForValue = (value) => STRATEGY_CHART_PAD_TOP + (1 - ((value - min) / range)) * drawableHeight;

  const polyline = visible.map((point, index) => `${xForIndex(index).toFixed(2)},${yForValue(point.c).toFixed(2)}`).join(" ");
  const lastPointX = xForIndex(Math.max(visibleCount - 1, 0));
  const lastPointY = yForValue(visible[visibleCount - 1].c);

  const yTickCount = 4;
  const yGridLines = [];
  const yTicks = [];
  for (let idx = 0; idx < yTickCount; idx += 1) {
    const ratio = idx / Math.max(yTickCount - 1, 1);
    const y = STRATEGY_CHART_PAD_TOP + (ratio * drawableHeight);
    const value = max - (range * ratio);
    yGridLines.push(
      `<line x1="${axisX}" y1="${y.toFixed(2)}" x2="${(STRATEGY_CHART_WIDTH - STRATEGY_CHART_PAD_X).toFixed(2)}" y2="${y.toFixed(2)}" class="symbol-chart-grid-line"></line>`
    );
    yTicks.push(
      `<text x="${(axisX - 4).toFixed(2)}" y="${(y + 4).toFixed(2)}" class="symbol-chart-axis-label" text-anchor="end">${escapeHtml(fmtSignedPct(value, 1))}</text>`
    );
  }

  if (min < 0 && max > 0) {
    const zeroY = yForValue(0);
    yGridLines.push(
      `<line x1="${axisX}" y1="${zeroY.toFixed(2)}" x2="${(STRATEGY_CHART_WIDTH - STRATEGY_CHART_PAD_X).toFixed(2)}" y2="${zeroY.toFixed(2)}" class="symbol-chart-grid-line llg-chart-zero-line"></line>`
    );
  }

  const xTickCount = Math.min(6, visibleCount);
  const xTicks = [];
  for (let idx = 0; idx < xTickCount; idx += 1) {
    const ratio = idx / Math.max(xTickCount - 1, 1);
    const pointIndex = Math.round(ratio * xDenom);
    const point = visible[pointIndex];
    const x = xForIndex(pointIndex);
    const anchor = idx === 0 ? "start" : (idx === xTickCount - 1 ? "end" : "middle");
    xTicks.push(
      `<line x1="${x.toFixed(2)}" y1="${axisY.toFixed(2)}" x2="${x.toFixed(2)}" y2="${(axisY + 5).toFixed(2)}" class="symbol-chart-axis-tick"></line>`
    );
    xTicks.push(
      `<text x="${x.toFixed(2)}" y="${(STRATEGY_CHART_HEIGHT - 8).toFixed(2)}" class="symbol-chart-axis-label" text-anchor="${anchor}">${escapeHtml(formatChartDateLabel(point?.t))}</text>`
    );
  }

  const hitPoints = visible.map((point, index) => {
    const x = xForIndex(index);
    const y = yForValue(point.c);
    const tooltipParts = [
      point.t,
      `Annual Return: ${fmtSignedPct(point.c)}`,
      `Daily Gross: ${fmtSignedPct(point.gross_return_pct)}`,
      `Daily x252: ${fmtSignedPct(point.annualized_daily_return_pct)}`,
      `Obs: ${point.observation_index}`,
    ];
    if (Number.isFinite(point.breadth)) {
      tooltipParts.push(`Breadth: ${fmtNum(point.breadth, 0)}`);
    }
    if (Number.isFinite(point.bucket_size)) {
      tooltipParts.push(`Bucket: ${fmtNum(point.bucket_size, 0)}`);
    }
    return `
      <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="8" class="llg-chart-hit-point">
        <title>${escapeHtml(tooltipParts.join(" / "))}</title>
      </circle>
    `;
  }).join("");

  const firstPoint = visible[0];
  const latestPoint = visible[visibleCount - 1];
  const isZoomed = visibleStart > 0 || visibleEnd < maxIndex;

  return `
    <div class="chart-controls">
      <div class="chart-toolbar">
        <div class="chart-zoom-actions">
          <button type="button" class="minor-action" data-strategy-chart-action="zoom-in">+</button>
          <button type="button" class="minor-action" data-strategy-chart-action="zoom-out">-</button>
          <button type="button" class="minor-action" data-strategy-chart-action="zoom-reset">Reset</button>
        </div>
        <span class="chart-hint">Wheel: zoom / Drag: pan / Double click: reset</span>
      </div>
      <div class="chart-caption">${visibleCount} / ${safe.length} points${isZoomed ? " (zoomed)" : ""} / ${escapeHtml(firstPoint.t)} - ${escapeHtml(latestPoint.t)} / latest ${escapeHtml(fmtSignedPct(latestPoint.c))}</div>
    </div>
    <div class="chart-scroll-shell">
      <div class="chart-canvas-host">
        <svg class="symbol-chart interactive llg-strategy-line-chart" viewBox="0 0 ${STRATEGY_CHART_WIDTH} ${STRATEGY_CHART_HEIGHT}" preserveAspectRatio="none" role="img" aria-label="annual return trace">
          <rect x="0" y="0" width="${STRATEGY_CHART_WIDTH}" height="${STRATEGY_CHART_HEIGHT}" class="symbol-chart-bg"></rect>
          ${yGridLines.join("")}
          <line x1="${axisX}" y1="${STRATEGY_CHART_PAD_TOP}" x2="${axisX}" y2="${axisY}" class="symbol-chart-axis-line"></line>
          <line x1="${axisX}" y1="${axisY}" x2="${(STRATEGY_CHART_WIDTH - STRATEGY_CHART_PAD_X).toFixed(2)}" y2="${axisY}" class="symbol-chart-axis-line"></line>
          <polyline class="symbol-chart-line" points="${polyline}"></polyline>
          <circle class="symbol-chart-point" cx="${lastPointX.toFixed(2)}" cy="${lastPointY.toFixed(2)}" r="4"></circle>
          ${hitPoints}
          ${yTicks.join("")}
          ${xTicks.join("")}
        </svg>
      </div>
    </div>
  `;
}

function renderStrategyChartFromState() {
  if (!strategyChartEl || !strategyChartMetaEl) return;
  const points = Array.isArray(strategyChartState.points) ? strategyChartState.points : [];
  if (!points.length) {
    strategyChartMetaEl.textContent = "バックテストが無効か、観測できた営業日がありません。";
    strategyChartEl.innerHTML = buildStrategyChartHtml(points);
    return;
  }

  const firstPoint = points[0];
  const latestPoint = points[points.length - 1];
  strategyChartMetaEl.textContent = `${points.length} observed target dates / ${firstPoint.t} -> ${latestPoint.t} / latest ${fmtSignedPct(latestPoint.c)}`;
  strategyChartEl.innerHTML = buildStrategyChartHtml(points);

  const safe = points;
  if (safe.length < 2) return;

  const svg = strategyChartEl.querySelector(".symbol-chart.interactive");
  const zoomInBtn = strategyChartEl.querySelector('[data-strategy-chart-action="zoom-in"]');
  const zoomOutBtn = strategyChartEl.querySelector('[data-strategy-chart-action="zoom-out"]');
  const zoomResetBtn = strategyChartEl.querySelector('[data-strategy-chart-action="zoom-reset"]');
  if (!svg) return;
  if (strategyChartPanState.active) {
    svg.classList.add("dragging");
  }

  const getIndexFromClientX = (clientX) => {
    const rect = svg.getBoundingClientRect();
    const ratio = clampNumber((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    return strategyChartState.viewportStart + (ratio * Math.max(1, strategyChartState.viewportEnd - strategyChartState.viewportStart));
  };

  const zoomAtClientX = (clientX, factor) => {
    const maxIndex = Math.max(1, safe.length - 1);
    const fullSpan = maxIndex;
    const minSpan = Math.max(1, Math.min(STRATEGY_CHART_MIN_VISIBLE_POINTS - 1, fullSpan));
    const currentSpan = Math.max(1, strategyChartState.viewportEnd - strategyChartState.viewportStart);
    const targetSpan = clampNumber(currentSpan * factor, minSpan, fullSpan);
    if (Math.abs(targetSpan - currentSpan) < 0.001) return;
    const centerIndex = clampNumber(getIndexFromClientX(clientX), 0, maxIndex);
    const ratio = (centerIndex - strategyChartState.viewportStart) / currentSpan;
    const nextStart = centerIndex - (targetSpan * ratio);
    const nextEnd = nextStart + targetSpan;
    setStrategyViewport(nextStart, nextEnd, safe.length);
    renderStrategyChartFromState();
  };

  zoomInBtn?.addEventListener("click", () => {
    const rect = svg.getBoundingClientRect();
    zoomAtClientX(rect.left + (rect.width / 2), STRATEGY_CHART_ZOOM_IN_FACTOR);
  });
  zoomOutBtn?.addEventListener("click", () => {
    const rect = svg.getBoundingClientRect();
    zoomAtClientX(rect.left + (rect.width / 2), STRATEGY_CHART_ZOOM_OUT_FACTOR);
  });
  zoomResetBtn?.addEventListener("click", () => {
    resetStrategyViewport(safe.length);
    renderStrategyChartFromState();
  });
  svg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? STRATEGY_CHART_ZOOM_IN_FACTOR : STRATEGY_CHART_ZOOM_OUT_FACTOR;
      zoomAtClientX(event.clientX, factor);
    },
    { passive: false }
  );
  svg.addEventListener("dblclick", () => {
    resetStrategyViewport(safe.length);
    renderStrategyChartFromState();
  });
  svg.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    strategyChartPanState = {
      active: true,
      lastClientX: event.clientX,
    };
    svg.classList.add("dragging");
    document.body.classList.add("chart-panning");
  });
}

function renderStrategyChart(strategy) {
  const points = buildStrategyAnnualReturnSeries(strategy);
  const nextSeriesKey = buildStrategySeriesKey(points);
  if (strategyChartState.seriesKey !== nextSeriesKey) {
    strategyChartState.seriesKey = nextSeriesKey;
    resetStrategyViewport(points.length);
  } else {
    setStrategyViewport(strategyChartState.viewportStart, strategyChartState.viewportEnd, points.length);
  }
  strategyChartState.points = points;
  renderStrategyChartFromState();
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
    const labelEl = document.createElement("span");
    labelEl.className = "label";
    labelEl.appendChild(buildLabeledHelpRow(label, strategyMetricHelp[label] || null));
    const valueEl = document.createElement("strong");
    valueEl.textContent = String(value ?? "-");
    card.append(labelEl, valueEl);
    strategyGridEl.appendChild(card);
  });
  renderStrategyChart(strategy);
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
    {
      label: "Included US",
      value: Array.isArray(summary.included_us_symbols) ? summary.included_us_symbols.length : 0,
      help: summaryMetricHelp["Included US"],
    },
    {
      label: "Included JP",
      value: Array.isArray(summary.included_jp_symbols) ? summary.included_jp_symbols.length : 0,
      help: summaryMetricHelp["Included JP"],
    },
    {
      label: "Signals",
      value: range.generated_signals ?? "-",
      help: summaryMetricHelp.Signals,
    },
    {
      label: "Range",
      value: range.from && range.to ? `${range.from} -> ${range.to}` : "-",
      help: summaryMetricHelp.Range,
    },
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

window.addEventListener("mousemove", (event) => {
  if (!strategyChartPanState.active) return;
  const safe = Array.isArray(strategyChartState.points) ? strategyChartState.points : [];
  if (safe.length < 2 || !strategyChartEl) return;

  const svg = strategyChartEl.querySelector(".symbol-chart.interactive");
  if (!svg) return;

  const rect = svg.getBoundingClientRect();
  const drawableWidth = rect.width * ((STRATEGY_CHART_WIDTH - (STRATEGY_CHART_PAD_X * 2)) / STRATEGY_CHART_WIDTH);
  const deltaX = event.clientX - strategyChartPanState.lastClientX;
  strategyChartPanState.lastClientX = event.clientX;

  const currentStart = strategyChartState.viewportStart;
  const currentEnd = strategyChartState.viewportEnd;
  const span = Math.max(1, currentEnd - currentStart);
  const deltaIndex = (deltaX / Math.max(1, drawableWidth)) * span;
  setStrategyViewport(currentStart - deltaIndex, currentEnd - deltaIndex, safe.length);
  renderStrategyChartFromState();
});

window.addEventListener("mouseup", () => {
  if (!strategyChartPanState.active) return;
  strategyChartPanState.active = false;
  document.body.classList.remove("chart-panning");
  if (!strategyChartEl) return;
  const svg = strategyChartEl.querySelector(".symbol-chart.interactive");
  if (svg) {
    svg.classList.remove("dragging");
  }
});
