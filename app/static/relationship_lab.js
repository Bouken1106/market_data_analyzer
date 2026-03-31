const formEl = document.getElementById("rel-form");
const statusEl = document.getElementById("rel-status");

const symbolsEl = document.getElementById("rel-symbols");
const monthsEl = document.getElementById("rel-months");
const windowDaysEl = document.getElementById("rel-window-days");
const topPairsEl = document.getElementById("rel-top-pairs");
const refreshEl = document.getElementById("rel-refresh");

const avgCorrEl = document.getElementById("rel-avg-corr");
const mostConnectedEl = document.getElementById("rel-most-connected");
const mostDiversifyingEl = document.getElementById("rel-most-diversifying");
const dataSummaryEl = document.getElementById("rel-data-summary");
const skippedEl = document.getElementById("rel-skipped");
const rollingListEl = document.getElementById("rel-rolling-list");
const correlationWrapEl = document.getElementById("rel-correlation-wrap");
const covarianceWrapEl = document.getElementById("rel-covariance-wrap");
const pairsBodyEl = document.getElementById("rel-pairs-body");

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

function fmtNum(value, digits = 3) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function fmtPct(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${num.toFixed(digits)}%`;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json().catch(() => ({}));
  return { response, result };
}

function createHeatColor(value, isCovariance = false) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "transparent";
  const clamped = isCovariance
    ? Math.max(-1, Math.min(1, num * 1000))
    : Math.max(-1, Math.min(1, num));
  if (Math.abs(clamped) < 1e-9) return "rgba(120, 138, 158, 0.10)";
  if (clamped > 0) {
    return `rgba(20, 184, 166, ${0.16 + Math.abs(clamped) * 0.42})`;
  }
  return `rgba(239, 68, 68, ${0.16 + Math.abs(clamped) * 0.42})`;
}

function renderMatrix(targetEl, rows, isCovariance = false) {
  const safeRows = Array.isArray(rows) ? rows : [];
  if (!safeRows.length) {
    targetEl.innerHTML = "<p>No matrix data.</p>";
    return;
  }

  const symbols = safeRows.map((item) => item.symbol || "-");
  const headerCells = symbols.map((symbol) => `<th>${symbol}</th>`).join("");
  const bodyRows = safeRows.map((row) => {
    const cells = Array.isArray(row.values) ? row.values : [];
    const valueCells = cells.map((value) => {
      const text = isCovariance ? fmtNum(value, 6) : fmtNum(value, 3);
      const bg = createHeatColor(value, isCovariance);
      return `<td style="background:${bg}">${text}</td>`;
    }).join("");
    return `<tr><th>${row.symbol || "-"}</th>${valueCells}</tr>`;
  }).join("");

  targetEl.innerHTML = `
    <table class="rel-matrix-table">
      <thead>
        <tr>
          <th>Symbol</th>
          ${headerCells}
        </tr>
      </thead>
      <tbody>${bodyRows}</tbody>
    </table>
  `;
}

function renderPairs(pairs) {
  const safePairs = Array.isArray(pairs) ? pairs : [];
  pairsBodyEl.innerHTML = "";
  if (!safePairs.length) {
    pairsBodyEl.innerHTML = '<tr><td colspan="8">No pair candidates</td></tr>';
    return;
  }

  safePairs.forEach((item) => {
    const tr = document.createElement("tr");
    const spread = Number(item.spread_zscore);
    const spreadClass = Number.isFinite(spread)
      ? spread >= 1.5
        ? "rel-positive"
        : spread <= -1.5
          ? "rel-negative"
          : ""
      : "";
    tr.innerHTML = `
      <td>${item.left || "-"} / ${item.right || "-"}</td>
      <td>${fmtNum(item.correlation, 3)}</td>
      <td>${fmtNum(item.corr_20d, 3)}</td>
      <td>${fmtNum(item.corr_60d, 3)}</td>
      <td>${fmtNum(item.corr_120d, 3)}</td>
      <td>${fmtNum(item.covariance, 6)}</td>
      <td>${fmtNum(item.beta_left_to_right, 3)}</td>
      <td class="${spreadClass}">${fmtNum(item.spread_zscore, 3)}</td>
    `;
    pairsBodyEl.appendChild(tr);
  });
}

function sparklineSvg(series) {
  const values = (Array.isArray(series) ? series : [])
    .map((item) => Number(item?.value))
    .filter((value) => Number.isFinite(value));
  if (values.length < 2) {
    return '<div class="hint">Not enough rolling points.</div>';
  }

  const width = 260;
  const height = 70;
  const min = Math.min(...values, -1);
  const max = Math.max(...values, 1);
  const range = Math.max(max - min, 1e-6);
  const points = values.map((value, index) => {
    const x = (index / (values.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const latest = values[values.length - 1];
  const stroke = latest >= 0 ? "#14b8a6" : "#ef4444";
  return `
    <svg viewBox="0 0 ${width} ${height}" class="rel-sparkline" preserveAspectRatio="none" aria-hidden="true">
      <polyline fill="none" stroke="${stroke}" stroke-width="3" points="${points}"></polyline>
    </svg>
  `;
}

function renderRolling(items) {
  const safeItems = Array.isArray(items) ? items : [];
  if (!safeItems.length) {
    rollingListEl.innerHTML = '<p class="hint">No rolling correlation series.</p>';
    return;
  }

  rollingListEl.innerHTML = safeItems.map((item) => {
    const series = Array.isArray(item.series) ? item.series : [];
    const latest = series.length ? series[series.length - 1] : null;
    return `
      <article class="rel-rolling-card">
        <div class="rel-rolling-head">
          <strong>${item.left || "-"} / ${item.right || "-"}</strong>
          <span>${fmtNum(latest?.value, 3)}</span>
        </div>
        ${sparklineSvg(series)}
      </article>
    `;
  }).join("");
}

function renderSummary(result) {
  const summary = result?.summary || {};
  const dataSummary = result?.data_summary || {};
  const strongest = summary?.most_connected_symbol || {};
  const weakest = summary?.most_diversifying_symbol || {};

  avgCorrEl.textContent = fmtNum(summary.average_abs_correlation, 3);
  mostConnectedEl.textContent = strongest.symbol
    ? `${strongest.symbol} (${fmtNum(strongest.mean_abs_correlation, 3)})`
    : "-";
  mostDiversifyingEl.textContent = weakest.symbol
    ? `${weakest.symbol} (${fmtNum(weakest.mean_abs_correlation, 3)})`
    : "-";
  dataSummaryEl.textContent = `Aligned data: ${dataSummary.from || "-"} -> ${dataSummary.to || "-"} | prices=${dataSummary.price_points || 0} | returns=${dataSummary.return_points || 0}`;

  const skipped = Array.isArray(result?.skipped_symbols) ? result.skipped_symbols : [];
  skippedEl.textContent = skipped.length
    ? `Skipped: ${skipped.map((item) => `${item.symbol} (${item.reason})`).join(", ")}`
    : "Skipped: none";
}

function payloadQuery() {
  const params = new URLSearchParams({
    symbols: String(symbolsEl.value || "").trim(),
    months: String(Number(monthsEl.value || 12)),
    window_days: String(Number(windowDaysEl.value || 60)),
    top_pairs: String(Number(topPairsEl.value || 10)),
  });
  if (refreshEl.checked) {
    params.set("refresh", "true");
  }
  return params.toString();
}

async function runAnalysis() {
  setStatus("Relationship analysis is running...");
  const { response, result } = await fetchJson(`/api/relationships?${payloadQuery()}`);
  if (!response.ok) {
    setStatus(result.detail || "Relationship analysis failed.", true);
    return;
  }

  renderSummary(result);
  renderRolling(result.rolling_correlations);
  renderMatrix(correlationWrapEl, result.correlation_matrix, false);
  renderMatrix(covarianceWrapEl, result.covariance_matrix, true);
  renderPairs(result.pair_candidates);
  setStatus(`Relationship analysis completed for ${Array.isArray(result.analyzed_symbols) ? result.analyzed_symbols.length : 0} symbols.`);
}

formEl?.addEventListener("submit", (event) => {
  event.preventDefault();
  runAnalysis().catch((error) => {
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

window.addEventListener("DOMContentLoaded", () => {
  runAnalysis().catch((error) => {
    setStatus(error instanceof Error ? error.message : "Unexpected error occurred.", true);
  });
});
