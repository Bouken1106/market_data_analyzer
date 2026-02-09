const formEl = document.getElementById("compare-form");
const statusEl = document.getElementById("cmp-status");
const runBtn = document.getElementById("cmp-run");
const cancelBtn = document.getElementById("cmp-cancel");
const progressWrapEl = document.getElementById("cmp-progress-wrap");
const progressBarEl = document.getElementById("cmp-progress-bar");
const progressTextEl = document.getElementById("cmp-progress-text");
const summaryBodyEl = document.getElementById("cmp-summary-body");
const detailBodyEl = document.getElementById("cmp-detail-body");

const symbolsEl = document.getElementById("cmp-symbols");
const modelLstmEl = document.getElementById("cmp-model-lstm");
const modelPatchtstEl = document.getElementById("cmp-model-patchtst");
const yearsEl = document.getElementById("cmp-years");
const seqLenEl = document.getElementById("cmp-seq-len");
const hiddenEl = document.getElementById("cmp-hidden");
const layersEl = document.getElementById("cmp-layers");
const dropoutEl = document.getElementById("cmp-dropout");
const lrEl = document.getElementById("cmp-lr");
const batchEl = document.getElementById("cmp-batch");
const epochsEl = document.getElementById("cmp-epochs");
const patienceEl = document.getElementById("cmp-patience");
const seedEl = document.getElementById("cmp-seed");
const refreshEl = document.getElementById("cmp-refresh");
const evalMonthsEl = document.getElementById("cmp-eval-months");

let currentJobId = "";
let pollingTimer = null;
let running = false;
let cancelling = false;

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

function setProgress(value, message = "") {
  const safe = Math.max(0, Math.min(100, Number(value) || 0));
  progressWrapEl.classList.remove("hidden");
  progressBarEl.style.width = `${safe}%`;
  progressBarEl.setAttribute("aria-valuenow", String(Math.round(safe)));
  progressTextEl.textContent = message ? `${Math.round(safe)}% | ${message}` : `${Math.round(safe)}%`;
}

function resetProgress() {
  progressWrapEl.classList.add("hidden");
  progressBarEl.style.width = "0%";
  progressBarEl.setAttribute("aria-valuenow", "0");
  progressTextEl.textContent = "0%";
}

function syncButtons() {
  runBtn.disabled = running;
  cancelBtn.disabled = !running || !currentJobId || cancelling;
  cancelBtn.textContent = cancelling ? "停止中..." : "停止";
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json().catch(() => ({}));
  return { response, result };
}

function fmt(value, digits = 6) {
  if (value === null || value === undefined) return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function fmtInt(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return String(Math.round(num));
}

function renderSummary(rows) {
  summaryBodyEl.innerHTML = "";
  const list = Array.isArray(rows) ? rows : [];
  if (list.length === 0) {
    summaryBodyEl.innerHTML = '<tr><td colspan="7">No results</td></tr>';
    return;
  }

  list.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.model_id || "-"}</td>
      <td>${fmtInt(item.success_count)}</td>
      <td>${fmt(item.mean_pinball_loss, 6)}</td>
      <td>${fmt(item.mean_mae_return, 6)}</td>
      <td>${fmt(item.mean_rmse_return, 6)}</td>
      <td>${fmt(item.mean_mape_price_pct, 4)}</td>
      <td>${fmt(item.mean_coverage_90, 4)}</td>
    `;
    summaryBodyEl.appendChild(tr);
  });
}

function renderDetails(rows) {
  detailBodyEl.innerHTML = "";
  const list = Array.isArray(rows) ? rows : [];
  if (list.length === 0) {
    detailBodyEl.innerHTML = '<tr><td colspan="15">No results</td></tr>';
    return;
  }

  list.forEach((item) => {
    const metrics = item.metrics || {};
    const range = metrics.test_from && metrics.test_to ? `${metrics.test_from} - ${metrics.test_to}` : "-";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.symbol || "-"}</td>
      <td>${item.model_id || "-"}</td>
      <td>${item.status || "-"}</td>
      <td>${range}</td>
      <td>${fmt(metrics.mean_pinball_loss, 6)}</td>
      <td>${fmt(metrics.mae_return, 6)}</td>
      <td>${fmt(metrics.rmse_return, 6)}</td>
      <td>${fmt(metrics.mae_price, 4)}</td>
      <td>${fmt(metrics.rmse_price, 4)}</td>
      <td>${fmt(metrics.mape_price_pct, 4)}</td>
      <td>${fmt(metrics.smape_price_pct, 4)}</td>
      <td>${fmt(metrics.coverage_90, 4)}</td>
      <td>${fmt(metrics.coverage_50, 4)}</td>
      <td>${fmtInt(metrics.epochs_trained)}</td>
      <td>${item.error || "-"}</td>
    `;
    detailBodyEl.appendChild(tr);
  });
}

function collectModels() {
  const out = [];
  if (modelLstmEl.checked) out.push("quantile_lstm");
  if (modelPatchtstEl.checked) out.push("patchtst_quantile");
  return out;
}

function requestPayload() {
  return {
    symbols: String(symbolsEl.value || "").trim(),
    models: collectModels().join(","),
    years: Number(yearsEl.value || 5),
    sequence_length: Number(seqLenEl.value || 60),
    hidden_size: Number(hiddenEl.value || 64),
    num_layers: Number(layersEl.value || 2),
    dropout: Number(dropoutEl.value || 0.2),
    learning_rate: Number(lrEl.value || 0.001),
    batch_size: Number(batchEl.value || 64),
    max_epochs: Number(epochsEl.value || 80),
    patience: Number(patienceEl.value || 10),
    seed: Number(seedEl.value || 42),
    refresh: Boolean(refreshEl.checked),
    eval_months: Number(evalMonthsEl.value || 2),
  };
}

function stopPolling() {
  if (pollingTimer) {
    window.clearTimeout(pollingTimer);
    pollingTimer = null;
  }
}

async function pollJob() {
  if (!currentJobId) return;
  const { response, result } = await fetchJson(`/api/ml/jobs/${encodeURIComponent(currentJobId)}`);
  if (!response.ok) {
    setStatus(result.detail || "ジョブ状態の取得に失敗しました。", true);
    running = false;
    cancelling = false;
    syncButtons();
    return;
  }

  const job = result;
  setProgress(Number(job.progress || 0), String(job.message || ""));

  if (job.status === "completed") {
    running = false;
    cancelling = false;
    syncButtons();
    const payload = job.result || {};
    renderSummary(payload.summary_by_model || []);
    renderDetails(payload.rows || []);
    setStatus(`比較が完了しました。成功 ${payload.success_count || 0} / 失敗 ${payload.failed_count || 0}`);
    return;
  }

  if (job.status === "failed") {
    running = false;
    cancelling = false;
    syncButtons();
    setStatus(job.error || "比較に失敗しました。", true);
    return;
  }

  if (job.status === "cancelled") {
    running = false;
    cancelling = false;
    syncButtons();
    setStatus(job.message || "停止しました。");
    return;
  }

  if (job.status === "cancelling") {
    cancelling = true;
    syncButtons();
  }

  pollingTimer = window.setTimeout(() => {
    pollJob().catch((error) => {
      running = false;
      cancelling = false;
      syncButtons();
      setStatus(error instanceof Error ? error.message : "ジョブ監視中にエラーが発生しました。", true);
    });
  }, 1000);
}

async function startComparison() {
  const models = collectModels();
  if (models.length === 0) {
    setStatus("比較対象モデルを1つ以上選択してください。", true);
    return;
  }

  const payload = requestPayload();
  running = true;
  cancelling = false;
  currentJobId = "";
  syncButtons();
  setProgress(0, "ジョブを開始しています。");
  setStatus("比較ジョブを開始しています...");

  const { response, result } = await fetchJson("/api/ml/compare/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    running = false;
    cancelling = false;
    syncButtons();
    setStatus(result.detail || "比較ジョブの開始に失敗しました。", true);
    return;
  }

  currentJobId = String(result.job_id || "");
  if (!currentJobId) {
    running = false;
    syncButtons();
    setStatus("ジョブIDを取得できませんでした。", true);
    return;
  }

  await pollJob();
}

async function cancelComparison() {
  if (!currentJobId || !running) return;
  cancelling = true;
  syncButtons();

  const { response, result } = await fetchJson(`/api/ml/jobs/${encodeURIComponent(currentJobId)}/cancel`, {
    method: "POST",
  });

  if (!response.ok) {
    cancelling = false;
    syncButtons();
    setStatus(result.detail || "停止要求に失敗しました。", true);
    return;
  }

  setStatus(result.message || "停止を要求しました。");
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPolling();
  await startComparison();
});

cancelBtn.addEventListener("click", async () => {
  await cancelComparison();
});

window.addEventListener("beforeunload", () => {
  stopPolling();
});

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) {
    closeParamHelpPopovers();
    return;
  }
  const clickedHelp = target.closest("details.param-help");
  closeParamHelpPopovers(clickedHelp);
});

resetProgress();
renderSummary([]);
renderDetails([]);
syncButtons();
