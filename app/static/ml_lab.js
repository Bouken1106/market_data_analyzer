const headerUpdatedEl = document.getElementById("mlops-header-updated");
const headerEnvEl = document.getElementById("mlops-header-env");
const predictionDateEl = document.getElementById("mlops-prediction-date");
const universeFilterEl = document.getElementById("mlops-universe-filter");
const modelFamilyEl = document.getElementById("mlops-model-family");
const featureSetEl = document.getElementById("mlops-feature-set");
const costBufferEl = document.getElementById("mlops-cost-buffer");
const runNoteEl = document.getElementById("mlops-run-note");
const refreshDataBtn = document.getElementById("mlops-refresh-data");
const runInferenceBtn = document.getElementById("mlops-run-inference");
const createTrainingBtn = document.getElementById("mlops-create-training");
const exportReportBtn = document.getElementById("mlops-export-report");
const exportCsvBtn = document.getElementById("mlops-export-csv");
const runBacktestBtn = document.getElementById("mlops-run-backtest");
const roleBadgeEl = document.getElementById("mlops-role-badge");
const globalBadgeEl = document.getElementById("mlops-global-badge");
const globalStatusEl = document.getElementById("mlops-global-status");
const jobBadgeEl = document.getElementById("mlops-job-badge");
const jobStatusEl = document.getElementById("mlops-job-status");
const sidebarStatusEl = document.getElementById("mlops-sidebar-status");
const tabs = Array.from(document.querySelectorAll(".mlops-tab"));
const tabPanels = Array.from(document.querySelectorAll(".mlops-tab-panel"));
const flowTargets = Array.from(document.querySelectorAll("[data-tab-target]"));

const dashboardSummaryEl = document.getElementById("mlops-dashboard-summary");
const dashboardCaptionEl = document.getElementById("mlops-dashboard-caption");
const dashboardFootnoteEl = document.getElementById("mlops-dashboard-footnote");
const predictionSearchEl = document.getElementById("mlops-prediction-search");
const predictionBodyEl = document.getElementById("mlops-prediction-body");
const detailHeadEl = document.getElementById("mlops-detail-head");
const detailMetricsEl = document.getElementById("mlops-detail-metrics");
const featureContribEl = document.getElementById("mlops-feature-contrib");
const detailNotesEl = document.getElementById("mlops-detail-notes");
const sectorScoresEl = document.getElementById("mlops-sector-scores");
const alertListEl = document.getElementById("mlops-alert-list");
const inferenceLogEl = document.getElementById("mlops-inference-log");

const trainConfigEl = document.getElementById("mlops-train-config");
const trainRulesEl = document.getElementById("mlops-train-rules");
const trainSummaryEl = document.getElementById("mlops-train-summary");
const trainCompareBodyEl = document.getElementById("mlops-train-compare-body");
const trainAcceptanceEl = document.getElementById("mlops-train-acceptance");
const foldBodyEl = document.getElementById("mlops-fold-body");
const foldChartEl = document.getElementById("mlops-fold-chart");
const scoreDistributionEl = document.getElementById("mlops-score-distribution");
const trainWindowMonthsEl = document.getElementById("mlops-train-window-months");
const gapDaysEl = document.getElementById("mlops-gap-days");
const validWindowMonthsEl = document.getElementById("mlops-valid-window-months");
const randomSeedEl = document.getElementById("mlops-random-seed");
const trainNoteEl = document.getElementById("mlops-train-note");

const backtestSettingsEl = document.getElementById("mlops-backtest-settings");
const backtestCompareBodyEl = document.getElementById("mlops-backtest-compare-body");
const backtestSummaryEl = document.getElementById("mlops-backtest-summary");
const equityChartEl = document.getElementById("mlops-equity-chart");
const monthlyHeatmapEl = document.getElementById("mlops-monthly-heatmap");
const backtestDailyDistributionEl = document.getElementById("mlops-backtest-daily-distribution");
const backtestExceptionBodyEl = document.getElementById("mlops-backtest-exception-body");

const modelBodyEl = document.getElementById("mlops-model-body");
const adoptModelBtn = document.getElementById("mlops-adopt-model");
const modelDetailHeadEl = document.getElementById("mlops-model-detail-head");
const modelDetailGridEl = document.getElementById("mlops-model-detail-grid");
const modelDecisionEl = document.getElementById("mlops-model-decision");
const modelExplainabilityEl = document.getElementById("mlops-model-explainability");
const modelAuditEl = document.getElementById("mlops-model-audit");

const pipelineGridEl = document.getElementById("mlops-pipeline-grid");
const opsSummaryEl = document.getElementById("mlops-ops-summary");
const opsAlertsEl = document.getElementById("mlops-ops-alerts");
const opsLogsEl = document.getElementById("mlops-ops-logs");
const opsCoverageBreakdownEl = document.getElementById("mlops-ops-coverage-breakdown");
const opsDriftChartEl = document.getElementById("mlops-ops-drift-chart");
const opsDriftNoteEl = document.getElementById("mlops-ops-drift-note");

const appState = {
  activeTab: "dashboard",
  snapshot: null,
  job: null,
  jobPollTimer: 0,
  jobPollToken: 0,
  filters: {
    prediction_date: "",
    universe_filter: "jp_large_cap_stooq_v1",
    model_family: "LightGBM Classifier",
    feature_set: "base_v1",
    cost_buffer: "0.0",
    train_window_months: "12",
    gap_days: "5",
    valid_window_months: "1",
    random_seed: "42",
    train_note: "",
    run_note: "",
  },
  search: "",
  selectedCode: "",
  selectedModelVersion: "",
  busyAction: "",
};

const JOB_TERMINAL_STATUSES = new Set(["SUCCEEDED", "FAILED", "CANCELLED"]);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function levelClass(level) {
  if (level === "normal") return "is-normal";
  if (level === "warning") return "is-warning";
  if (level === "error") return "is-error";
  return "is-unknown";
}

function jobLevel(status) {
  if (status === "SUCCEEDED") return "normal";
  if (status === "FAILED" || status === "CANCELLED") return "error";
  if (status === "RUNNING" || status === "QUEUED") return "warning";
  return "unknown";
}

function jobKindLabel(kind) {
  if (kind === "stock_prediction_run") return "推論ジョブ";
  if (kind === "stock_training_job") return "学習ジョブ";
  if (kind === "stock_backtest_run") return "バックテスト";
  return "ジョブ";
}

function hasActiveJob() {
  return Boolean(appState.job && !JOB_TERMINAL_STATUSES.has(appState.job.status || ""));
}

function stopJobPolling() {
  if (appState.jobPollTimer) {
    window.clearTimeout(appState.jobPollTimer);
    appState.jobPollTimer = 0;
  }
  appState.jobPollToken += 1;
}

function normalizeJobPayload(payload, kindOverride = "") {
  return {
    job_id: String(payload?.job_id || ""),
    kind: String(payload?.kind || kindOverride || ""),
    status: String(payload?.status || "UNKNOWN"),
    status_raw: String(payload?.status_raw || "").trim(),
    progress: Number(payload?.progress ?? 0),
    message: String(payload?.message || ""),
    error: String(payload?.error || ""),
    stage_name: String(payload?.stage_name || ""),
    error_code: String(payload?.error_code || ""),
    retryable: Boolean(payload?.retryable),
    created_at: String(payload?.created_at || payload?.accepted_at || ""),
    updated_at: String(payload?.updated_at || ""),
  };
}

function setJobState(job) {
  appState.job = job ? { ...job } : null;
  renderJobStatus();
  applyActionPermissions();
}

function makeBadge(label, level) {
  return `<span class="mlops-badge ${levelClass(level)}">${escapeHtml(label)}</span>`;
}

function formatPercentRatio(value, digits = 1) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${(num * 100).toFixed(digits)}%`;
}

function formatSignedPercentRatio(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  const pct = (num * 100).toFixed(digits);
  if (num > 0) return `+${pct}%`;
  if (num < 0) return `${pct}%`;
  return `0.${"0".repeat(digits)}%`;
}

function formatScore(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(2);
}

function formatNullableValue(value) {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "number") return String(value);
  return String(value);
}

function formatJobTimestamp(value) {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString("ja-JP", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function makeSummaryCards(cards) {
  return (Array.isArray(cards) ? cards : [])
    .map((item) => `
      <article class="mlops-metric-card${item.action_tab ? " is-actionable" : ""}"${item.action_tab ? ` data-action-tab="${escapeHtml(item.action_tab)}" tabindex="0" role="button"` : ""}>
        <span class="label">${escapeHtml(item.label)}</span>
        <strong>${escapeHtml(item.value)}</strong>
        ${item.sub ? `<span class="hint">${escapeHtml(item.sub)}</span>` : ""}
      </article>
    `)
    .join("");
}

function makeKeyValueGrid(items) {
  return (Array.isArray(items) ? items : [])
    .map((item) => `
      <div class="mlops-kv-item">
        <span class="label">${escapeHtml(item.label)}</span>
        <span>${escapeHtml(item.value)}</span>
      </div>
    `)
    .join("");
}

function makeStackList(items) {
  return (Array.isArray(items) ? items : [])
    .map((item) => `
      <article class="mlops-stack-item">
        <div class="mlops-stack-item-head">
          ${makeBadge(item.level === "normal" ? "NORMAL" : item.level === "warning" ? "WARNING" : item.level === "error" ? "ERROR" : "INFO", item.level)}
          <strong>${escapeHtml(item.title)}</strong>
        </div>
        <p>${escapeHtml(item.detail)}</p>
      </article>
    `)
    .join("");
}

function makeLogList(items) {
  return (Array.isArray(items) ? items : [])
    .map((item) => `
      <article class="mlops-log-item">
        <div class="mlops-log-top">
          ${makeBadge(item.status || "-", item.level)}
          <span>${escapeHtml(item.time || "-")}</span>
          <span>${escapeHtml(item.stage || "-")}</span>
        </div>
        <p>${escapeHtml(item.message || "")}</p>
      </article>
    `)
    .join("");
}

function makeBarList(items, formatter = (value) => formatPercentRatio(value, 1)) {
  const rows = Array.isArray(items) ? items : [];
  const maxValue = Math.max(...rows.map((item) => Number(item.value) || 0), 0.01);
  return rows
    .map((item) => `
      <div class="mlops-bar-row">
        <div class="mlops-bar-meta">
          <span>${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(formatter(item.value))}</strong>
        </div>
        <div class="mlops-bar-track">
          <div class="mlops-bar-fill" style="width:${Math.max(10, ((Number(item.value) || 0) / maxValue) * 100)}%"></div>
        </div>
      </div>
    `)
    .join("");
}

function buildLineChartSvg(labels, seriesList) {
  const width = 620;
  const height = 240;
  const padLeft = 32;
  const padRight = 14;
  const padTop = 18;
  const padBottom = 28;
  const values = seriesList.flatMap((series) => series.values || []);
  if (values.length === 0) {
    return `<div class="pf-empty">No chart data</div>`;
  }
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const valueSpan = Math.max(0.001, maxValue - minValue);
  const xFor = (index) => {
    if (labels.length <= 1) return padLeft;
    return padLeft + ((width - padLeft - padRight) * index) / (labels.length - 1);
  };
  const yFor = (value) => padTop + ((maxValue - value) / valueSpan) * (height - padTop - padBottom);
  const gridLines = [0, 0.25, 0.5, 0.75, 1]
    .map((ratio) => {
      const y = padTop + ((height - padTop - padBottom) * ratio);
      return `<line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" class="mlops-chart-gridline"></line>`;
    })
    .join("");
  const paths = seriesList
    .map((series) => {
      const path = (series.values || [])
        .map((value, index) => `${index === 0 ? "M" : "L"} ${xFor(index)} ${yFor(value)}`)
        .join(" ");
      const points = (series.values || [])
        .map((value, index) => `<circle cx="${xFor(index)}" cy="${yFor(value)}" r="3" fill="${series.color}"></circle>`)
        .join("");
      return `<path d="${path}" fill="none" stroke="${series.color}" stroke-width="2.2"></path>${points}`;
    })
    .join("");
  const xLabels = labels
    .map((label, index) => `<text x="${xFor(index)}" y="${height - 8}" text-anchor="middle" class="mlops-chart-label">${escapeHtml(label)}</text>`)
    .join("");
  const legends = seriesList
    .map((series, index) => `
      <g transform="translate(${padLeft + (index * 180)}, 10)">
        <rect width="12" height="3" rx="2" fill="${series.color}"></rect>
        <text x="18" y="4" class="mlops-chart-label" dominant-baseline="middle">${escapeHtml(series.label)}</text>
      </g>
    `)
    .join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" class="mlops-line-chart" role="img" aria-label="line chart">
      ${gridLines}
      ${paths}
      ${xLabels}
      ${legends}
    </svg>
  `;
}

function buildGroupedBarChartSvg(labels, seriesList) {
  const width = 640;
  const height = 260;
  const padLeft = 28;
  const padRight = 18;
  const padTop = 18;
  const padBottom = 42;
  const groups = Array.isArray(labels) ? labels : [];
  const series = Array.isArray(seriesList) ? seriesList : [];
  const values = series.flatMap((item) => Array.isArray(item.values) ? item.values : []);
  if (groups.length === 0 || values.length === 0) {
    return `<div class="pf-empty">No chart data</div>`;
  }
  const maxValue = Math.max(...values.map((value) => Number(value) || 0), 1);
  const innerWidth = width - padLeft - padRight;
  const groupWidth = innerWidth / Math.max(groups.length, 1);
  const barGap = 6;
  const seriesWidth = Math.min(groupWidth - 10, Math.max(18, groupWidth - 22));
  const barWidth = Math.max(12, Math.min(30, (seriesWidth - (Math.max(series.length - 1, 0) * barGap)) / Math.max(series.length, 1)));
  const yFor = (value) => padTop + ((maxValue - value) / maxValue) * (height - padTop - padBottom);
  const gridLines = [0, 0.25, 0.5, 0.75, 1]
    .map((ratio) => {
      const y = padTop + ((height - padTop - padBottom) * ratio);
      return `<line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" class="mlops-chart-gridline"></line>`;
    })
    .join("");
  const bars = groups
    .map((label, groupIndex) => {
      const startX = padLeft + (groupIndex * groupWidth) + ((groupWidth - ((barWidth * series.length) + (barGap * Math.max(series.length - 1, 0)))) / 2);
      const rects = series
        .map((item, seriesIndex) => {
          const value = Number(item.values?.[groupIndex]) || 0;
          const x = startX + (seriesIndex * (barWidth + barGap));
          const y = yFor(value);
          const h = Math.max(1, (height - padBottom) - y);
          return `
            <rect x="${x}" y="${y}" width="${barWidth}" height="${h}" rx="4" fill="${item.color || "#58a6ff"}">
              <title>${escapeHtml(item.label || "-")} / ${escapeHtml(label)} / ${escapeHtml(String(value))}</title>
            </rect>
          `;
        })
        .join("");
      const labelX = padLeft + (groupIndex * groupWidth) + (groupWidth / 2);
      return `${rects}<text x="${labelX}" y="${height - 12}" text-anchor="middle" class="mlops-chart-label">${escapeHtml(label)}</text>`;
    })
    .join("");
  const legends = series
    .map((item, index) => `
      <g transform="translate(${padLeft + (index * 150)}, 10)">
        <rect width="12" height="12" rx="3" fill="${item.color || "#58a6ff"}"></rect>
        <text x="18" y="8" class="mlops-chart-label" dominant-baseline="middle">${escapeHtml(item.label || "-")}</text>
      </g>
    `)
    .join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" class="mlops-line-chart" role="img" aria-label="grouped bar chart">
      ${gridLines}
      ${bars}
      ${legends}
    </svg>
  `;
}

function heatColor(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "rgba(88, 166, 255, 0.12)";
  if (num >= 2) return "rgba(38, 217, 127, 0.30)";
  if (num >= 1) return "rgba(38, 217, 127, 0.18)";
  if (num >= 0) return "rgba(88, 166, 255, 0.18)";
  if (num >= -1) return "rgba(240, 136, 62, 0.18)";
  return "rgba(244, 112, 103, 0.28)";
}

function setBusyAction(action) {
  appState.busyAction = action || "";
  const isBusy = Boolean(appState.busyAction);
  refreshDataBtn.textContent = appState.busyAction === "refresh" ? "更新中..." : "データ更新";
  runInferenceBtn.textContent = appState.busyAction === "run-inference" ? "推論実行中..." : "推論実行";
  createTrainingBtn.textContent = appState.busyAction === "training" ? "集計中..." : "学習ジョブ作成";
  if (runBacktestBtn) {
    runBacktestBtn.textContent = appState.busyAction === "backtest" ? "再計算中..." : "バックテスト再計算";
  }
  exportCsvBtn.textContent = appState.busyAction === "export-csv" ? "CSV出力中..." : "CSV出力";
  exportReportBtn.textContent = appState.busyAction === "export-report" ? "出力中..." : "レポート出力";
  if (!isBusy) {
    exportCsvBtn.textContent = "CSV出力";
    exportReportBtn.textContent = "レポート出力";
  }
  applyActionPermissions();
}

function syncStateFromSnapshot() {
  const snapshot = appState.snapshot;
  if (!snapshot) return;
  appState.filters = { ...appState.filters, ...(snapshot.filters || {}) };
  const dashboardRows = snapshot.dashboard?.rows || [];
  if (!dashboardRows.some((row) => row.code === appState.selectedCode)) {
    appState.selectedCode = dashboardRows[0]?.code || "";
  }
  const modelRows = snapshot.models?.rows || [];
  if (!modelRows.some((row) => row.model_version === appState.selectedModelVersion)) {
    const defaultVersion = snapshot.models?.default_versions?.[appState.filters.model_family];
    appState.selectedModelVersion = defaultVersion || modelRows[0]?.model_version || "";
  }
}

async function fetchSnapshot({ refresh = false } = {}) {
  setBusyAction(refresh ? "refresh" : "");
  try {
    const params = new URLSearchParams({
      universe_filter: appState.filters.universe_filter,
      model_family: appState.filters.model_family,
      feature_set: appState.filters.feature_set,
      cost_buffer: appState.filters.cost_buffer,
      train_window_months: appState.filters.train_window_months,
      gap_days: appState.filters.gap_days,
      valid_window_months: appState.filters.valid_window_months,
      random_seed: appState.filters.random_seed,
      train_note: appState.filters.train_note || "",
      run_note: appState.filters.run_note || "",
      refresh: refresh ? "true" : "false",
    });
    if (appState.filters.prediction_date) {
      params.set("prediction_date", appState.filters.prediction_date);
    }
    const response = await fetch(`/api/ml/stock-page?${params.toString()}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) {
      throw new Error(payload.detail || "ML page snapshot の取得に失敗しました。");
    }
    appState.snapshot = payload;
    syncStateFromSnapshot();
    renderAll();
  } catch (error) {
    const message = error instanceof Error ? error.message : "ML page snapshot の取得に失敗しました。";
    renderErrorState(message);
  } finally {
    setBusyAction("");
  }
}

async function postAction(endpoint, body, busyAction) {
  setBusyAction(busyAction);
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) {
      throw new Error(payload.detail || "操作に失敗しました。");
    }
    appState.snapshot = payload;
    syncStateFromSnapshot();
    renderAll();
  } catch (error) {
    renderErrorState(error instanceof Error ? error.message : "操作に失敗しました。");
  } finally {
    setBusyAction("");
  }
}

async function requestDownload(endpoint, body, busyAction, fallbackMessage) {
  setBusyAction(busyAction);
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) {
      throw new Error(payload.detail || fallbackMessage);
    }
    if (payload.snapshot) {
      appState.snapshot = payload.snapshot;
      syncStateFromSnapshot();
      renderAll();
    }
    downloadTextFile(payload.filename || "download.txt", payload.content || "", busyAction === "export-csv"
      ? "text/csv;charset=utf-8"
      : "application/json;charset=utf-8");
  } catch (error) {
    renderErrorState(error instanceof Error ? error.message : fallbackMessage);
  } finally {
    setBusyAction("");
  }
}

async function refreshSnapshotAfterJob(job) {
  try {
    await fetchSnapshot();
    setJobState(job);
  } catch (error) {
    const message = error instanceof Error ? error.message : "最新スナップショットの再取得に失敗しました。";
    setJobState({
      ...job,
      status: "FAILED",
      error: message,
      message,
    });
    renderErrorState(message);
  }
}

async function pollJobStatus(jobId, kind) {
  stopJobPolling();
  const token = appState.jobPollToken;

  const pollOnce = async () => {
    if (token !== appState.jobPollToken) return;
    try {
      const response = await fetch(`/api/ml/jobs/${encodeURIComponent(jobId)}`);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.ok) {
        throw new Error(payload.detail || "ジョブ状態の取得に失敗しました。");
      }
      const job = normalizeJobPayload(payload, kind);
      setJobState(job);
      if (JOB_TERMINAL_STATUSES.has(job.status)) {
        if (job.status === "SUCCEEDED") {
          await refreshSnapshotAfterJob(job);
        } else {
          renderErrorState(job.error || job.message || `${jobKindLabel(job.kind)} が失敗しました。`);
        }
        return;
      }
      appState.jobPollTimer = window.setTimeout(() => {
        void pollOnce();
      }, 1200);
    } catch (error) {
      const message = error instanceof Error ? error.message : "ジョブ状態の取得に失敗しました。";
      setJobState({
        job_id: jobId,
        kind,
        status: "FAILED",
        status_raw: "failed",
        progress: 0,
        message,
        error: message,
        stage_name: "",
        error_code: "",
        retryable: true,
        created_at: "",
        updated_at: "",
      });
      renderErrorState(message);
    }
  };

  void pollOnce();
}

async function startAsyncJob(endpoint, body, busyAction, kind, fallbackMessage) {
  stopJobPolling();
  setBusyAction(busyAction);
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) {
      const error = new Error(payload.detail || fallbackMessage);
      error.status = response.status;
      throw error;
    }
    const job = normalizeJobPayload(payload, kind);
    setJobState(job);
    await pollJobStatus(job.job_id, job.kind);
    return job;
  } finally {
    setBusyAction("");
  }
}

async function runInferenceAction(confirmRegenerate = false) {
  try {
    await startAsyncJob(
      "/api/ml/predictions/run",
      {
        ...currentFilterPayload(),
        confirm_regenerate: confirmRegenerate,
      },
      "run-inference",
      "stock_prediction_run",
      "推論実行に失敗しました。",
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "推論実行に失敗しました。";
    const status = Number(error?.status || 0);
    if (status === 409 && !confirmRegenerate) {
      const approved = window.confirm(`${message}\n\n再生成しますか？`);
      if (approved) {
        await runInferenceAction(true);
      }
      return;
    }
    renderErrorState(message);
  }
}

function renderErrorState(message) {
  if (globalBadgeEl) {
    globalBadgeEl.className = "mlops-badge is-error";
    globalBadgeEl.textContent = "ERROR";
  }
  if (globalStatusEl) {
    globalStatusEl.textContent = message;
  }
}

function renderHeader() {
  const snapshot = appState.snapshot;
  if (!snapshot) return;
  headerUpdatedEl.textContent = snapshot.header?.updated_at || "-";
  headerEnvEl.textContent = snapshot.header?.env || "dev";
}

function actionPermission(actionKey) {
  return appState.snapshot?.permissions?.actions?.[actionKey] || { allowed: true, reason: "" };
}

function applyActionPermissions() {
  const isBusy = Boolean(appState.busyAction);
  const jobActive = hasActiveJob();
  const buttonSpecs = [
    { button: refreshDataBtn, actionKey: "refresh_data" },
    { button: runInferenceBtn, actionKey: "run_inference" },
    { button: createTrainingBtn, actionKey: "create_training_job" },
    { button: runBacktestBtn, actionKey: "run_backtest" },
    { button: exportReportBtn, actionKey: "export_report" },
    { button: exportCsvBtn, actionKey: "export_csv" },
  ];
  buttonSpecs.forEach(({ button, actionKey }) => {
    if (!button) return;
    const permission = actionPermission(actionKey);
    const blockedByJob = jobActive && !["export_report", "export_csv"].includes(actionKey);
    button.disabled = isBusy || blockedByJob || !permission.allowed;
    button.title = !permission.allowed
      ? permission.reason || ""
      : blockedByJob
        ? "別のジョブが実行中です。完了後に再度実行してください。"
        : "";
  });
  if (adoptModelBtn) {
    const selected = getSelectedModelRow();
    const permission = actionPermission("adopt_model");
    const modelEligible = Boolean(selected && selected.adoptable && selected.status !== "adopted");
    adoptModelBtn.disabled = isBusy || jobActive || !permission.allowed || !modelEligible;
    adoptModelBtn.title = !permission.allowed
      ? permission.reason || ""
      : jobActive
        ? "別のジョブが実行中です。完了後に採用切替してください。"
      : modelEligible
        ? ""
        : selected
          ? selected.status === "adopted"
            ? "採用中モデルは切替不要です。"
            : (selected.adopt_blockers || []).join(" / ") || "このモデルは現在採用できません。"
          : "";
  }
}

function renderFilters() {
  const snapshot = appState.snapshot;
  if (!snapshot) return;
  const options = snapshot.filter_options || {};
  const renderOptions = (selectEl, rows, currentValue) => {
    if (!selectEl) return;
    selectEl.innerHTML = (Array.isArray(rows) ? rows : [])
      .map((item) => `<option value="${escapeHtml(item.value)}"${item.value === currentValue ? " selected" : ""}>${escapeHtml(item.label)}</option>`)
      .join("");
  };
  renderOptions(predictionDateEl, options.prediction_dates, appState.filters.prediction_date);
  renderOptions(universeFilterEl, options.universe_filters, appState.filters.universe_filter);
  renderOptions(modelFamilyEl, options.model_families, appState.filters.model_family);
  renderOptions(featureSetEl, options.feature_sets, appState.filters.feature_set);
  renderOptions(costBufferEl, options.cost_buffers, appState.filters.cost_buffer);
  runNoteEl.value = appState.filters.run_note || "";
}

function renderTrainingInputs() {
  const snapshot = appState.snapshot;
  if (!snapshot) return;
  const options = snapshot.filter_options || {};
  const renderOptions = (selectEl, rows, currentValue) => {
    if (!selectEl) return;
    selectEl.innerHTML = (Array.isArray(rows) ? rows : [])
      .map((item) => `<option value="${escapeHtml(item.value)}"${item.value === currentValue ? " selected" : ""}>${escapeHtml(item.label)}</option>`)
      .join("");
  };
  renderOptions(trainWindowMonthsEl, options.train_window_months, appState.filters.train_window_months);
  renderOptions(gapDaysEl, options.gap_days, appState.filters.gap_days);
  renderOptions(validWindowMonthsEl, options.valid_window_months, appState.filters.valid_window_months);
  if (randomSeedEl) {
    randomSeedEl.value = appState.filters.random_seed || "42";
  }
  if (trainNoteEl) {
    trainNoteEl.value = appState.filters.train_note || "";
  }
}

function renderGlobalStatus() {
  const snapshot = appState.snapshot;
  if (!snapshot) return;
  const status = snapshot.global_status || {};
  globalBadgeEl.className = `mlops-badge ${levelClass(status.level)}`;
  globalBadgeEl.textContent = status.badge || "INFO";
  globalStatusEl.textContent = status.text || "";
  const role = snapshot.permissions?.role || "-";
  if (roleBadgeEl) {
    roleBadgeEl.textContent = `role: ${role}`;
    roleBadgeEl.className = `pill ${role === "admin" ? "chip-green" : role === "analyst" ? "chip-amber" : "chip-cyan"}`;
    roleBadgeEl.title = role === "admin"
      ? "推論実行・データ更新・採用モデル切替が可能です。"
      : role === "analyst"
        ? "学習ジョブ作成・バックテスト・レポート出力が可能です。"
        : "viewer は閲覧と CSV 出力のみ可能です。";
  }
  applyActionPermissions();
}

function renderJobStatus() {
  const job = appState.job;
  if (!jobBadgeEl || !jobStatusEl) return;
  if (!job) {
    jobBadgeEl.className = "mlops-badge is-unknown";
    jobBadgeEl.textContent = "IDLE";
    jobStatusEl.textContent = "ジョブ待機中。固定アクションから実行すると、ここに進捗を表示します。";
    return;
  }
  const level = jobLevel(job.status);
  const details = [
    jobKindLabel(job.kind),
    Number.isFinite(job.progress) ? `${Math.max(0, Math.round(job.progress))}%` : "",
    job.stage_name ? `stage=${job.stage_name}` : "",
    job.updated_at ? `updated ${formatJobTimestamp(job.updated_at)}` : "",
  ].filter(Boolean);
  jobBadgeEl.className = `mlops-badge ${levelClass(level)}`;
  jobBadgeEl.textContent = job.status || "UNKNOWN";
  jobStatusEl.textContent = [job.message || `${jobKindLabel(job.kind)} を実行しています。`, details.join(" / ")].filter(Boolean).join(" ");
}

function renderSidebarStatus() {
  const snapshot = appState.snapshot;
  if (!snapshot) return;
  const items = Array.isArray(snapshot.sidebar_status) ? snapshot.sidebar_status : [];
  sidebarStatusEl.innerHTML = items
    .map((item) => `
      <article class="mlops-status-card">
        <div class="mlops-status-card-top">
          <span class="label">${escapeHtml(item.label)}</span>
          ${makeBadge(item.badge?.label || "-", item.badge?.level || "unknown")}
        </div>
        <strong>${escapeHtml(item.value)}</strong>
        <p class="hint">${escapeHtml(item.note || "")}</p>
      </article>
    `)
    .join("");
}

function renderTabs() {
  tabs.forEach((tab) => {
    const active = tab.dataset.tab === appState.activeTab;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  tabPanels.forEach((panel) => {
    panel.hidden = panel.dataset.tabPanel !== appState.activeTab;
  });
}

function getDashboardRows() {
  const rows = appState.snapshot?.dashboard?.rows || [];
  const query = appState.search.trim().toLowerCase();
  if (!query) return rows;
  return rows.filter((row) =>
    String(row.code || "").toLowerCase().includes(query)
    || String(row.company_name || "").toLowerCase().includes(query)
    || String(row.sector || "").toLowerCase().includes(query),
  );
}

function getSelectedDashboardRow() {
  const rows = getDashboardRows();
  return rows.find((row) => row.code === appState.selectedCode) || rows[0] || null;
}

function renderDashboard() {
  const dashboard = appState.snapshot?.dashboard;
  if (!dashboard) return;
  dashboardSummaryEl.innerHTML = makeSummaryCards(dashboard.summary_cards);
  dashboardCaptionEl.textContent = dashboard.caption || "";
  dashboardFootnoteEl.textContent = dashboard.footnote || "";
  const rows = getDashboardRows();
  if (!rows.some((row) => row.code === appState.selectedCode)) {
    appState.selectedCode = rows[0]?.code || "";
  }
  const selected = getSelectedDashboardRow();
  if (rows.length === 0) {
    predictionBodyEl.innerHTML = `<tr><td colspan="7" class="pf-empty">該当する銘柄がありません。</td></tr>`;
  } else {
    predictionBodyEl.innerHTML = rows
      .map((row) => `
        <tr data-code="${escapeHtml(row.code)}"${row.code === selected?.code ? " class=\"is-selected\"" : ""}>
          <td>${escapeHtml(row.code)}</td>
          <td>${escapeHtml(row.company_name)}</td>
          <td>${escapeHtml(row.sector)}</td>
          <td>${escapeHtml(formatPercentRatio(row.prob_up, 1))}</td>
          <td>${escapeHtml(formatScore(row.score_cls))}</td>
          <td>${escapeHtml(formatSignedPercentRatio(row.expected_return, 2))}</td>
          <td>${escapeHtml((row.warnings || []).join(" / ") || "-")}</td>
        </tr>
      `)
      .join("");
  }
  if (!selected) {
    detailHeadEl.innerHTML = "<strong>銘柄未選択</strong>";
    detailMetricsEl.innerHTML = "";
    featureContribEl.innerHTML = "";
    detailNotesEl.innerHTML = "";
  } else {
    detailHeadEl.innerHTML = `
      <div>
        <span class="label">選択銘柄</span>
        <strong>${escapeHtml(selected.code)} / ${escapeHtml(selected.company_name)}</strong>
      </div>
      <div>${makeBadge((selected.warnings || []).length ? "WARNING" : "NORMAL", (selected.warnings || []).length ? "warning" : "normal")}</div>
    `;
    detailMetricsEl.innerHTML = (selected.recent_metrics || [])
      .map((item) => `
        <article class="mlops-mini-metric">
          <span class="label">${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(item.value)}</strong>
        </article>
      `)
      .join("")
      + `
        <article class="mlops-mini-metric">
          <span class="label">prob_up / score_cls</span>
          <strong>${escapeHtml(formatPercentRatio(selected.prob_up, 1))} / ${escapeHtml(formatScore(selected.score_cls))}</strong>
        </article>
      `;
    const contrib = selected.feature_contrib || [];
    const maxAbs = Math.max(...contrib.map((item) => Math.abs(Number(item.value) || 0)), 0.01);
    featureContribEl.innerHTML = contrib
      .map((item) => {
        const value = Number(item.value) || 0;
        return `
          <div class="mlops-feature-row">
            <div class="mlops-feature-meta">
              <span>${escapeHtml(item.name)}</span>
              <strong>${value >= 0 ? "+" : ""}${escapeHtml(value.toFixed(2))}</strong>
            </div>
            <div class="mlops-feature-track">
              <div class="mlops-feature-fill ${value >= 0 ? "is-positive" : "is-negative"}" style="width:${(Math.abs(value) / maxAbs) * 100}%"></div>
            </div>
          </div>
        `;
      })
      .join("");
    detailNotesEl.innerHTML = [
      selected.event_proximity,
      selected.score_delta,
      selected.note,
      (selected.warnings || []).length ? `警告: ${(selected.warnings || []).join(" / ")}` : "警告なし",
    ].map((item) => `<p>${escapeHtml(item)}</p>`).join("");
  }
  sectorScoresEl.innerHTML = makeBarList(dashboard.sector_scores || []);
  alertListEl.innerHTML = makeStackList(dashboard.alerts || []);
  inferenceLogEl.innerHTML = makeLogList(dashboard.logs || []);
}

function renderTrain() {
  const train = appState.snapshot?.train;
  if (!train) return;
  trainConfigEl.innerHTML = makeKeyValueGrid(train.config_items || []);
  trainRulesEl.innerHTML = (train.rules || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  trainSummaryEl.innerHTML = makeSummaryCards(train.summary_cards || []);
  trainCompareBodyEl.innerHTML = (train.compare_rows || [])
    .map((row) => `
      <tr>
        <td>${escapeHtml(row.model)}</td>
        <td>${escapeHtml(Number(row.roc_auc || 0).toFixed(3))}</td>
        <td>${escapeHtml(Number(row.pr_auc || 0).toFixed(3))}</td>
        <td>${escapeHtml(Number(row.balanced_accuracy || 0).toFixed(3))}</td>
        <td>${escapeHtml(Number(row.hit_ratio || 0).toFixed(3))}</td>
        <td>${escapeHtml(row.train_time || "-")}</td>
        <td>${escapeHtml(row.inference_time || "-")}</td>
        <td>${escapeHtml(String(row.features || "-"))}</td>
        <td>${escapeHtml(row.missing_rate || "-")}</td>
      </tr>
    `)
    .join("");
  trainAcceptanceEl.innerHTML = makeStackList(train.acceptance || []);
  foldBodyEl.innerHTML = (train.folds || [])
    .map((row) => `
      <tr>
        <td>${escapeHtml(row.fold)}</td>
        <td>${escapeHtml(row.train)}</td>
        <td>${escapeHtml(row.gap)}</td>
        <td>${escapeHtml(row.valid)}</td>
        <td>${escapeHtml(row.samples)}</td>
        <td>${escapeHtml(Number(row.lgbm_roc_auc || 0).toFixed(3))}</td>
        <td>${escapeHtml(Number(row.logreg_roc_auc || 0).toFixed(3))}</td>
      </tr>
    `)
    .join("");
  foldChartEl.innerHTML = buildLineChartSvg(
    (train.folds || []).map((item) => item.fold.replace("Fold ", "F")),
    [
      { label: "LightGBM", color: "#39d2c0", values: (train.folds || []).map((item) => item.lgbm_roc_auc) },
      { label: "LogReg", color: "#58a6ff", values: (train.folds || []).map((item) => item.logreg_roc_auc) },
    ],
  );
  const distribution = train.distribution || [];
  const maxValue = Math.max(...distribution.map((item) => Number(item.value) || 0), 1);
  scoreDistributionEl.innerHTML = distribution
    .map((item) => `
      <div class="mlops-dist-card">
        <span>${escapeHtml(item.label)}</span>
        <div class="mlops-dist-track">
          <div class="mlops-dist-fill" style="height:${Math.max(10, ((Number(item.value) || 0) / maxValue) * 120)}px"></div>
        </div>
        <strong>${escapeHtml(String(item.value))}</strong>
      </div>
    `)
    .join("");
}

function renderBacktest() {
  const backtest = appState.snapshot?.backtest;
  if (!backtest) return;
  backtestSettingsEl.innerHTML = makeKeyValueGrid(backtest.settings || []);
  backtestCompareBodyEl.innerHTML = (backtest.compare_rows || [])
    .map((row) => `
      <tr>
        <td>${escapeHtml(row.model)}</td>
        <td>${escapeHtml(row.cagr)}</td>
        <td>${escapeHtml(row.sharpe)}</td>
        <td>${escapeHtml(row.mdd)}</td>
        <td>${escapeHtml(row.turnover)}</td>
        <td>${escapeHtml(row.win_rate)}</td>
        <td>${escapeHtml(row.unable)}</td>
      </tr>
    `)
    .join("");
  backtestSummaryEl.innerHTML = makeSummaryCards(backtest.summary_cards || []);
  equityChartEl.innerHTML = buildLineChartSvg(backtest.equity_labels || [], backtest.equity_series || []);
  monthlyHeatmapEl.innerHTML = (backtest.monthly_returns || [])
    .map((item) => `
      <div class="mlops-heat-cell" style="background:${heatColor(item.value)}">
        <span>${escapeHtml(item.month)}</span>
        <strong>${item.value > 0 ? "+" : ""}${escapeHtml(Number(item.value).toFixed(1))}%</strong>
      </div>
    `)
    .join("");
  backtestDailyDistributionEl.innerHTML = buildGroupedBarChartSvg(
    backtest.daily_return_distribution?.labels || [],
    backtest.daily_return_distribution?.series || [],
  );
  backtestExceptionBodyEl.innerHTML = (backtest.exceptions || [])
    .map((item) => `
      <tr>
        <td>${escapeHtml(item.date)}</td>
        <td>${escapeHtml(item.type)}</td>
        <td>${escapeHtml(item.count)}</td>
        <td>${escapeHtml(item.impact)}</td>
        <td>${escapeHtml(item.note)}</td>
      </tr>
    `)
    .join("");
}

function getSelectedModelRow() {
  const rows = appState.snapshot?.models?.rows || [];
  return rows.find((row) => row.model_version === appState.selectedModelVersion) || rows[0] || null;
}

function renderModels() {
  const models = appState.snapshot?.models;
  if (!models) return;
  const rows = models.rows || [];
  const selected = getSelectedModelRow();
  modelBodyEl.innerHTML = rows
    .map((row) => `
      <tr data-model-version="${escapeHtml(row.model_version)}"${row.model_version === selected?.model_version ? " class=\"is-selected\"" : ""}>
        <td>${escapeHtml(row.model_version)}</td>
        <td>${escapeHtml(row.feature_version)}</td>
        <td>${escapeHtml(row.family)}</td>
        <td>${makeBadge(
          row.status === "adopted"
            ? (row.adoptable ? "採用中" : "採用中要確認")
            : row.adoptable
              ? (row.status === "candidate" ? "候補" : row.status)
              : "採用不可",
          row.status === "adopted"
            ? (row.adoptable ? "normal" : "error")
            : row.adoptable
              ? (row.status === "candidate" ? "warning" : "unknown")
              : "error",
        )}</td>
        <td>${escapeHtml(row.summary_metrics)}</td>
        <td>${escapeHtml((row.warnings || []).join(" / ") || "-")}</td>
      </tr>
    `)
    .join("");
  if (!selected) {
    modelDetailHeadEl.innerHTML = "<strong>モデル未選択</strong>";
    modelDetailGridEl.innerHTML = "";
    modelDecisionEl.innerHTML = "";
    modelExplainabilityEl.innerHTML = "";
    modelAuditEl.innerHTML = "";
    adoptModelBtn.disabled = true;
    adoptModelBtn.title = "";
    return;
  }
  modelDetailHeadEl.innerHTML = `
    <div>
      <span class="label">model_version</span>
      <strong>${escapeHtml(selected.model_version)}</strong>
    </div>
    <div>
      ${makeBadge(
        selected.status === "adopted"
          ? (selected.adoptable ? "採用中" : "採用中要確認")
          : selected.adoptable
            ? "候補"
            : "採用不可",
        selected.status === "adopted"
          ? (selected.adoptable ? "normal" : "error")
          : selected.adoptable
            ? "warning"
            : "error",
      )}
      ${makeBadge(selected.adoptable ? "ADOPTABLE" : "BLOCKED", selected.adoptable ? "normal" : "error")}
    </div>
  `;
  modelDetailGridEl.innerHTML = makeKeyValueGrid([...(selected.train_conditions || []), ...(selected.eval_conditions || [])]);
  modelDecisionEl.innerHTML = makeStackList(selected.decision || []);
  modelExplainabilityEl.innerHTML = makeStackList(selected.explainability || []);
  modelAuditEl.innerHTML = (selected.audit || []).map((item) => `<p>${escapeHtml(item)}</p>`).join("");
  applyActionPermissions();
}

function renderOps() {
  const ops = appState.snapshot?.ops;
  if (!ops) return;
  pipelineGridEl.innerHTML = (ops.pipeline || [])
    .map((item) => `
      <article class="mlops-stage-card">
        <div class="mlops-stage-top">
          <span class="label">${escapeHtml(item.title)}</span>
          ${makeBadge(item.level === "normal" ? "正常" : item.level === "warning" ? "警告" : item.level === "error" ? "異常" : "不明", item.level)}
        </div>
        <strong>${escapeHtml(item.name)}</strong>
        <p>${escapeHtml(item.detail)}</p>
        <span class="hint">${escapeHtml(item.updated)}</span>
      </article>
    `)
    .join("");
  opsSummaryEl.innerHTML = makeSummaryCards(ops.summary_cards || []);
  opsAlertsEl.innerHTML = makeStackList(ops.alerts || []);
  opsLogsEl.innerHTML = makeLogList(ops.logs || []);
  const coverageBreakdown = Array.isArray(ops.coverage_breakdown) ? ops.coverage_breakdown : [];
  opsCoverageBreakdownEl.innerHTML = coverageBreakdown.length
    ? coverageBreakdown
      .map((item) => `
        <div class="mlops-kv-item">
          <span class="label">${escapeHtml(item.label || "-")}</span>
          <span>${escapeHtml(`${Number(item.count) || 0}件`)}</span>
          <span class="hint">${escapeHtml(item.detail || "")}</span>
        </div>
      `)
      .join("")
    : `<div class="pf-empty">除外理由はありません。</div>`;
  const drift = ops.score_drift_distribution || {};
  opsDriftChartEl.innerHTML = buildGroupedBarChartSvg(drift.labels || [], drift.series || []);
  const psi = Number(drift.psi);
  const meanShift = Number(drift.mean_shift);
  const references = [];
  if (Number.isFinite(psi)) {
    references.push(`PSI ${psi.toFixed(3)}`);
  }
  if (Number.isFinite(meanShift)) {
    references.push(`平均スコア差 ${meanShift >= 0 ? "+" : ""}${meanShift.toFixed(3)}`);
  }
  opsDriftNoteEl.textContent = [drift.note || "", references.join(" / ")].filter(Boolean).join(" ");
}

function renderAll() {
  renderHeader();
  renderFilters();
  renderTrainingInputs();
  renderGlobalStatus();
  renderJobStatus();
  renderSidebarStatus();
  renderTabs();
  renderDashboard();
  renderTrain();
  renderBacktest();
  renderModels();
  renderOps();
}

function currentFilterPayload() {
  return {
    prediction_date: appState.filters.prediction_date || null,
    universe_filter: appState.filters.universe_filter,
    model_family: appState.filters.model_family,
    feature_set: appState.filters.feature_set,
    cost_buffer: Number(appState.filters.cost_buffer || 0),
    train_window_months: Number(appState.filters.train_window_months || 12),
    gap_days: Number(appState.filters.gap_days || 5),
    valid_window_months: Number(appState.filters.valid_window_months || 1),
    random_seed: Number(appState.filters.random_seed || 42),
    train_note: appState.filters.train_note || "",
    run_note: appState.filters.run_note || "",
  };
}

function currentExportPayload() {
  return {
    ...currentFilterPayload(),
    search_query: appState.search || "",
  };
}

function downloadTextFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(url);
}

async function exportCsv() {
  await requestDownload(
    "/api/ml/stock-page/actions/export-csv",
    currentExportPayload(),
    "export-csv",
    "CSV 出力に失敗しました。",
  );
}

async function exportReport() {
  await requestDownload(
    "/api/ml/stock-page/actions/export-report",
    currentExportPayload(),
    "export-report",
    "レポート出力に失敗しました。",
  );
}

function closeInlineHelp(except = null) {
  document.querySelectorAll(".mlops-inline-help[open]").forEach((element) => {
    if (element !== except) {
      element.open = false;
    }
  });
}

function bindInlineHelp() {
  document.querySelectorAll(".mlops-inline-help").forEach((element) => {
    element.addEventListener("toggle", () => {
      if (element.open) {
        closeInlineHelp(element);
      }
    });
  });
  document.addEventListener("click", (event) => {
    if (event.target instanceof Element && event.target.closest(".mlops-inline-help")) {
      return;
    }
    closeInlineHelp();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeInlineHelp();
    }
  });
}

function bindEvents() {
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      appState.activeTab = tab.dataset.tab || "dashboard";
      renderTabs();
    });
  });

  flowTargets.forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      appState.activeTab = link.dataset.tabTarget || "dashboard";
      renderTabs();
      document.getElementById("mlops-tabs")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  predictionDateEl?.addEventListener("change", async (event) => {
    appState.filters.prediction_date = event.target.value;
    await fetchSnapshot();
  });

  universeFilterEl?.addEventListener("change", async (event) => {
    appState.filters.universe_filter = event.target.value;
    await fetchSnapshot();
  });

  modelFamilyEl?.addEventListener("change", async (event) => {
    appState.filters.model_family = event.target.value;
    await fetchSnapshot();
  });

  featureSetEl?.addEventListener("change", async (event) => {
    appState.filters.feature_set = event.target.value;
    await fetchSnapshot();
  });

  costBufferEl?.addEventListener("change", async (event) => {
    appState.filters.cost_buffer = event.target.value;
    await fetchSnapshot();
  });

  trainWindowMonthsEl?.addEventListener("change", async (event) => {
    appState.filters.train_window_months = event.target.value;
    await fetchSnapshot();
  });

  gapDaysEl?.addEventListener("change", async (event) => {
    appState.filters.gap_days = event.target.value;
    await fetchSnapshot();
  });

  validWindowMonthsEl?.addEventListener("change", async (event) => {
    appState.filters.valid_window_months = event.target.value;
    await fetchSnapshot();
  });

  randomSeedEl?.addEventListener("change", async (event) => {
    appState.filters.random_seed = event.target.value || "42";
    await fetchSnapshot();
  });

  trainNoteEl?.addEventListener("input", (event) => {
    appState.filters.train_note = event.target.value.slice(0, 500);
  });

  runNoteEl?.addEventListener("input", (event) => {
    appState.filters.run_note = event.target.value.slice(0, 200);
  });

  predictionSearchEl?.addEventListener("input", (event) => {
    appState.search = event.target.value;
    renderDashboard();
  });

  predictionBodyEl?.addEventListener("click", (event) => {
    const row = event.target instanceof Element ? event.target.closest("tr[data-code]") : null;
    if (!row) return;
    appState.selectedCode = row.dataset.code || "";
    renderDashboard();
  });

  dashboardSummaryEl?.addEventListener("click", (event) => {
    const card = event.target instanceof Element ? event.target.closest("[data-action-tab]") : null;
    if (!card) return;
    appState.activeTab = card.dataset.actionTab || "dashboard";
    renderTabs();
    document.getElementById("mlops-tabs")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  dashboardSummaryEl?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const card = event.target instanceof Element ? event.target.closest("[data-action-tab]") : null;
    if (!card) return;
    event.preventDefault();
    appState.activeTab = card.dataset.actionTab || "dashboard";
    renderTabs();
    document.getElementById("mlops-tabs")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  modelBodyEl?.addEventListener("click", (event) => {
    const row = event.target instanceof Element ? event.target.closest("tr[data-model-version]") : null;
    if (!row) return;
    appState.selectedModelVersion = row.dataset.modelVersion || "";
    renderModels();
  });

  refreshDataBtn?.addEventListener("click", async () => {
    await postAction("/api/ml/stock-page/actions/refresh", currentFilterPayload(), "refresh");
  });

  runInferenceBtn?.addEventListener("click", async () => {
    await runInferenceAction();
  });

  createTrainingBtn?.addEventListener("click", async () => {
    appState.activeTab = "train";
    renderTabs();
    try {
      await startAsyncJob(
        "/api/ml/training/jobs",
        currentFilterPayload(),
        "training",
        "stock_training_job",
        "学習ジョブ作成に失敗しました。",
      );
    } catch (error) {
      renderErrorState(error instanceof Error ? error.message : "学習ジョブ作成に失敗しました。");
    }
  });

  runBacktestBtn?.addEventListener("click", async () => {
    appState.activeTab = "backtest";
    renderTabs();
    try {
      await startAsyncJob(
        "/api/ml/backtests/run",
        currentFilterPayload(),
        "backtest",
        "stock_backtest_run",
        "バックテスト再計算に失敗しました。",
      );
    } catch (error) {
      renderErrorState(error instanceof Error ? error.message : "バックテスト再計算に失敗しました。");
    }
  });

  adoptModelBtn?.addEventListener("click", async () => {
    const selected = getSelectedModelRow();
    if (!selected) return;
    await postAction("/api/ml/stock-page/actions/adopt-model", {
      ...currentFilterPayload(),
      model_version: selected.model_version,
    }, "adopt");
  });

  exportCsvBtn?.addEventListener("click", async () => {
    await exportCsv();
  });
  exportReportBtn?.addEventListener("click", async () => {
    await exportReport();
  });
}

async function init() {
  bindInlineHelp();
  bindEvents();
  await fetchSnapshot();
}

init();
