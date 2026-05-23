(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const DEFAULT_MOVING_AVERAGES = [
    { key: "short", label: "MA5", window: 5 },
    { key: "mid", label: "MA20", window: 20 },
  ];
  const DETAIL_CHART_INTERVAL = "5m";

  const state = {
    selectedMarket: "us",
    selectedMode: "intraday",
    selectedTradeMode: "long_only",
    activeChartInterval: "15m",
    loading: false,
    session: null,
    revealedCount: 0,
    position: null,
    realized: 0,
    trades: [],
    closedPositions: [],
    actedIndexes: new Set(),
    gameDone: false,
  };

  const el = {
    marketUs: $("dtg-market-us"),
    marketJp: $("dtg-market-jp"),
    modeIntraday: $("dtg-mode-intraday"),
    modeDaily: $("dtg-mode-daily"),
    tradeModeLongOnly: $("dtg-trade-mode-long-only"),
    tradeModeLongShort: $("dtg-trade-mode-long-short"),
    newGame: $("dtg-new-game"),
    next: $("dtg-next"),
    long: $("dtg-long"),
    short: $("dtg-short"),
    close: $("dtg-close"),
    message: $("dtg-message"),
    symbol: $("dtg-symbol"),
    date: $("dtg-date"),
    market: $("dtg-market"),
    bars: $("dtg-bars"),
    interval: $("dtg-interval"),
    source: $("dtg-source"),
    rule: $("dtg-rule"),
    clock: $("dtg-clock"),
    progressLabel: $("dtg-progress-label"),
    progressStrip: $("dtg-progress-strip-text"),
    chartFrameSwitch: $("dtg-chart-frame-switch"),
    chartFramePrimary: $("dtg-chart-frame-primary"),
    chartFrame5m: $("dtg-chart-frame-5m"),
    chart: $("dtg-chart"),
    chartEmpty: $("dtg-chart-empty"),
    currentTime: $("dtg-current-time"),
    currentPrice: $("dtg-current-price"),
    position: $("dtg-position"),
    unrealized: $("dtg-unrealized"),
    realized: $("dtg-realized"),
    score: $("dtg-score"),
    entry: $("dtg-entry"),
    last: $("dtg-last"),
    side: $("dtg-side"),
    tradeCount: $("dtg-trade-count"),
    tradesBody: $("dtg-trades-body"),
    modeSubtitle: $("dtg-mode-subtitle"),
    chartTitle: $("dtg-chart-title"),
    tradeSubtitle: $("dtg-trade-subtitle"),
  };

  function currentCandle() {
    if (!state.session || state.revealedCount <= 0) return null;
    return state.session.candles[state.revealedCount - 1] || null;
  }

  function currentIndex() {
    return state.revealedCount - 1;
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  }

  function formatPrice(value) {
    const numeric = finiteNumber(value);
    if (numeric === null) return "-";
    const digits = priceDigits();
    const symbol = state.session?.currency_symbol || "";
    return `${symbol}${numeric.toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    })}`;
  }

  function priceDigits() {
    const digits = Number(state.session?.price_digits ?? state.session?.currency_digits ?? 2);
    if (!Number.isFinite(digits)) return 2;
    return Math.max(0, Math.min(4, Math.trunc(digits)));
  }

  function formatPnL(value) {
    const numeric = finiteNumber(value);
    if (numeric === null) return "-";
    const sign = numeric > 0 ? "+" : numeric < 0 ? "-" : "";
    return `${sign}${formatPrice(Math.abs(numeric))}`;
  }

  function sideLabel(side) {
    return side === "long" ? "Long" : "Short";
  }

  function positionSign(side) {
    if (side === "long") return 1;
    if (side === "short") return -1;
    return 0;
  }

  function isLongOnlyTradeMode() {
    return state.selectedTradeMode === "long_only";
  }

  function tradeModeLabel(mode = state.selectedTradeMode) {
    const tradeModes = Array.isArray(state.session?.trade_modes) ? state.session.trade_modes : [];
    const match = tradeModes.find((item) => item?.key === mode);
    if (match?.label) return String(match.label);
    return mode === "long_short" ? "Long/Short" : "Long Only";
  }

  function formatScore(value) {
    const numeric = finiteNumber(value);
    if (numeric === null) return "undefined";
    return numeric.toLocaleString(undefined, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    });
  }

  function formatDateRange(session) {
    if (!session) return "-";
    const start = session.start_date || session.date || "";
    const end = session.end_date || start;
    if (!start) return "-";
    return end && end !== start ? `${start} to ${end}` : start;
  }

  function formatSymbol(session) {
    if (!session) return "-";
    return session.symbol_label || session.symbol || "-";
  }

  function isMultiDaySession(session) {
    if (!session) return false;
    if (Array.isArray(session.session_dates)) {
      return new Set(session.session_dates).size > 1;
    }
    return Boolean(session.start_date && session.end_date && session.start_date !== session.end_date);
  }

  function isDailySession(session) {
    return Boolean(session && (session.mode === "daily" || session.interval === "1d"));
  }

  function primaryChartInterval(session) {
    return String(session?.interval || (state.selectedMode === "daily" ? "1d" : "15m"));
  }

  function chartIntervalLabel(interval) {
    if (interval === "1d") return "日足";
    return String(interval || "-");
  }

  function chartTimeframes(session) {
    if (!session) return [];
    const primary = primaryChartInterval(session);
    const frames = Array.isArray(session.chart_timeframes) ? session.chart_timeframes : [];
    const normalized = frames
      .map((frame) => ({
        interval: String(frame?.interval || frame?.key || ""),
        label: String(frame?.label || frame?.interval || frame?.key || ""),
        chartLabel: String(frame?.chart_label || ""),
      }))
      .filter((frame) => frame.interval);
    if (!normalized.some((frame) => frame.interval === primary)) {
      normalized.unshift({
        interval: primary,
        label: chartIntervalLabel(primary),
        chartLabel: String(session.chart_label || ""),
      });
    }
    return normalized;
  }

  function chartCandlesForInterval(session, interval) {
    if (!session) return [];
    const key = String(interval || primaryChartInterval(session));
    const charts = session.chart_candles && typeof session.chart_candles === "object"
      ? session.chart_candles
      : {};
    if (Array.isArray(charts[key])) return charts[key];
    if (key === primaryChartInterval(session) && Array.isArray(session.candles)) return session.candles;
    return [];
  }

  function hasChartInterval(session, interval) {
    return chartCandlesForInterval(session, interval).length > 0;
  }

  function defaultChartInterval(session) {
    const primary = primaryChartInterval(session);
    if (hasChartInterval(session, primary)) return primary;
    return chartTimeframes(session).find((frame) => hasChartInterval(session, frame.interval))?.interval || primary;
  }

  function effectiveChartInterval(session) {
    if (!session) return state.activeChartInterval || primaryChartInterval(session);
    const selected = state.activeChartInterval || defaultChartInterval(session);
    if (hasChartInterval(session, selected)) return selected;
    return defaultChartInterval(session);
  }

  function intervalMinutes(interval) {
    const normalized = String(interval || "").trim().toLowerCase();
    const match = normalized.match(/^(\d+)\s*m(?:in)?$/);
    if (!match) return null;
    const minutes = Number(match[1]);
    return Number.isFinite(minutes) && minutes > 0 ? minutes : null;
  }

  function candleTimestampMs(candle) {
    if (!candle) return null;
    const direct = Date.parse(String(candle.timestamp || ""));
    if (Number.isFinite(direct)) return direct;
    if (candle.date && candle.time) {
      const fallback = Date.parse(`${candle.date}T${candle.time}:00`);
      if (Number.isFinite(fallback)) return fallback;
    }
    return null;
  }

  function chartRevealCutoffMs(session, candle) {
    const base = candleTimestampMs(candle);
    if (base === null) return null;
    const minutes = intervalMinutes(primaryChartInterval(session));
    if (minutes === null) return base;
    return base + minutes * 60 * 1000;
  }

  function revealedChartSeries(session) {
    const interval = effectiveChartInterval(session);
    const allCandles = chartCandlesForInterval(session, interval);
    if (!session || !allCandles.length || state.revealedCount <= 0) {
      return { interval, allCandles, visible: [] };
    }

    const primary = primaryChartInterval(session);
    if (interval === primary) {
      return {
        interval,
        allCandles,
        visible: allCandles.slice(0, state.revealedCount),
      };
    }

    const current = Array.isArray(session.candles) ? session.candles[state.revealedCount - 1] : null;
    const cutoff = chartRevealCutoffMs(session, current);
    if (cutoff !== null) {
      const visible = allCandles.filter((candle) => {
        const timestamp = candleTimestampMs(candle);
        return timestamp !== null && timestamp < cutoff;
      });
      if (visible.length) return { interval, allCandles, visible };
    }

    const primaryMinutes = intervalMinutes(primary);
    const detailMinutes = intervalMinutes(interval);
    const ratio = primaryMinutes && detailMinutes ? primaryMinutes / detailMinutes : 1;
    const count = Math.max(0, Math.ceil(state.revealedCount * ratio));
    return {
      interval,
      allCandles,
      visible: allCandles.slice(0, count),
    };
  }

  function chartTitleText(session) {
    if (!session) return state.selectedMode === "daily" ? "Daily Chart" : "Intraday Chart";
    const interval = effectiveChartInterval(session);
    if (isDailySession(session)) return "Daily Chart";
    return `${chartIntervalLabel(interval)} Chart`;
  }

  function formatCandleTime(candle) {
    if (!candle) return "-";
    if (isDailySession(state.session)) return candle.date || "-";
    const time = candle.time || "-";
    if (!isMultiDaySession(state.session)) return time;
    return candle.date ? `${candle.date} ${time}` : time;
  }

  function formatAxisTime(candle) {
    if (!candle) return "";
    const date = String(candle.date || "");
    const shortDate = /^\d{4}-\d{2}-\d{2}$/.test(date) ? date.slice(5) : date;
    if (isDailySession(state.session)) return shortDate || date;
    const time = candle.time || "";
    if (!isMultiDaySession(state.session)) return time;
    return shortDate ? `${shortDate} ${time}` : time;
  }

  function formatSessionIdentity(session) {
    if (!session) return "-";
    return `${formatSymbol(session)} ${formatDateRange(session)}`;
  }

  function positionPnl(position, price) {
    return position.side === "long"
      ? price - position.entryPrice
      : position.entryPrice - price;
  }

  function setMessage(text, tone = "") {
    el.message.textContent = text;
    el.message.classList.toggle("error", tone === "error");
  }

  function resetGame(session) {
    state.session = session;
    state.activeChartInterval = defaultChartInterval(session);
    state.revealedCount = 0;
    state.position = null;
    state.realized = 0;
    state.trades = [];
    state.closedPositions = [];
    state.actedIndexes = new Set();
    state.gameDone = false;
  }

  async function loadNewGame() {
    state.loading = true;
    updateControls();
    setMessage("Loading yfinance data...");

    try {
      const url = new URL("/api/day-trading-game/session", window.location.origin);
      url.searchParams.set("market", state.selectedMarket);
      url.searchParams.set("mode", state.selectedMode);
      url.searchParams.set("_", Date.now().toString());
      const response = await fetch(url);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.detail || `Request failed (${response.status})`);
      }
      resetGame(payload);
      setMessage("Session ready");
    } catch (error) {
      state.session = null;
      state.gameDone = false;
      setMessage(error.message || "Failed to load session", "error");
    } finally {
      state.loading = false;
      render();
    }
  }

  function revealNext() {
    if (!state.session || state.gameDone) return;
    if (state.revealedCount >= state.session.candles.length) return;

    state.revealedCount += 1;
    const isLast = state.revealedCount >= state.session.candles.length;
    if (isLast) {
      if (state.position) {
        closePosition("Auto Close");
      }
      state.gameDone = true;
      setMessage(`Game complete: ${formatSessionIdentity(state.session)} / Score ${formatScore(finalScoreValue())}`);
    }
    render();
  }

  function openPosition(side) {
    const candle = currentCandle();
    const idx = currentIndex();
    if (!candle || state.gameDone || state.actedIndexes.has(idx)) return;
    if (side === "short" && isLongOnlyTradeMode()) return;

    const price = finiteNumber(candle.execution_price);
    if (price === null) return;

    if (state.position) {
      if (state.position.side === side) return;
      setMessage("Close first before switching sides.", "error");
      render();
      return;
    }

    state.position = {
      side,
      entryPrice: price,
      entryTime: formatCandleTime(candle),
      entryIndex: idx,
    };
    state.actedIndexes.add(idx);
    state.trades.unshift({
      time: formatCandleTime(candle),
      action: sideLabel(side),
      price,
      pnl: null,
    });
    setMessage(`${sideLabel(side)} @ ${formatPrice(price)}`);
    render();
  }

  function closePosition(action = "Close") {
    const candle = currentCandle();
    const idx = currentIndex();
    if (!candle || !state.position || state.actedIndexes.has(idx)) return;

    const price = finiteNumber(candle.execution_price);
    if (price === null) return;

    const closingPosition = state.position;
    const pnl = positionPnl(closingPosition, price);
    state.realized += pnl;
    state.closedPositions.push({
      side: closingPosition.side,
      entryIndex: closingPosition.entryIndex,
      exitIndex: idx,
      pnl,
    });
    state.trades.unshift({
      time: formatCandleTime(candle),
      action,
      price,
      pnl,
    });
    state.position = null;
    state.actedIndexes.add(idx);
    setMessage(`${action} @ ${formatPrice(price)} / ${formatPnL(pnl)}`);
  }

  function unrealizedPnL() {
    const candle = currentCandle();
    if (!candle || !state.position) return null;
    const price = finiteNumber(candle.execution_price);
    if (price === null) return null;
    return state.position.side === "long"
      ? price - state.position.entryPrice
      : state.position.entryPrice - price;
  }

  function sessionClosePrices() {
    const candles = Array.isArray(state.session?.candles) ? state.session.candles : [];
    const prices = candles.map((candle) => {
      const close = finiteNumber(candle?.close);
      return close === null ? finiteNumber(candle?.execution_price) : close;
    });
    return prices.every((price) => price !== null) ? prices : [];
  }

  function scoringMetadata() {
    if (state.session?.scoring && typeof state.session.scoring === "object") {
      return state.session.scoring;
    }
    return calculateScoringFromCandles();
  }

  function calculateScoringFromCandles() {
    const prices = sessionClosePrices();
    if (!prices.length || !(prices[0] > 0)) {
      return {
        long_only: { lower_return: null, max_return: null, denominator: null },
        long_short: { max_return: null },
      };
    }
    const base = prices[0];
    const deltas = prices.slice(0, -1).map((price, index) => prices[index + 1] - price);
    const buyHoldReturn = (prices[prices.length - 1] - prices[0]) / base;
    const lowerReturn = Math.min(0, buyHoldReturn);
    const maxLongReturn = deltas.reduce((sum, delta) => sum + Math.max(delta, 0), 0) / base;
    const denominator = maxLongReturn - lowerReturn;
    return {
      base_price: base,
      buy_hold_return: buyHoldReturn,
      long_only: {
        lower_return: lowerReturn,
        max_return: maxLongReturn,
        denominator,
      },
      long_short: {
        max_return: longShortMaxReturn(prices),
      },
    };
  }

  function longShortMaxReturn(prices) {
    if (!prices.length || !(prices[0] > 0)) return null;
    let flat = 0;
    let long = -prices[0];
    let short = prices[0];
    prices.slice(1).forEach((price) => {
      const nextFlat = Math.max(flat, long + price, short - price);
      const nextLong = Math.max(long, flat - price);
      const nextShort = Math.max(short, flat + price);
      flat = nextFlat;
      long = nextLong;
      short = nextShort;
    });
    return flat / prices[0];
  }

  function playerReturn() {
    const prices = sessionClosePrices();
    if (!prices.length || !(prices[0] > 0)) return null;
    const intervalPositions = new Array(Math.max(0, prices.length - 1)).fill(0);
    state.closedPositions.forEach((position) => {
      const sign = positionSign(position.side);
      const start = Math.max(0, Number(position.entryIndex));
      const end = Math.min(intervalPositions.length, Number(position.exitIndex));
      if (!sign || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
      for (let index = start; index < end; index += 1) {
        intervalPositions[index] = sign;
      }
    });
    const pnl = intervalPositions.reduce((sum, q, index) => (
      sum + q * (prices[index + 1] - prices[index])
    ), 0);
    return pnl / prices[0];
  }

  function finalScoreValue() {
    if (!state.gameDone || !state.session) return null;
    const rPlayer = playerReturn();
    if (rPlayer === null) return null;

    const scoring = scoringMetadata();
    if (state.selectedTradeMode === "long_short") {
      const maxReturn = finiteNumber(scoring?.long_short?.max_return);
      if (maxReturn === null || Math.abs(maxReturn) <= 1e-12) return null;
      return 100 * rPlayer / maxReturn;
    }

    const lowerReturn = finiteNumber(scoring?.long_only?.lower_return);
    const denominator = finiteNumber(scoring?.long_only?.denominator);
    if (lowerReturn === null || denominator === null || Math.abs(denominator) <= 1e-12) return null;
    return 100 * (rPlayer - lowerReturn) / denominator;
  }

  function updateControls() {
    const candle = currentCandle();
    const idx = currentIndex();
    const canAct = Boolean(
      state.session
      && candle
      && !state.loading
      && !state.gameDone
      && !state.actedIndexes.has(idx),
    );
    el.newGame.disabled = state.loading;
    el.next.disabled = Boolean(
      state.loading
      || !state.session
      || state.gameDone
      || state.revealedCount >= (state.session?.candles.length || 0),
    );
    el.long.disabled = !canAct || Boolean(state.position);
    el.short.disabled = !canAct || isLongOnlyTradeMode() || Boolean(state.position);
    el.close.disabled = !canAct || !state.position;
    renderChartFrameControls(state.session);
  }

  function render() {
    renderSessionMeta();
    renderKpis();
    renderTrades();
    drawChart();
    updateControls();
  }

  function renderSessionMeta() {
    const session = state.session;
    el.symbol.textContent = formatSymbol(session);
    el.date.textContent = formatDateRange(session);
    el.market.textContent = session?.market_label || "-";
    el.bars.textContent = session ? String(session.candle_count) : "-";
    el.interval.textContent = session?.interval || "-";
    el.source.textContent = session?.source || "-";
    el.rule.textContent = session?.execution_price_rule || "-";
    el.clock.textContent = session ? `${session.timezone}` : "-";
    el.progressStrip.textContent = session
      ? `${state.revealedCount} / ${session.candles.length}`
      : "0 / 0";
    el.progressLabel.textContent = state.gameDone ? "Closed" : state.session ? "Replay" : "Waiting";
    el.progressLabel.classList.toggle("open", Boolean(state.session && !state.gameDone));
    el.progressLabel.classList.toggle("closed", Boolean(state.gameDone));
    el.next.textContent = session?.step_label || (state.selectedMode === "daily" ? "Next Day" : "Next 15m");
    const replayLabel = session?.mode_label
      ? `${session.mode_label} Replay`
      : state.selectedMode === "daily" ? "Daily Replay" : "15m Replay";
    el.modeSubtitle.textContent = `${replayLabel} / ${tradeModeLabel()}`;
    el.chartTitle.textContent = chartTitleText(session);
    if (el.chart) {
      const activeFrame = chartTimeframes(session).find((frame) => frame.interval === effectiveChartInterval(session));
      el.chart.setAttribute("aria-label", activeFrame?.chartLabel || session?.chart_label || chartTitleText(session));
    }
    renderChartFrameControls(session);
    if (el.tradeSubtitle) {
      el.tradeSubtitle.textContent = tradeModeLabel();
    }
  }

  function renderChartFrameControls(session) {
    if (!el.chartFrameSwitch || !el.chartFramePrimary || !el.chartFrame5m) return;

    const primary = primaryChartInterval(session);
    const active = effectiveChartInterval(session);
    const hasPrimary = hasChartInterval(session, primary);
    const hasDetail = hasChartInterval(session, DETAIL_CHART_INTERVAL);
    const showSwitch = Boolean(session && !isDailySession(session) && hasDetail);

    el.chartFrameSwitch.classList.toggle("hidden", !showSwitch);
    el.chartFramePrimary.textContent = chartIntervalLabel(primary);
    el.chartFrame5m.textContent = chartIntervalLabel(DETAIL_CHART_INTERVAL);
    el.chartFramePrimary.classList.toggle("active", active === primary);
    el.chartFrame5m.classList.toggle("active", active === DETAIL_CHART_INTERVAL);
    el.chartFramePrimary.setAttribute("aria-pressed", active === primary ? "true" : "false");
    el.chartFrame5m.setAttribute("aria-pressed", active === DETAIL_CHART_INTERVAL ? "true" : "false");
    el.chartFramePrimary.disabled = state.loading || !hasPrimary;
    el.chartFrame5m.disabled = state.loading || !hasDetail;
  }

  function renderKpis() {
    const candle = currentCandle();
    const lastPrice = candle ? finiteNumber(candle.execution_price) : null;
    const unrealized = unrealizedPnL();
    const finalScore = finalScoreValue();

    el.currentTime.textContent = formatCandleTime(candle);
    el.currentPrice.textContent = formatPrice(lastPrice);
    el.last.textContent = formatPrice(lastPrice);
    el.entry.textContent = state.position ? formatPrice(state.position.entryPrice) : "-";
    el.side.textContent = state.position ? state.position.side.toUpperCase() : "-";
    el.tradeCount.textContent = String(state.trades.length);
    el.position.textContent = state.position
      ? `${state.position.side.toUpperCase()} 1`
      : "Flat";
    el.unrealized.textContent = formatPnL(unrealized);
    el.realized.textContent = formatPnL(state.realized);
    el.score.textContent = state.gameDone ? formatScore(finalScore) : "-";
    el.unrealized.classList.toggle("up", Boolean(unrealized && unrealized > 0));
    el.unrealized.classList.toggle("down", Boolean(unrealized && unrealized < 0));
    el.realized.classList.toggle("up", state.realized > 0);
    el.realized.classList.toggle("down", state.realized < 0);
    el.score.classList.toggle("up", Boolean(state.gameDone && finalScore !== null && finalScore > 0));
    el.score.classList.toggle("down", Boolean(state.gameDone && finalScore !== null && finalScore < 0));
  }

  function renderTrades() {
    el.tradesBody.innerHTML = "";
    if (!state.trades.length) {
      const row = document.createElement("tr");
      row.innerHTML = `<td colspan="4" class="muted">-</td>`;
      el.tradesBody.appendChild(row);
      return;
    }

    state.trades.forEach((trade) => {
      const row = document.createElement("tr");
      const pnl = finiteNumber(trade.pnl);
      if (pnl !== null && pnl > 0) row.classList.add("up");
      if (pnl !== null && pnl < 0) row.classList.add("down");
      row.innerHTML = `
        <td>${escapeHtml(trade.time)}</td>
        <td>${escapeHtml(trade.action)}</td>
        <td>${formatPrice(trade.price)}</td>
        <td>${pnl === null ? "-" : formatPnL(pnl)}</td>
      `;
      el.tradesBody.appendChild(row);
    });
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function drawChart() {
    clearSvg(el.chart);
    const session = state.session;
    const chartSeries = revealedChartSeries(session);
    const visible = chartSeries.visible;
    const allCandles = chartSeries.allCandles;
    el.chartEmpty.classList.toggle("hidden", visible.length > 0);
    if (!session || !visible.length) return;

    const width = 920;
    const height = 420;
    const pad = { top: 18, right: 64, bottom: 42, left: 58 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const movingAverageConfigs = sessionMovingAverages(session);
    const highs = visible.map((item) => finiteNumber(item.high)).filter((value) => value !== null);
    const lows = visible.map((item) => finiteNumber(item.low)).filter((value) => value !== null);
    const maValues = visible.flatMap((item) => (
      movingAverageConfigs
        .map((config) => movingAverageValue(item, config))
        .filter((value) => value !== null)
    ));
    if (!highs.length || !lows.length) return;

    let minPrice = Math.min(...lows, ...maValues);
    let maxPrice = Math.max(...highs, ...maValues);
    if (maxPrice === minPrice) {
      maxPrice += 1;
      minPrice -= 1;
    }
    const padding = (maxPrice - minPrice) * 0.08;
    maxPrice += padding;
    minPrice = Math.max(0, minPrice - padding);

    const total = Math.max(allCandles.length - 1, 1);
    const step = plotW / total;
    const candleW = Math.max(4, Math.min(16, step * 0.55));
    const xFor = (index) => pad.left + step * index;
    const yFor = (price) => pad.top + ((maxPrice - price) / (maxPrice - minPrice)) * plotH;

    appendSvg("rect", {
      x: pad.left,
      y: pad.top,
      width: plotW,
      height: plotH,
      class: "day-game-chart-bg",
    });

    for (let i = 0; i <= 4; i += 1) {
      const y = pad.top + (plotH / 4) * i;
      const value = maxPrice - ((maxPrice - minPrice) / 4) * i;
      appendSvg("line", {
        x1: pad.left,
        x2: pad.left + plotW,
        y1: y,
        y2: y,
        class: "symbol-chart-grid-line",
      });
      appendSvg("text", {
        x: width - 8,
        y: y + 4,
        "text-anchor": "end",
        class: "symbol-chart-axis-label",
      }, compactNumber(value));
    }

    appendSvg("line", {
      x1: pad.left,
      x2: pad.left,
      y1: pad.top,
      y2: pad.top + plotH,
      class: "symbol-chart-axis-line",
    });
    appendSvg("line", {
      x1: pad.left,
      x2: pad.left + plotW,
      y1: pad.top + plotH,
      y2: pad.top + plotH,
      class: "symbol-chart-axis-line",
    });

    if (!isDailySession(session)) {
      visible.forEach((item, index) => {
        if (index === 0 || !item.date || item.date === visible[index - 1]?.date) return;
        const x = xFor(index);
        appendSvg("line", {
          x1: x,
          x2: x,
          y1: pad.top,
          y2: pad.top + plotH,
          class: "day-game-day-divider",
        });
        appendSvg("text", {
          x: x + 6,
          y: pad.top + 14,
          class: "day-game-day-label",
        }, String(item.date).slice(5));
      });
    }

    visible.forEach((item, index) => {
      const open = finiteNumber(item.open);
      const high = finiteNumber(item.high);
      const low = finiteNumber(item.low);
      const close = finiteNumber(item.close);
      if ([open, high, low, close].some((value) => value === null)) return;

      const x = xFor(index);
      const up = close >= open;
      const yOpen = yFor(open);
      const yClose = yFor(close);
      const bodyTop = Math.min(yOpen, yClose);
      const bodyH = Math.max(2, Math.abs(yClose - yOpen));
      appendSvg("line", {
        x1: x,
        x2: x,
        y1: yFor(high),
        y2: yFor(low),
        class: up ? "day-game-wick up" : "day-game-wick down",
      });
      appendSvg("rect", {
        x: x - candleW / 2,
        y: bodyTop,
        width: candleW,
        height: bodyH,
        rx: 1,
        class: up ? "day-game-candle up" : "day-game-candle down",
      });

      if (index === visible.length - 1) {
        appendSvg("circle", {
          cx: x,
          cy: yFor(close),
          r: 4,
          class: "day-game-current-dot",
        });
      }
    });

    drawMovingAverageLines(visible, { xFor, yFor, configs: movingAverageConfigs });

    const labelEvery = Math.max(1, Math.ceil(allCandles.length / 6));
    visible.forEach((item, index) => {
      if (index !== 0 && index !== visible.length - 1 && index % labelEvery !== 0) return;
      const x = xFor(index);
      appendSvg("text", {
        x,
        y: height - 14,
        "text-anchor": "middle",
        class: "symbol-chart-axis-label",
      }, formatAxisTime(item));
    });
  }

  function sessionMovingAverages(session) {
    if (!Array.isArray(session?.moving_averages) || !session.moving_averages.length) {
      return DEFAULT_MOVING_AVERAGES;
    }
    return session.moving_averages
      .map((config) => ({
        key: String(config.key || ""),
        label: String(config.label || config.key || ""),
        window: Number(config.window),
      }))
      .filter((config) => config.key && config.label && Number.isFinite(config.window) && config.window > 0);
  }

  function drawMovingAverageLines(visible, { xFor, yFor, configs }) {
    const series = configs.map((config) => ({
      ...config,
      segments: movingAverageSegments(visible, config),
    }));

    series.forEach((config) => {
      config.segments.forEach((segment) => {
        if (segment.length < 2) return;
        appendSvg("polyline", {
          points: segment.map((point) => `${xFor(point.index).toFixed(2)},${yFor(point.value).toFixed(2)}`).join(" "),
          class: `day-game-ma-line ${config.key}`,
        });
      });
    });

    const available = series.filter((config) => config.segments.some((segment) => segment.length > 0));
    if (!available.length) return;

    const legend = appendSvg("g", {
      class: "day-game-ma-legend",
      transform: "translate(72 30)",
    });
    available.forEach((config, index) => {
      const x = index * 58;
      const swatch = document.createElementNS(SVG_NS, "line");
      swatch.setAttribute("x1", String(x));
      swatch.setAttribute("x2", String(x + 18));
      swatch.setAttribute("y1", "0");
      swatch.setAttribute("y2", "0");
      swatch.setAttribute("class", `day-game-ma-legend-line ${config.key}`);
      legend.appendChild(swatch);

      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", String(x + 24));
      label.setAttribute("y", "4");
      label.setAttribute("class", "day-game-ma-legend-label");
      label.textContent = config.label;
      legend.appendChild(label);
    });
  }

  function movingAverageValue(candle, config) {
    const values = candle?.moving_averages;
    if (!values || !Object.prototype.hasOwnProperty.call(values, config.key)) return null;
    return finiteNumber(values[config.key]);
  }

  function movingAverageSegments(candles, config) {
    const hasPayloadValues = candles.some((item) => (
      item?.moving_averages
      && Object.prototype.hasOwnProperty.call(item.moving_averages, config.key)
    ));
    if (hasPayloadValues) {
      return movingAveragePayloadSegments(candles, config);
    }
    return rollingMovingAverageSegments(candles, config.window);
  }

  function movingAveragePayloadSegments(candles, config) {
    const segments = [];
    let currentSegment = [];
    candles.forEach((item, index) => {
      const value = movingAverageValue(item, config);
      if (value === null) {
        if (currentSegment.length) {
          segments.push(currentSegment);
          currentSegment = [];
        }
        return;
      }
      currentSegment.push({ index, value });
    });
    if (currentSegment.length) {
      segments.push(currentSegment);
    }
    return segments;
  }

  function rollingMovingAverageSegments(candles, windowSize) {
    if (windowSize <= 0 || candles.length < windowSize) return [];
    const segments = [];
    let currentSegment = [];
    let sum = 0;
    const queue = [];
    candles.forEach((item, index) => {
      const close = finiteNumber(item.close);
      if (close === null) {
        sum = 0;
        queue.length = 0;
        if (currentSegment.length) {
          segments.push(currentSegment);
          currentSegment = [];
        }
        return;
      }
      queue.push(close);
      sum += close;
      if (queue.length > windowSize) {
        sum -= queue.shift();
      }
      if (queue.length === windowSize) {
        currentSegment.push({
          index,
          value: sum / windowSize,
        });
      }
    });
    if (currentSegment.length) {
      segments.push(currentSegment);
    }
    return segments;
  }

  function compactNumber(value) {
    const digits = priceDigits();
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function clearSvg(svg) {
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
  }

  function appendSvg(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => {
      node.setAttribute(key, String(value));
    });
    if (text !== undefined) node.textContent = text;
    el.chart.appendChild(node);
    return node;
  }

  function setMarket(market) {
    if (state.selectedMarket === market) return;
    state.selectedMarket = market;
    el.marketUs.classList.toggle("active", market === "us");
    el.marketJp.classList.toggle("active", market === "jp");
    resetGame(null);
    setMessage("No session");
    render();
  }

  function setMode(mode) {
    if (state.selectedMode === mode) return;
    state.selectedMode = mode;
    el.modeIntraday.classList.toggle("active", mode === "intraday");
    el.modeDaily.classList.toggle("active", mode === "daily");
    resetGame(null);
    setMessage("No session");
    render();
  }

  function setTradeMode(mode) {
    if (state.selectedTradeMode === mode) return;
    state.selectedTradeMode = mode;
    el.tradeModeLongOnly.classList.toggle("active", mode === "long_only");
    el.tradeModeLongShort.classList.toggle("active", mode === "long_short");
    resetGame(null);
    setMessage("No session");
    render();
  }

  function setChartInterval(interval) {
    if (!state.session || !hasChartInterval(state.session, interval)) return;
    if (state.activeChartInterval === interval) return;
    state.activeChartInterval = interval;
    render();
  }

  el.marketUs.addEventListener("click", () => setMarket("us"));
  el.marketJp.addEventListener("click", () => setMarket("jp"));
  el.modeIntraday.addEventListener("click", () => setMode("intraday"));
  el.modeDaily.addEventListener("click", () => setMode("daily"));
  el.tradeModeLongOnly.addEventListener("click", () => setTradeMode("long_only"));
  el.tradeModeLongShort.addEventListener("click", () => setTradeMode("long_short"));
  el.chartFramePrimary.addEventListener("click", () => setChartInterval(primaryChartInterval(state.session)));
  el.chartFrame5m.addEventListener("click", () => setChartInterval(DETAIL_CHART_INTERVAL));
  el.newGame.addEventListener("click", loadNewGame);
  el.next.addEventListener("click", revealNext);
  el.long.addEventListener("click", () => openPosition("long"));
  el.short.addEventListener("click", () => openPosition("short"));
  el.close.addEventListener("click", () => {
    closePosition("Close");
    render();
  });

  render();
})();
