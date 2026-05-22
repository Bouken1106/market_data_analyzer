(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);

  const state = {
    selectedMarket: "us",
    loading: false,
    session: null,
    revealedCount: 0,
    position: null,
    realized: 0,
    trades: [],
    actedIndexes: new Set(),
    gameDone: false,
  };

  const el = {
    marketUs: $("dtg-market-us"),
    marketJp: $("dtg-market-jp"),
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
    const digits = Number(state.session?.currency_digits ?? 2);
    const symbol = state.session?.currency_symbol || "";
    return `${symbol}${numeric.toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    })}`;
  }

  function formatPnL(value) {
    const numeric = finiteNumber(value);
    if (numeric === null) return "-";
    const sign = numeric > 0 ? "+" : numeric < 0 ? "-" : "";
    return `${sign}${formatPrice(Math.abs(numeric))}`;
  }

  function setMessage(text, tone = "") {
    el.message.textContent = text;
    el.message.classList.toggle("error", tone === "error");
  }

  function resetGame(session) {
    state.session = session;
    state.revealedCount = 0;
    state.position = null;
    state.realized = 0;
    state.trades = [];
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
      setMessage("Game complete");
    }
    render();
  }

  function openPosition(side) {
    const candle = currentCandle();
    const idx = currentIndex();
    if (!candle || state.position || state.gameDone || state.actedIndexes.has(idx)) return;

    const price = finiteNumber(candle.execution_price);
    if (price === null) return;

    state.position = {
      side,
      entryPrice: price,
      entryTime: candle.time,
      entryIndex: idx,
    };
    state.actedIndexes.add(idx);
    state.trades.unshift({
      time: candle.time,
      action: side === "long" ? "Long" : "Short",
      price,
      pnl: null,
    });
    setMessage(`${side === "long" ? "Long" : "Short"} @ ${formatPrice(price)}`);
    render();
  }

  function closePosition(action = "Close") {
    const candle = currentCandle();
    const idx = currentIndex();
    if (!candle || !state.position || state.actedIndexes.has(idx)) return;

    const price = finiteNumber(candle.execution_price);
    if (price === null) return;

    const pnl = state.position.side === "long"
      ? price - state.position.entryPrice
      : state.position.entryPrice - price;
    state.realized += pnl;
    state.trades.unshift({
      time: candle.time,
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
    el.short.disabled = !canAct || Boolean(state.position);
    el.close.disabled = !canAct || !state.position;
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
    el.symbol.textContent = session?.symbol || "-";
    el.date.textContent = session?.date || "-";
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
  }

  function renderKpis() {
    const candle = currentCandle();
    const lastPrice = candle ? finiteNumber(candle.execution_price) : null;
    const unrealized = unrealizedPnL();
    const totalScore = state.realized + (state.gameDone ? 0 : (unrealized || 0));

    el.currentTime.textContent = candle?.time || "-";
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
    el.score.textContent = formatPnL(totalScore);
    el.unrealized.classList.toggle("up", Boolean(unrealized && unrealized > 0));
    el.unrealized.classList.toggle("down", Boolean(unrealized && unrealized < 0));
    el.realized.classList.toggle("up", state.realized > 0);
    el.realized.classList.toggle("down", state.realized < 0);
    el.score.classList.toggle("up", totalScore > 0);
    el.score.classList.toggle("down", totalScore < 0);
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
    const visible = session ? session.candles.slice(0, state.revealedCount) : [];
    el.chartEmpty.classList.toggle("hidden", visible.length > 0);
    if (!session || !visible.length) return;

    const width = 920;
    const height = 420;
    const pad = { top: 18, right: 64, bottom: 42, left: 58 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const highs = visible.map((item) => finiteNumber(item.high)).filter((value) => value !== null);
    const lows = visible.map((item) => finiteNumber(item.low)).filter((value) => value !== null);
    if (!highs.length || !lows.length) return;

    let minPrice = Math.min(...lows);
    let maxPrice = Math.max(...highs);
    if (maxPrice === minPrice) {
      maxPrice += 1;
      minPrice -= 1;
    }
    const padding = (maxPrice - minPrice) * 0.08;
    maxPrice += padding;
    minPrice = Math.max(0, minPrice - padding);

    const total = Math.max(session.candles.length - 1, 1);
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

    const labelEvery = Math.max(1, Math.ceil(session.candles.length / 6));
    visible.forEach((item, index) => {
      if (index !== 0 && index !== visible.length - 1 && index % labelEvery !== 0) return;
      const x = xFor(index);
      appendSvg("text", {
        x,
        y: height - 14,
        "text-anchor": "middle",
        class: "symbol-chart-axis-label",
      }, item.time);
    });
  }

  function compactNumber(value) {
    const digits = Number(state.session?.currency_digits ?? 2);
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
    state.selectedMarket = market;
    el.marketUs.classList.toggle("active", market === "us");
    el.marketJp.classList.toggle("active", market === "jp");
  }

  el.marketUs.addEventListener("click", () => setMarket("us"));
  el.marketJp.addEventListener("click", () => setMarket("jp"));
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
