(() => {
  const HELP_DETAIL_SELECTOR = "details.param-help, details.mlops-inline-help";
  const OPEN_HELP_DETAIL_SELECTOR = "details.param-help[open], details.mlops-inline-help[open]";
  const VIEWPORT_MARGIN = 12;
  const VERTICAL_GAP = 8;
  const overlayStateByDetail = new Map();
  const boundDetails = new WeakSet();
  let refreshFrame = 0;

  function getSummary(detail) {
    return Array.from(detail.children).find((child) => child.tagName === "SUMMARY") || null;
  }

  function getPanel(detail) {
    return Array.from(detail.children).find((child) => child.tagName !== "SUMMARY") || null;
  }

  function scheduleRefresh() {
    if (refreshFrame) return;
    refreshFrame = window.requestAnimationFrame(() => {
      refreshFrame = 0;
      refreshOpenPopovers();
    });
  }

  function closeOtherPopovers(exceptDetail) {
    document.querySelectorAll(OPEN_HELP_DETAIL_SELECTOR).forEach((detail) => {
      if (detail === exceptDetail) return;
      detail.open = false;
    });
  }

  function applyPanelStyles(sourcePanel, overlayPanel) {
    const computed = window.getComputedStyle(sourcePanel);
    [
      "padding",
      "font-size",
      "font-weight",
      "font-family",
      "line-height",
      "letter-spacing",
      "text-transform",
      "text-align",
      "color",
      "background",
      "background-image",
      "background-color",
      "border",
      "border-radius",
      "box-shadow",
      "white-space",
      "overflow-wrap",
      "word-break",
    ].forEach((property) => {
      overlayPanel.style.setProperty(property, computed.getPropertyValue(property));
    });
  }

  function syncOverlayContent(detail) {
    const state = overlayStateByDetail.get(detail);
    if (!state) return;
    state.overlay.innerHTML = state.panel.innerHTML;
  }

  function positionPopover(detail) {
    const state = overlayStateByDetail.get(detail);
    if (!state) return;
    if (!detail.open || !detail.isConnected) {
      closePopover(detail);
      return;
    }

    const summaryRect = state.summary.getBoundingClientRect();
    const panelRect = state.panel.getBoundingClientRect();
    const computed = window.getComputedStyle(state.panel);
    const preferRight = computed.getPropertyValue("right") !== "auto" && computed.getPropertyValue("left") === "auto";
    const maxWidth = Math.max(160, window.innerWidth - (VIEWPORT_MARGIN * 2));
    const width = Math.min(Math.max(panelRect.width || state.overlay.scrollWidth || 240, 160), maxWidth);

    state.overlay.style.maxWidth = `${maxWidth}px`;
    state.overlay.style.width = `${Math.round(width)}px`;
    state.overlay.style.maxHeight = `${Math.max(120, window.innerHeight - (VIEWPORT_MARGIN * 2))}px`;
    state.overlay.style.visibility = "hidden";
    state.overlay.style.left = `${VIEWPORT_MARGIN}px`;
    state.overlay.style.top = `${VIEWPORT_MARGIN}px`;

    const overlayRect = state.overlay.getBoundingClientRect();
    let left = preferRight ? summaryRect.right - overlayRect.width : summaryRect.left;
    left = Math.min(Math.max(left, VIEWPORT_MARGIN), window.innerWidth - overlayRect.width - VIEWPORT_MARGIN);

    const spaceBelow = window.innerHeight - summaryRect.bottom - VIEWPORT_MARGIN;
    const spaceAbove = summaryRect.top - VIEWPORT_MARGIN;
    let top = summaryRect.bottom + VERTICAL_GAP;
    if (overlayRect.height > spaceBelow && spaceAbove > spaceBelow) {
      top = summaryRect.top - overlayRect.height - VERTICAL_GAP;
    }
    top = Math.min(Math.max(top, VIEWPORT_MARGIN), window.innerHeight - overlayRect.height - VIEWPORT_MARGIN);

    state.overlay.style.left = `${Math.round(left)}px`;
    state.overlay.style.top = `${Math.round(top)}px`;
    state.overlay.style.visibility = "visible";
  }

  function openPopover(detail) {
    const summary = getSummary(detail);
    const panel = getPanel(detail);
    if (!summary || !panel) return;

    closeOtherPopovers(detail);

    let state = overlayStateByDetail.get(detail);
    if (!state) {
      const overlay = panel.cloneNode(true);
      overlay.removeAttribute("id");
      overlay.classList.add("help-popover-layer");
      overlay.setAttribute("role", "tooltip");
      overlay.addEventListener("click", (event) => {
        event.stopPropagation();
      });
      overlay.addEventListener("mousedown", (event) => {
        event.stopPropagation();
      });

      const observer = new MutationObserver(() => {
        syncOverlayContent(detail);
        scheduleRefresh();
      });

      state = { overlay, summary, panel, observer };
      overlayStateByDetail.set(detail, state);
      document.body.appendChild(overlay);
      observer.observe(panel, { childList: true, subtree: true, characterData: true });
    } else {
      state.summary = summary;
      state.panel = panel;
    }

    syncOverlayContent(detail);
    applyPanelStyles(panel, state.overlay);
    detail.setAttribute("data-help-popover-active", "true");
    positionPopover(detail);
  }

  function closePopover(detail) {
    detail.removeAttribute("data-help-popover-active");
    const state = overlayStateByDetail.get(detail);
    if (!state) return;
    state.observer.disconnect();
    state.overlay.remove();
    overlayStateByDetail.delete(detail);
  }

  function refreshOpenPopovers() {
    document.querySelectorAll(OPEN_HELP_DETAIL_SELECTOR).forEach((detail) => {
      openPopover(detail);
    });

    Array.from(overlayStateByDetail.keys()).forEach((detail) => {
      if (!detail.open || !detail.isConnected) {
        closePopover(detail);
      }
    });
  }

  function bindDetail(detail) {
    if (!(detail instanceof HTMLDetailsElement) || boundDetails.has(detail)) return;
    boundDetails.add(detail);
    detail.addEventListener("toggle", () => {
      if (detail.open) {
        openPopover(detail);
      } else {
        closePopover(detail);
      }
    });
    if (detail.open) {
      openPopover(detail);
    }
  }

  function bindExistingDetails(root = document) {
    root.querySelectorAll(HELP_DETAIL_SELECTOR).forEach((detail) => {
      bindDetail(detail);
    });
  }

  bindExistingDetails();

  const mutationObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (!(node instanceof Element)) return;
        if (node.matches(HELP_DETAIL_SELECTOR)) {
          bindDetail(node);
        }
        bindExistingDetails(node);
      });
      mutation.removedNodes.forEach((node) => {
        if (!(node instanceof Element)) return;
        if (node.matches(OPEN_HELP_DETAIL_SELECTOR)) {
          closePopover(node);
        }
        node.querySelectorAll?.(OPEN_HELP_DETAIL_SELECTOR).forEach((detail) => {
          closePopover(detail);
        });
      });
    });
  });

  if (document.body) {
    mutationObserver.observe(document.body, { childList: true, subtree: true });
  }

  window.addEventListener("resize", scheduleRefresh, { passive: true });
  window.addEventListener("scroll", scheduleRefresh, { passive: true, capture: true });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    document.querySelectorAll(OPEN_HELP_DETAIL_SELECTOR).forEach((detail) => {
      detail.open = false;
    });
  });
})();
