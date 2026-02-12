(() => {
  // Highlight the active page in the navigation menu
  const path = window.location.pathname.replace(/\/+$/, "") || "/";
  const tabs = document.querySelectorAll(".nav-tab");

  tabs.forEach((tab) => {
    const tabPath = (tab.getAttribute("data-path") || "").replace(/\/+$/, "") || "/";
    if (path === tabPath) {
      tab.classList.add("active");
    }
  });

  const nav = document.querySelector(".nav-tabs");
  if (!nav) return;

  // Keep monitor terminal layout stable; skip drawer conversion on this page.
  if (document.body.classList.contains("terminal-page")) {
    return;
  }

  document.body.classList.add("has-drawer-menu");

  const menuButton = document.createElement("button");
  menuButton.type = "button";
  menuButton.className = "nav-menu-toggle";
  menuButton.setAttribute("aria-label", "Open navigation menu");
  menuButton.setAttribute("aria-controls", "site-nav");
  menuButton.setAttribute("aria-expanded", "false");
  menuButton.innerHTML = `
    <span class="menu-line"></span>
    <span class="menu-line"></span>
    <span class="menu-line"></span>
  `;

  nav.id = "site-nav";

  const backdrop = document.createElement("button");
  backdrop.type = "button";
  backdrop.className = "nav-drawer-backdrop";
  backdrop.setAttribute("aria-hidden", "true");
  backdrop.tabIndex = -1;

  function setMenuOpen(isOpen) {
    document.body.classList.toggle("menu-open", isOpen);
    menuButton.setAttribute("aria-expanded", String(isOpen));
  }

  menuButton.addEventListener("click", () => {
    const isOpen = document.body.classList.contains("menu-open");
    setMenuOpen(!isOpen);
  });

  backdrop.addEventListener("click", () => setMenuOpen(false));

  nav.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof Element && target.closest(".nav-tab")) {
      setMenuOpen(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setMenuOpen(false);
    }
  });

  document.body.append(menuButton, backdrop);
})();
