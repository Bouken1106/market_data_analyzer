(() => {
  // Highlight the active page in the top tab bar
  const path = window.location.pathname.replace(/\/+$/, "") || "/";
  const tabs = document.querySelectorAll(".nav-tab");

  tabs.forEach((tab) => {
    const tabPath = (tab.getAttribute("data-path") || "").replace(/\/+$/, "") || "/";
    if (path === tabPath) {
      tab.classList.add("active");
    }
  });
})();
