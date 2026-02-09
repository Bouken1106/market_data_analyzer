(() => {
  const pageMenus = Array.from(document.querySelectorAll("details.page-menu"));
  if (pageMenus.length === 0) return;

  document.addEventListener("pointerdown", (event) => {
    const target = event.target;
    pageMenus.forEach((menu) => {
      if (!menu.open) return;
      if (target instanceof Node && menu.contains(target)) return;
      menu.removeAttribute("open");
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    pageMenus.forEach((menu) => {
      if (menu.open) menu.removeAttribute("open");
    });
  });
})();
