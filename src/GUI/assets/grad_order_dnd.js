(function () {
  var dragLevel = null;

  function levelsFromList(list) {
    return Array.prototype.map.call(
      list.querySelectorAll(".grad-ord-item"),
      function (el) { return el.getAttribute("data-level"); }
    );
  }

  document.addEventListener("dragstart", function (e) {
    var el = e.target && e.target.closest && e.target.closest(".grad-ord-item");
    if (!el) return;
    dragLevel = el.getAttribute("data-level");
    try {
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", dragLevel || "");
    } catch (err) {}
    el.classList.add("opacity-50");
  });

  document.addEventListener("dragend", function (e) {
    var el = e.target && e.target.closest && e.target.closest(".grad-ord-item");
    if (el) el.classList.remove("opacity-50");
    document.querySelectorAll(".grad-ord-drag-over").forEach(function (node) {
      node.classList.remove("grad-ord-drag-over");
    });
    dragLevel = null;
  });

  document.addEventListener("dragover", function (e) {
    var el = e.target && e.target.closest && e.target.closest(".grad-ord-item");
    if (!el) return;
    e.preventDefault();
    try { e.dataTransfer.dropEffect = "move"; } catch (err) {}
    document.querySelectorAll(".grad-ord-drag-over").forEach(function (node) {
      if (node !== el) node.classList.remove("grad-ord-drag-over");
    });
    el.classList.add("grad-ord-drag-over");
  });

  document.addEventListener("drop", function (e) {
    var el = e.target && e.target.closest && e.target.closest(".grad-ord-item");
    if (!el || dragLevel == null) return;
    e.preventDefault();
    el.classList.remove("grad-ord-drag-over");
    var list = el.closest("#grad-order-list");
    if (!list) return;
    var targetLevel = el.getAttribute("data-level");
    var levels = levelsFromList(list);
    var from = levels.indexOf(dragLevel);
    var to = levels.indexOf(targetLevel);
    if (from < 0 || to < 0 || from === to) return;
    levels.splice(from, 1);
    levels.splice(to, 0, dragLevel);
    if (window.dash_clientside && typeof window.dash_clientside.set_props === "function") {
      window.dash_clientside.set_props("grad-order-store", { data: levels });
    }
  });
})();
