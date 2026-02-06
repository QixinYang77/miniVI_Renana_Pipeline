(function () {
  if (window.__dashBadSpanHotkeyInit) {
    return;
  }
  window.__dashBadSpanHotkeyInit = true;
  window.__dashSyncHighlightHeld = false;

  function isTypingTarget(el) {
    if (!el) return false;
    var tag = (el.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (el.isContentEditable) return true;
    return false;
  }

  document.addEventListener("keydown", function (e) {
    if (e.defaultPrevented) return;
    if (e.repeat) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isTypingTarget(e.target)) return;

    var key = String(e.key || "").toLowerCase();
    if (key === "b") {
      var btnToggle = document.getElementById("hotkey-toggle-bad-span-btn");
      if (!btnToggle) return;
      btnToggle.click();
      e.preventDefault();
      return;
    }
    if (key === "s") {
      if (window.__dashSyncHighlightHeld) return;
      window.__dashSyncHighlightHeld = true;
      var btnSyncDown = document.getElementById("hotkey-sync-highlight-down-btn");
      if (!btnSyncDown) return;
      btnSyncDown.click();
      e.preventDefault();
      return;
    }
    if (key === "enter") {
      var btnConfirm = document.getElementById("hotkey-confirm-bad-span-btn");
      if (!btnConfirm) return;
      btnConfirm.click();
      e.preventDefault();
      return;
    }
    if (key === "escape") {
      var btnCancel = document.getElementById("hotkey-cancel-pending-btn");
      if (!btnCancel) return;
      btnCancel.click();
      e.preventDefault();
    }
  });

  document.addEventListener("keyup", function (e) {
    var key = String(e.key || "").toLowerCase();
    if (key !== "s") return;
    if (!window.__dashSyncHighlightHeld) return;
    window.__dashSyncHighlightHeld = false;
    var btnSyncUp = document.getElementById("hotkey-sync-highlight-up-btn");
    if (!btnSyncUp) return;
    btnSyncUp.click();
  });

  window.addEventListener("blur", function () {
    if (!window.__dashSyncHighlightHeld) return;
    window.__dashSyncHighlightHeld = false;
    var btnSyncUp = document.getElementById("hotkey-sync-highlight-up-btn");
    if (!btnSyncUp) return;
    btnSyncUp.click();
  });
})();
