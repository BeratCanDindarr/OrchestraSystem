// Fleet interactivity: drag-a-task-onto-a-model, and token-streamed chat.
// Progressive enhancement over the HTMX dashboard (no build step, vanilla JS).
(function () {
  "use strict";

  // ── Chat: stream the reply token-by-token via SSE ──────────────────────────
  document.body.addEventListener("submit", function (e) {
    var form = e.target;
    if (!form.matches || !form.matches("form.chat-form[data-chat-cid]")) return;
    e.preventDefault();
    var cid = form.getAttribute("data-chat-cid");
    var input = form.querySelector('input[name="message"]');
    var message = (input && input.value || "").trim();
    var log = document.getElementById("chat-log");
    if (!cid || !message || !log) return;

    var you = document.createElement("div");
    you.className = "chat-msg you";
    you.textContent = message;
    log.appendChild(you);

    var them = document.createElement("div");
    them.className = "chat-msg them";
    them.textContent = "…";
    log.appendChild(them);
    log.scrollTop = log.scrollHeight;
    input.value = "";

    var url = "/api/character/" + encodeURIComponent(cid) +
              "/chat/stream?message=" + encodeURIComponent(message);
    var es = new EventSource(url);
    var acc = "";
    es.addEventListener("token", function (ev) {
      acc += ev.data;
      them.textContent = acc;
      log.scrollTop = log.scrollHeight;
    });
    es.addEventListener("done", function () {
      es.close();
      if (!acc) them.textContent = "(no reply)";
    });
    es.onerror = function () { es.close(); if (!acc) them.textContent = "(connection lost)"; };
  });

  // ── Drag a typed task onto a model card to assign it there ──────────────────
  var dragTask = "";
  document.body.addEventListener("dragstart", function (e) {
    var grip = e.target.closest && e.target.closest("[data-drag-grip]");
    if (!grip) return;
    var input = document.querySelector("[data-drag-task]");
    dragTask = input ? (input.value || "").trim() : "";
    if (e.dataTransfer) {
      e.dataTransfer.setData("text/plain", dragTask);
      e.dataTransfer.effectAllowed = "copy";
    }
  });
  document.body.addEventListener("dragover", function (e) {
    var card = e.target.closest && e.target.closest("[data-cid]");
    if (card) { e.preventDefault(); card.classList.add("drop-hot"); }
  });
  document.body.addEventListener("dragleave", function (e) {
    var card = e.target.closest && e.target.closest("[data-cid]");
    if (card) card.classList.remove("drop-hot");
  });
  document.body.addEventListener("drop", function (e) {
    var card = e.target.closest && e.target.closest("[data-cid]");
    if (!card) return;
    e.preventDefault();
    card.classList.remove("drop-hot");
    var cid = card.getAttribute("data-cid");
    var task = ((e.dataTransfer && e.dataTransfer.getData("text/plain")) || dragTask || "").trim();
    if (!cid || !task) return;
    var body = new URLSearchParams({ character_id: cid, task: task });
    fetch("/api/dispatch", { method: "POST", body: body })
      .then(function (r) { return r.text(); })
      .then(function (html) {
        var fleet = document.getElementById("fleet");
        if (fleet) fleet.innerHTML = html;
      })
      .catch(function () {});
  });
})();

// ── Notifications: toast + beep when a NEW result lands in the inbox ─────────
(function () {
  var seen = new Set();
  var first = true;
  function toast(msg) {
    var box = document.getElementById("toasts");
    if (!box) return;
    var t = document.createElement("div");
    t.className = "toast";
    var who = document.createElement("span"); who.className = "t-who"; who.textContent = "✅ ";
    t.appendChild(who); t.appendChild(document.createTextNode(msg));
    box.appendChild(t);
    setTimeout(function () { t.remove(); }, 5000);
  }
  function beep() {
    try {
      var AC = window.AudioContext || window.webkitAudioContext;
      var ctx = new AC(), o = ctx.createOscillator(), g = ctx.createGain();
      o.connect(g); g.connect(ctx.destination);
      o.frequency.value = 660; g.gain.value = 0.04;
      o.start(); o.stop(ctx.currentTime + 0.12);
    } catch (e) {}
  }
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (!e.target || e.target.id !== "inbox") return;
    var done = e.target.querySelectorAll(".inbox-item.done[data-did], .inbox-item.failed[data-did]");
    var fresh = [];
    done.forEach(function (el) {
      var did = el.getAttribute("data-did");
      if (!seen.has(did)) { seen.add(did); if (!first) fresh.push(el); }
    });
    first = false;
    fresh.forEach(function (el) {
      var who = el.querySelector(".ii-who");
      toast((who ? who.textContent : "A model") + " finished a task");
      beep();
    });
  });
})();

// ── Office: animate EXISTING sprites in place (walk to desk / back to lounge) ─
// Re-rendering the whole office would recreate the sprites and skip the CSS walk
// transition; instead we toggle classes on the live elements so they animate.
(function () {
  var BUBBLE = { busy: "typing…", available: "ready", idle: "zzz", offline: "away" };
  function applyOfficeStates() {
    var sprites = document.querySelectorAll(".sprite[data-cid]");
    if (!sprites.length) return;
    fetch("/api/office/states").then(function (r) { return r.json(); }).then(function (states) {
      sprites.forEach(function (sp) {
        var s = states[sp.getAttribute("data-cid")];
        if (!s) return;
        ["st-busy", "st-available", "st-idle", "st-offline", "at-desk", "has-result"]
          .forEach(function (c) { sp.classList.remove(c); });
        sp.classList.add("st-" + s.state);
        if (s.at_desk) sp.classList.add("at-desk");
        if (s.has_result) sp.classList.add("has-result");
        var b = sp.querySelector(".bubble");
        if (b) b.textContent = s.has_result ? "✅ done!" : (BUBBLE[s.state] || "");
      });
    }).catch(function () {});
  }
  setInterval(applyOfficeStates, 2500);
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (e.target && e.target.id === "fleet") setTimeout(applyOfficeStates, 60);
  });
})();

// walking bob: toggle .walking while a sprite glides between lounge and desk
(function () {
  function isMove(e) {
    return e.target && e.target.classList && e.target.classList.contains("sprite")
      && (e.propertyName === "left" || e.propertyName === "top");
  }
  document.body.addEventListener("transitionstart", function (e) { if (isMove(e)) e.target.classList.add("walking"); });
  document.body.addEventListener("transitionend", function (e) { if (isMove(e)) e.target.classList.remove("walking"); });
})();

// ── Isometric office: zoom (wheel) + pan (drag empty floor) + reset ──────────
(function () {
  var view = { scale: 1, x: 0, y: 0 };          // reset whenever the office re-renders
  var MIN = 0.6, MAX = 2.2;
  function stage() { return document.querySelector(".office.iso .iso-stage"); }
  function apply() {
    var st = stage();
    if (!st) return;
    st.style.setProperty("--scale", view.scale);
    st.style.setProperty("--panx", view.x + "px");
    st.style.setProperty("--pany", view.y + "px");
  }
  function reset() { view = { scale: 1, x: 0, y: 0 }; apply(); }

  document.body.addEventListener("wheel", function (e) {
    var office = e.target.closest && e.target.closest(".office.iso");
    if (!office || !stage()) return;
    e.preventDefault();
    var rect = office.getBoundingClientRect();
    var cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    var factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    var ns = Math.min(MAX, Math.max(MIN, view.scale * factor));
    var k = ns / view.scale;
    view.x = cx - (cx - view.x) * k;             // keep the point under the cursor fixed
    view.y = cy - (cy - view.y) * k;
    view.scale = ns;
    apply();
  }, { passive: false });

  var drag = null;
  document.body.addEventListener("pointerdown", function (e) {
    var office = e.target.closest && e.target.closest(".office.iso");
    if (!office) return;
    if (e.target.closest(".sprite") || e.target.closest(".iso-reset")) return;  // let clicks/DnD through
    drag = { sx: e.clientX, sy: e.clientY, ox: view.x, oy: view.y };
    office.classList.add("grabbing");
  });
  document.body.addEventListener("pointermove", function (e) {
    if (!drag) return;
    view.x = drag.ox + (e.clientX - drag.sx);
    view.y = drag.oy + (e.clientY - drag.sy);
    apply();
  });
  function endDrag() {
    drag = null;
    var o = document.querySelector(".office.iso.grabbing");
    if (o) o.classList.remove("grabbing");
  }
  document.body.addEventListener("pointerup", endDrag);
  document.body.addEventListener("pointercancel", endDrag);
  document.body.addEventListener("click", function (e) {
    if (e.target.closest && e.target.closest(".iso-reset")) reset();
  });
  document.body.addEventListener("dblclick", function (e) {
    if (e.target.closest && e.target.closest(".office.iso")) reset();
  });
  // a freshly rendered office starts from the default view
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (e.target && e.target.id === "fleet") reset();
  });
})();

// ── Live connection indicator (SSE open/error) ──────────────────────────────
(function () {
  function setConn(state, text) {
    var el = document.getElementById("conn");
    if (!el) return;
    el.className = "conn " + state;
    el.textContent = text;
  }
  document.body.addEventListener("htmx:sseOpen", function () { setConn("up", "live"); });
  document.body.addEventListener("htmx:sseError", function () { setConn("down", "reconnecting…"); });
  document.body.addEventListener("htmx:sseClose", function () { setConn("down", "offline"); });
})();

// ── Global loading indicator (top progress bar) ─────────────────────────────
(function () {
  document.body.addEventListener("htmx:beforeRequest", function () {
    var b = document.getElementById("progress"); if (b) b.classList.add("on");
  });
  document.body.addEventListener("htmx:afterRequest", function () {
    var b = document.getElementById("progress"); if (b) b.classList.remove("on");
  });
})();

// ── ⌘K command palette: one keyboard entry point to every dashboard action ───
(function () {
  var box = document.getElementById("cmdk");
  if (!box) return;
  var input = box.querySelector(".cmdk-input");
  var list = box.querySelector(".cmdk-list");
  var filtered = [], active = 0;

  function taskValue() {
    var t = document.querySelector(".task-in");
    return t ? (t.value || "").trim() : "";
  }
  function focusEl(sel) { var el = document.querySelector(sel); if (el) el.focus(); }
  function clickEl(sel) { var el = document.querySelector(sel); if (el) el.click(); }
  function goTo(sel) {
    var el = document.querySelector(sel);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function dispatchTo(cid) {
    var task = taskValue();
    if (!task) { focusEl(".task-in"); return; }
    fetch("/api/dispatch", { method: "POST", body: new URLSearchParams({ character_id: cid, task: task }) })
      .then(function (r) { return r.text(); })
      .then(function (html) { var f = document.getElementById("fleet"); if (f) f.innerHTML = html; })
      .catch(function () {});
  }

  // Built fresh on each open so the model list reflects the live fleet.
  function commands() {
    var cmds = [
      { label: "Focus task box", hint: "dispatch", run: function () { focusEl(".task-in"); } },
      { label: "Auto-cast current task", hint: "✨", run: function () {
          if (taskValue()) clickEl(".autocast button:not([type=button])"); else focusEl(".task-in"); } },
      { label: "Send task to council", hint: "👥", run: function () {
          if (taskValue()) clickEl(".autocast button[type=button]"); else focusEl(".task-in"); } },
      { label: "Search runs", hint: "find", run: function () { focusEl(".search-in"); } },
      { label: "Toggle settings", hint: "config", run: function () {
          var d = document.querySelector(".settings-wrap"); if (d) d.open = !d.open; } },
      { label: "Reset office view", hint: "office", run: function () { clickEl(".iso-reset"); } },
      { label: "Go to Office", hint: "jump", run: function () { goTo(".fleet-panel"); } },
      { label: "Go to Trends", hint: "jump", run: function () { goTo(".chart-panel"); } },
      { label: "Go to Runs", hint: "jump", run: function () { goTo(".runs-panel"); } },
      { label: "Go to Audit log", hint: "jump", run: function () { goTo(".audit-panel"); } },
      { label: "Go to Live feed", hint: "jump", run: function () { goTo(".live"); } }
    ];
    document.querySelectorAll(".sprite[data-cid]").forEach(function (sp) {
      var cid = sp.getAttribute("data-cid");
      var tag = sp.querySelector(".tag");
      var name = tag ? tag.textContent : cid;
      cmds.push({ label: "Assign task to " + name, hint: cid, run: function () { dispatchTo(cid); } });
    });
    return cmds;
  }

  function render() {
    list.innerHTML = "";
    filtered.forEach(function (c, i) {
      var row = document.createElement("div");
      row.className = "cmdk-item" + (i === active ? " active" : "");
      row.innerHTML = '<span class="ck-label"></span><span class="ck-hint"></span>';
      row.querySelector(".ck-label").textContent = c.label;
      row.querySelector(".ck-hint").textContent = c.hint || "";
      row.addEventListener("mousedown", function (e) { e.preventDefault(); choose(i); });
      list.appendChild(row);
    });
  }
  function apply() {
    var q = input.value.trim().toLowerCase();
    var all = commands();
    filtered = q ? all.filter(function (c) {
      return (c.label + " " + (c.hint || "")).toLowerCase().indexOf(q) !== -1;
    }) : all;
    active = 0;
    render();
  }
  function choose(i) {
    var c = filtered[i];
    close();
    if (c) c.run();
  }
  function open() {
    box.hidden = false;
    input.value = "";
    apply();
    input.focus();
  }
  function close() { box.hidden = true; }
  function isOpen() { return !box.hidden; }

  document.addEventListener("keydown", function (e) {
    if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
      e.preventDefault(); isOpen() ? close() : open(); return;
    }
    if (!isOpen()) return;
    if (e.key === "Escape") { e.preventDefault(); close(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); active = Math.min(active + 1, filtered.length - 1); render(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); active = Math.max(active - 1, 0); render(); }
    else if (e.key === "Enter") { e.preventDefault(); choose(active); }
  });
  input.addEventListener("input", apply);
  box.addEventListener("mousedown", function (e) { if (e.target === box) close(); });
  document.body.addEventListener("click", function (e) {
    if (e.target.closest && e.target.closest("[data-cmdk-open]")) open();
  });
})();
