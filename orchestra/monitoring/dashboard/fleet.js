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

  // ── Drag a model chip from the roster into a task room to assign it ─────────
  // Dropping onto a room dispatches that room's task to the model; dropping onto
  // the draft room uses the task typed in the input above. Multiple models can be
  // dropped into the same room to collaborate on one task.
  var dragCid = "";
  function taskInput() {
    var i = document.querySelector("[data-drag-task]");
    return i ? (i.value || "").trim() : "";
  }
  document.body.addEventListener("dragstart", function (e) {
    var chip = e.target.closest && e.target.closest(".roster-chip[data-cid]");
    if (!chip) { dragCid = ""; return; }
    dragCid = chip.getAttribute("data-cid");
    chip.classList.add("dragging");
    if (e.dataTransfer) {
      e.dataTransfer.setData("text/plain", dragCid);
      e.dataTransfer.effectAllowed = "copy";
    }
  });
  document.body.addEventListener("dragend", function (e) {
    var chip = e.target.closest && e.target.closest(".roster-chip");
    if (chip) chip.classList.remove("dragging");
  });
  document.body.addEventListener("dragover", function (e) {
    var room = e.target.closest && e.target.closest("[data-drop]");
    if (room && (dragCid || (e.dataTransfer && e.dataTransfer.types.indexOf("text/plain") !== -1))) {
      e.preventDefault(); room.classList.add("drop-hot");
    }
  });
  document.body.addEventListener("dragleave", function (e) {
    var room = e.target.closest && e.target.closest("[data-drop]");
    if (room) room.classList.remove("drop-hot");
  });
  document.body.addEventListener("drop", function (e) {
    var room = e.target.closest && e.target.closest("[data-drop]");
    if (!room) return;
    e.preventDefault();
    room.classList.remove("drop-hot");
    var cid = (dragCid || (e.dataTransfer && e.dataTransfer.getData("text/plain")) || "").trim();
    var roomId = room.getAttribute("data-room") || "";
    var task = taskInput();
    if (!cid) return;
    if (!task) { var i = document.querySelector("[data-drag-task]"); if (i) i.focus(); return; }
    fetch("/api/dispatch", {
      method: "POST",
      body: new URLSearchParams({ character_id: cid, task: task, room_id: roomId })
    })
      .then(function (r) { return r.text(); })
      .then(function (html) { var f = document.getElementById("fleet"); if (f) f.innerHTML = html; })
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

// ── Live dispatch output: stream tokens into the terminal viewer ────────────
// When a running dispatch's output view is swapped into #output, open an
// EventSource and append tokens as they arrive (mirrors the chat stream path).
(function () {
  var current = null;   // active EventSource, closed when a new view loads
  function connect(pre) {
    if (current) { try { current.close(); } catch (e) {} current = null; }
    var did = pre.getAttribute("data-stream-did");
    if (!did) return;
    var es = new EventSource("/api/dispatch/" + encodeURIComponent(did) + "/stream");
    current = es;
    es.addEventListener("token", function (ev) {
      pre.appendChild(document.createTextNode(ev.data));
      var panel = document.getElementById("output");
      if (panel) panel.scrollTop = panel.scrollHeight;
    });
    es.addEventListener("done", function () {
      es.close();
      if (current === es) current = null;
      // refresh the inbox so the item flips running → done
      if (window.htmx) { var ib = document.getElementById("inbox"); if (ib) window.htmx.trigger(ib, "sse:run_update"); }
    });
    es.onerror = function () { es.close(); if (current === es) current = null; };
  }
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (!e.target) return;
    var pre = e.target.querySelector ? e.target.querySelector("pre.ov-body.live[data-stream-did]") : null;
    if (pre) connect(pre);
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

// ── Routing wires: draw arrows between connected room cards on the board ─────
(function () {
  function draw() {
    var floor = document.querySelector(".office.board .rooms-floor");
    var svg = floor && floor.querySelector("svg.room-wires");
    if (!floor || !svg) return;
    var w = floor.scrollWidth, h = floor.scrollHeight;
    svg.setAttribute("width", w); svg.setAttribute("height", h);
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    var cards = {};
    floor.querySelectorAll(".room[data-room]").forEach(function (c) {
      cards[c.getAttribute("data-room")] = c;
    });
    var paths = "";
    floor.querySelectorAll(".room[data-connections]").forEach(function (src) {
      (src.getAttribute("data-connections") || "").split(",").forEach(function (tid) {
        var tgt = cards[tid];
        if (!tid || !tgt) return;
        var x1 = src.offsetLeft + src.offsetWidth / 2, y1 = src.offsetTop + src.offsetHeight;
        var x2 = tgt.offsetLeft + tgt.offsetWidth / 2, y2 = tgt.offsetTop;
        var my = (y1 + y2) / 2;
        paths += '<path class="wire" marker-end="url(#wire-arrow)" d="M' + x1 + ',' + y1 +
                 ' C' + x1 + ',' + my + ' ' + x2 + ',' + my + ' ' + x2 + ',' + y2 + '"/>';
      });
    });
    svg.innerHTML = '<defs><marker id="wire-arrow" markerWidth="9" markerHeight="9" refX="6" refY="3" ' +
      'orient="auto"><path d="M0,0 L6,3 L0,6 Z" class="wire-head"/></marker></defs>' + paths;
  }
  var t;
  function schedule() { clearTimeout(t); t = setTimeout(draw, 50); }
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (e.target && (e.target.id === "fleet" || e.target.querySelector && e.target.querySelector(".rooms-floor"))) schedule();
  });
  window.addEventListener("resize", schedule);
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
      { label: "Go to Office", hint: "jump", run: function () { goTo(".fleet-panel"); } },
      { label: "Go to Trends", hint: "jump", run: function () { goTo(".chart-panel"); } },
      { label: "Go to Runs", hint: "jump", run: function () { goTo(".runs-panel"); } },
      { label: "Go to Audit log", hint: "jump", run: function () { goTo(".audit-panel"); } },
      { label: "Go to Live feed", hint: "jump", run: function () { goTo(".live"); } }
    ];
    document.querySelectorAll(".roster-chip[data-cid]").forEach(function (chip) {
      var cid = chip.getAttribute("data-cid");
      var nm = chip.querySelector(".rc-name");
      var name = nm ? nm.textContent : cid;
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
