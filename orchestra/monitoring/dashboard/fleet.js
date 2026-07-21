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

  // ── Drag a model chip from the roster onto an AGENT node to set its model ───
  var dragCid = "";
  function swapFleet(html) { var f = document.getElementById("fleet"); if (f) f.innerHTML = html; }
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
  // a model can drop onto an agent node (sets its model) or a group node (adds to its panel)
  function dropTarget(el) {
    var node = el.closest && el.closest(".node");
    if (!node) return null;
    var t = node.getAttribute("data-node-type");
    return (t === "agent" || t === "discussion" || t === "consensus") ? node : null;
  }
  document.body.addEventListener("dragover", function (e) {
    var node = dropTarget(e.target);
    if (node) { e.preventDefault(); node.classList.add("drop-hot"); }
  });
  document.body.addEventListener("dragleave", function (e) {
    var node = dropTarget(e.target);
    if (node) node.classList.remove("drop-hot");
  });
  document.body.addEventListener("drop", function (e) {
    var node = dropTarget(e.target);
    if (!node) return;
    e.preventDefault();
    node.classList.remove("drop-hot");
    var cid = (dragCid || (e.dataTransfer && e.dataTransfer.getData("text/plain")) || "").trim();
    var nid = node.getAttribute("data-node");
    if (!cid || !nid) return;
    var group = node.getAttribute("data-node-type") !== "agent";
    var url = "/api/nodes/" + encodeURIComponent(nid) + (group ? "/panel" : "");
    var body = group ? { add: cid } : { default_model: cid };
    fetch(url, { method: "POST", body: new URLSearchParams(body) })
      .then(function (r) { return r.text(); })
      .then(function (html) { window._replaceNode(nid, html); })   // updates the node at once
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

// ── Live node status: patch members/running/output WITHOUT recreating nodes ──
// (so inline dropdowns, text edits and positions never get clobbered by refresh)
(function () {
  function apply() {
    if (!document.querySelector(".office.graph .node[data-node]")) return;
    fetch("/api/nodes/states").then(function (r) { return r.json(); }).then(function (states) {
      document.querySelectorAll(".office.graph .node[data-node]").forEach(function (n) {
        var s = states[n.getAttribute("data-node")];
        if (!s) return;
        n.classList.toggle("running", !!s.running);
        if (s.members != null) { var m = n.querySelector(".node-members"); if (m) m.innerHTML = s.members; }
        if (s.output != null) { var o = n.querySelector(".node-output"); if (o) o.textContent = s.output; }
      });
      var running = document.querySelectorAll(".office.graph .node.running").length;
      var st = document.querySelector(".graph-status");
      if (st) st.textContent = running ? "● running · " + running + " node" + (running > 1 ? "s" : "") : "";
      if (window._redrawWires) window._redrawWires();   // node heights may have changed → reattach wires
    }).catch(function () {});
  }
  setInterval(apply, 2000);
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (e.target && e.target.id === "fleet") setTimeout(apply, 80);
  });
})();

// ── Swap a single node in place (immediate update, no full-canvas re-render) ──
window._replaceNode = function (nid, html) {
  var old = document.querySelector('.node[data-node="' + nid + '"]');
  if (!old || !html || !html.trim()) return;
  var tmp = document.createElement("div"); tmp.innerHTML = html.trim();
  var fresh = tmp.firstElementChild;
  if (!fresh) return;
  old.parentNode.insertBefore(fresh, old);
  old.remove();
  if (window.htmx) { try { window.htmx.process(fresh); } catch (e) {} }
  window.dispatchEvent(new Event("resize"));   // redraw wires around the swapped node
};

// ── Inline node field save: persist a changed field via fetch, refresh that node ──
// Plain JS instead of htmx (the htmx select-change path raced with re-renders).
(function () {
  document.body.addEventListener("change", function (e) {
    var el = e.target;
    if (!el || !el.name || !el.closest || !el.hasAttribute("data-nf")) return;
    var node = el.closest(".node[data-node]");
    if (!node) return;
    var nid = node.getAttribute("data-node");
    var body = new URLSearchParams(); body.set(el.name, el.value);
    fetch("/api/nodes/" + encodeURIComponent(nid), { method: "POST", body: body })
      .then(function (r) { return r.text(); })
      .then(function (html) { window._replaceNode(nid, html); })   // node updates immediately
      .catch(function () {});
  });
  // remove a model from a group node's panel
  document.body.addEventListener("click", function (e) {
    var btn = e.target.closest && e.target.closest("[data-panel-remove]");
    if (!btn) return;
    var node = btn.closest(".node[data-node]");
    if (!node) return;
    var nid = node.getAttribute("data-node");
    fetch("/api/nodes/" + encodeURIComponent(nid) + "/panel", {
      method: "POST", body: new URLSearchParams({ remove: btn.getAttribute("data-panel-remove") })
    }).then(function (r) { return r.text(); })
      .then(function (html) { window._replaceNode(nid, html); }).catch(function () {});
  });
})();

// ── Blueprint canvas: draw wires port→port, drag nodes, connect ports ────────
(function () {
  var MARKER = '<defs><marker id="wire-arrow" markerWidth="9" markerHeight="9" refX="6" refY="3" ' +
    'orient="auto"><path d="M0,0 L6,3 L0,6 Z" class="wire-head"/></marker></defs>';
  function canvas() { return document.querySelector(".office.graph .graph-canvas"); }
  function nodeById(cv) {
    var m = {}; cv.querySelectorAll(".node[data-node]").forEach(function (n) { m[n.getAttribute("data-node")] = n; }); return m;
  }
  function outPos(n) { return { x: n.offsetLeft + n.offsetWidth, y: n.offsetTop + n.offsetHeight / 2 }; }
  function inPos(n) { return { x: n.offsetLeft, y: n.offsetTop + n.offsetHeight / 2 }; }
  function curve(a, b) {
    var dx = Math.max(40, Math.abs(b.x - a.x) / 2);
    return "M" + a.x + "," + a.y + " C" + (a.x + dx) + "," + a.y + " " + (b.x - dx) + "," + b.y + " " + b.x + "," + b.y;
  }
  function draw(extra) {
    var cv = canvas(); var svg = cv && cv.querySelector("svg.node-wires");
    if (!cv || !svg) return;
    svg.setAttribute("width", cv.scrollWidth); svg.setAttribute("height", cv.scrollHeight);
    svg.setAttribute("viewBox", "0 0 " + cv.scrollWidth + " " + cv.scrollHeight);
    var nodes = nodeById(cv), paths = "";
    cv.querySelectorAll(".node[data-connections]").forEach(function (src) {
      var sid = src.getAttribute("data-node");
      var live = src.classList.contains("running") ? " wire-active" : "";  // animate flow from a running node
      (src.getAttribute("data-connections") || "").split(",").forEach(function (tid) {
        var tgt = nodes[tid]; if (!tid || !tgt) return;
        var d = curve(outPos(src), inPos(tgt));
        paths += '<path class="wire' + live + '" marker-end="url(#wire-arrow)" d="' + d + '"/>';
        // wide transparent hit path → click to delete this edge
        paths += '<path class="wire-hit" data-src="' + sid + '" data-to="' + tid + '" d="' + d + '"/>';
      });
    });
    if (extra) paths += '<path class="wire wire-live" d="' + extra + '"/>';
    svg.innerHTML = MARKER + paths;
  }
  // click a wire to remove that edge
  document.body.addEventListener("click", function (e) {
    var hit = e.target.closest && e.target.closest(".wire-hit");
    if (!hit) return;
    var s = hit.getAttribute("data-src"), to = hit.getAttribute("data-to");
    if (!s || !to || !window.confirm("Remove this connection?")) return;
    fetch("/api/nodes/" + encodeURIComponent(s) + "/disconnect", {
      method: "POST", body: new URLSearchParams({ to: to })
    }).then(function (r) { return r.text(); })
      .then(function (h) { var f = document.getElementById("fleet"); if (f) f.innerHTML = h; })
      .catch(function () {});
  });
  var t;
  function schedule() { clearTimeout(t); t = setTimeout(function () { draw(); }, 40); }
  window._redrawWires = schedule;          // callable after node-status patches / replacements
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (e.target && (e.target.id === "fleet" || (e.target.querySelector && e.target.querySelector(".graph-canvas")))) schedule();
  });
  window.addEventListener("resize", schedule);
  // safety net: keep wires attached even if a node's height changes (members/output land)
  setInterval(function () { if (document.querySelector(".office.graph .node[data-connections]")) draw(); }, 1200);
  // after adding a node from the palette, scroll the newest node into view
  var scrollToNew = false;
  document.body.addEventListener("click", function (e) {
    if (e.target.closest && e.target.closest(".np-add")) scrollToNew = true;
  });
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (!scrollToNew || !e.target || e.target.id !== "fleet") return;
    scrollToNew = false;
    setTimeout(function () {
      var nodes = document.querySelectorAll(".graph-canvas .node");
      if (nodes.length) nodes[nodes.length - 1].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    }, 60);
  });

  // ── drag a node by its header to reposition it (persist on drop) ──
  var drag = null;
  var pan = null;
  document.body.addEventListener("pointerdown", function (e) {
    var port = e.target.closest && e.target.closest(".port.out");
    if (port) { startWire(e, port); return; }
    // inline controls (title input, selects, textareas, delete button) must not start a drag
    if (e.target.closest && e.target.closest("input, select, textarea, button")) return;
    var head = e.target.closest && e.target.closest(".node-head");
    if (head) {
      var node = head.closest(".node"); var cv = canvas(); if (!node || !cv) return;
      drag = { node: node, sx: e.clientX, sy: e.clientY, ox: node.offsetLeft, oy: node.offsetTop };
      node.classList.add("dragging"); node.style.zIndex = 50;
      e.preventDefault(); return;
    }
    // press on empty canvas background with the primary button → pan (scroll) the view
    if (e.button !== 0) return;
    var cvp = e.target.closest && e.target.closest(".graph-canvas");
    if (cvp && !(e.target.closest(".node"))) {
      pan = { sx: e.clientX, sy: e.clientY, sl: cvp.scrollLeft, st: cvp.scrollTop, cv: cvp };
      cvp.classList.add("panning");
    }
  });
  // ── drag from an output port to another node's input port to connect ──
  var wire = null;
  function startWire(e, port) {
    var cv = canvas(); var node = port.closest(".node"); if (!cv || !node) return;
    wire = { from: node.getAttribute("data-node"), a: outPos(node) };
    e.preventDefault();
  }
  // clear any in-progress interaction (persisting a drag). Runs on pointerup's siblings:
  // pointercancel / window blur / a self-heal when the button is no longer held — without
  // this a cancelled pointer leaves pan/drag/wire STUCK and eats all further pointer moves.
  function _endInteraction(persist) {
    if (pan && pan.cv) pan.cv.classList.remove("panning");
    if (drag && drag.node) {
      drag.node.classList.remove("dragging"); drag.node.style.zIndex = "";
      if (persist) {
        var nid = drag.node.getAttribute("data-node");
        fetch("/api/nodes/" + encodeURIComponent(nid) + "/move", {
          method: "POST",
          body: new URLSearchParams({ x: parseInt(drag.node.style.left, 10) || 0, y: parseInt(drag.node.style.top, 10) || 0 })
        }).catch(function () {});
      }
    }
    pan = null; drag = null; wire = null; draw();
  }
  document.body.addEventListener("pointercancel", function () { _endInteraction(true); });
  window.addEventListener("blur", function () { _endInteraction(true); });

  document.body.addEventListener("pointermove", function (e) {
    // self-heal: if a state is stuck but no button is held, end it (recovers from a missed up)
    if ((pan || drag || wire) && e.buttons === 0) { _endInteraction(true); return; }
    if (pan) {
      pan.cv.scrollLeft = pan.sl - (e.clientX - pan.sx);
      pan.cv.scrollTop = pan.st - (e.clientY - pan.sy);
      return;
    }
    var cv = canvas(); if (!cv) return;
    if (drag) {
      var nx = drag.ox + (e.clientX - drag.sx), ny = drag.oy + (e.clientY - drag.sy);
      drag.node.style.left = Math.max(0, nx) + "px"; drag.node.style.top = Math.max(0, ny) + "px";
      draw(); return;
    }
    if (wire) {
      var r = cv.getBoundingClientRect();
      draw(curve(wire.a, { x: e.clientX - r.left + cv.scrollLeft, y: e.clientY - r.top + cv.scrollTop }));
    }
  });
  document.body.addEventListener("pointerup", function (e) {
    if (pan) { pan.cv.classList.remove("panning"); pan = null; }
    if (drag) {
      var node = drag.node; node.classList.remove("dragging"); node.style.zIndex = "";
      var nid = node.getAttribute("data-node");
      fetch("/api/nodes/" + encodeURIComponent(nid) + "/move", {
        method: "POST", body: new URLSearchParams({ x: parseInt(node.style.left, 10) || 0, y: parseInt(node.style.top, 10) || 0 })
      }).catch(function () {});
      drag = null;
    }
    if (wire) {
      var tgtPort = e.target.closest && e.target.closest(".port.in");
      var tgtNode = tgtPort && tgtPort.closest(".node");
      if (tgtNode) {
        var to = tgtNode.getAttribute("data-node");
        fetch("/api/nodes/" + encodeURIComponent(wire.from) + "/connect", {
          method: "POST", body: new URLSearchParams({ to: to })
        }).then(function (r) { return r.text(); }).then(function (h) { var f = document.getElementById("fleet"); if (f) f.innerHTML = h; }).catch(function () {});
      }
      wire = null; draw();
    }
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
