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
