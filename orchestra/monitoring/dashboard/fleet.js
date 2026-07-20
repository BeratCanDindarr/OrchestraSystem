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
    var card = e.target.closest && e.target.closest(".char-card[data-cid]");
    if (card) { e.preventDefault(); card.classList.add("drop-hot"); }
  });
  document.body.addEventListener("dragleave", function (e) {
    var card = e.target.closest && e.target.closest(".char-card[data-cid]");
    if (card) card.classList.remove("drop-hot");
  });
  document.body.addEventListener("drop", function (e) {
    var card = e.target.closest && e.target.closest(".char-card[data-cid]");
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
