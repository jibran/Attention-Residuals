/**
 * AttnRes Chat — frontend application logic.
 *
 * Handles:
 * - Theme toggle (light / dark, persisted in localStorage)
 * - Sidebar toggle
 * - Slider sync with displayed value labels
 * - Model metadata display after selection
 * - Chat message rendering with streaming cursor effect
 * - POST /api/generate request and response rendering
 * - Error toast notifications
 * - Auto-resizing textarea
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 */

"use strict";

// ── State ──────────────────────────────────────────────────────────────────

let _generating = false;
let _modelData  = {};   // model_id → metadata dict from /api/models

// ── DOM references ─────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);

const themeToggle   = $("#themeToggle");
const sidebarToggle = $("#sidebarToggle");
const sidebar       = $("#sidebar");
const modelSelect   = $("#modelSelect");
const modelMeta     = $("#modelMeta");
const tempSlider    = $("#tempSlider");
const tempVal       = $("#tempVal");
const tokensSlider  = $("#tokensSlider");
const tokensVal     = $("#tokensVal");
const topkSlider    = $("#topkSlider");
const topkVal       = $("#topkVal");
const kvCacheToggle = $("#kvCacheToggle");
const clearBtn      = $("#clearBtn");
const messages      = $("#messages");
const promptInput   = $("#promptInput");
const sendBtn       = $("#sendBtn");

// ── Theme ──────────────────────────────────────────────────────────────────

/**
 * Apply the given theme to the document root and update the toggle icons.
 * @param {string} theme - "dark" or "light"
 */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  $(".icon-moon").style.display = theme === "dark"  ? "" : "none";
  $(".icon-sun").style.display  = theme === "light" ? "" : "none";
  localStorage.setItem("theme", theme);
}

applyTheme(localStorage.getItem("theme") || "dark");

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  applyTheme(next);
});

// ── Sidebar ─────────────────────────────────────────────────────────────────

sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("collapsed"));

// ── Slider sync ─────────────────────────────────────────────────────────────

function syncSlider(slider, label) {
  label.textContent = slider.value;
  slider.addEventListener("input", () => (label.textContent = slider.value));
}
syncSlider(tempSlider, tempVal);
syncSlider(tokensSlider, tokensVal);
syncSlider(topkSlider, topkVal);

// ── Model metadata ──────────────────────────────────────────────────────────

/**
 * Fetch model list from the API and populate the select element.
 */
async function loadModels() {
  try {
    const resp = await fetch("/api/models");
    if (!resp.ok) return;
    const models = await resp.json();
    _modelData = {};
    models.forEach((m) => (_modelData[m.model_id] = m));

    modelSelect.innerHTML = "";
    if (models.length === 0) {
      modelSelect.innerHTML = '<option value="">No checkpoints found</option>';
      return;
    }
    models.forEach((m) => {
      const opt   = document.createElement("option");
      opt.value   = m.model_id;
      opt.textContent = m.name;
      modelSelect.appendChild(opt);
    });
    showModelMeta(modelSelect.value);
  } catch (_) {
    /* silently ignore — models list may load later */
  }
}

/**
 * Render metadata for the currently selected model in the sidebar.
 * @param {string} modelId
 */
function showModelMeta(modelId) {
  const m = _modelData[modelId];
  if (!m) { modelMeta.classList.remove("visible"); return; }
  const lines = [];
  if (m.architecture)  lines.push(`<strong>Architecture:</strong> ${m.architecture}`);
  if (m.params_fmt)    lines.push(`<strong>Parameters:</strong> ${m.params_fmt}`);
  if (m.val_ppl)       lines.push(`<strong>Val PPL:</strong> ${m.val_ppl}`);
  if (m.epoch)         lines.push(`<strong>Epoch:</strong> ${m.epoch}`);
  if (m.dataset)       lines.push(`<strong>Dataset:</strong> ${m.dataset}`);
  modelMeta.innerHTML = lines.join("<br>");
  modelMeta.classList.toggle("visible", lines.length > 0);
}

modelSelect.addEventListener("change", () => showModelMeta(modelSelect.value));
loadModels();

// ── Auto-resize textarea ────────────────────────────────────────────────────

promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + "px";
});

// ── Toast notifications ─────────────────────────────────────────────────────

/**
 * Show a temporary error toast at the bottom-right of the screen.
 * @param {string} msg - Message text.
 * @param {number} [ms=4000] - Display duration in milliseconds.
 */
function showToast(msg, ms = 4000) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

// ── Message rendering ───────────────────────────────────────────────────────

/**
 * Remove the welcome placeholder from the messages area.
 */
function removeWelcome() {
  const w = messages.querySelector(".welcome-msg");
  if (w) w.remove();
}

/**
 * Append a user message bubble to the conversation.
 * @param {string} text
 */
function appendUserMessage(text) {
  removeWelcome();
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="msg-avatar">U</div>
    <div class="msg-body">
      <div class="msg-bubble">${escapeHtml(text)}</div>
    </div>`;
  messages.appendChild(row);
  scrollToBottom();
}

/**
 * Append an assistant message bubble and return handles to update it.
 * @param {string} [initialText=""]
 * @returns {{ bubble: HTMLElement, meta: HTMLElement }}
 */
function appendAssistantMessage(initialText = "") {
  removeWelcome();
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="msg-avatar">✦</div>
    <div class="msg-body">
      <div class="msg-bubble streaming">${escapeHtml(initialText)}</div>
      <div class="msg-meta"></div>
    </div>`;
  messages.appendChild(row);
  scrollToBottom();
  return {
    bubble: row.querySelector(".msg-bubble"),
    meta:   row.querySelector(".msg-meta"),
  };
}

/**
 * Escape HTML special characters.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Scroll the messages container to the bottom. */
function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

// ── Clear conversation ──────────────────────────────────────────────────────

clearBtn.addEventListener("click", () => {
  messages.innerHTML = `
    <div class="welcome-msg">
      <div class="welcome-icon">✦</div>
      <h2>AttnRes Language Model</h2>
      <p>Select a model and start typing to generate text.</p>
    </div>`;
});

// ── Generation ──────────────────────────────────────────────────────────────

/**
 * Read control values and POST to /api/generate.
 */
async function generate() {
  if (_generating) return;

  const modelId = modelSelect.value;
  if (!modelId) { showToast("No model selected."); return; }

  const prompt = promptInput.value.trim();
  if (!prompt) return;

  _generating = true;
  sendBtn.disabled = true;
  promptInput.value = "";
  promptInput.style.height = "auto";

  appendUserMessage(prompt);
  const { bubble, meta } = appendAssistantMessage();

  // Fake token-by-token appearance by revealing text in chunks
  let dotCount = 0;
  const thinkTimer = setInterval(() => {
    bubble.textContent = "Generating" + ".".repeat((dotCount++ % 3) + 1);
  }, 400);

  try {
    const resp = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id:       modelId,
        prompt:         prompt,
        max_new_tokens: parseInt(tokensSlider.value, 10),
        temperature:    parseFloat(tempSlider.value),
        top_k:          parseInt(topkSlider.value, 10),
        use_kv_cache:   kvCacheToggle.checked,
      }),
    });

    clearInterval(thinkTimer);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      bubble.classList.remove("streaming");
      bubble.textContent = "⚠ " + (err.detail || "Generation failed.");
      showToast(err.detail || "Generation failed.");
      return;
    }

    const data = await resp.json();

    // Reveal continuation character-by-character for a streaming feel
    bubble.classList.add("streaming");
    bubble.textContent = "";
    const text = data.generated;
    let i = 0;
    const revealTimer = setInterval(() => {
      const chunk = Math.min(8, text.length - i);
      bubble.textContent += text.slice(i, i + chunk);
      i += chunk;
      scrollToBottom();
      if (i >= text.length) {
        clearInterval(revealTimer);
        bubble.classList.remove("streaming");
        renderMeta(meta, data);
      }
    }, 16);

  } catch (err) {
    clearInterval(thinkTimer);
    bubble.classList.remove("streaming");
    bubble.textContent = "⚠ Network error: " + err.message;
    showToast("Network error: " + err.message);
  } finally {
    _generating = false;
    sendBtn.disabled = false;
    promptInput.focus();
  }
}

/**
 * Render timing / stats metadata below an assistant bubble.
 * @param {HTMLElement} meta - Container element.
 * @param {Object} data - API response object.
 */
function renderMeta(meta, data) {
  const badges = [
    `${data.new_tokens} tokens`,
    `${data.tok_per_sec} tok/s`,
    `${data.ms_per_tok} ms/tok`,
    `${data.elapsed_s}s`,
    data.use_kv_cache ? "KV cache ✓" : "KV cache ✗",
    data.model_id,
  ];
  meta.innerHTML = badges.map((b) => `<span class="meta-badge">${b}</span>`).join("");
}

// ── Keyboard handling ───────────────────────────────────────────────────────

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    generate();
  }
});

sendBtn.addEventListener("click", generate);
