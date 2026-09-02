// TRO Chatbot frontend — talks ONLY to the existing ClarificationService
// endpoint. No extraction, scoring, or clarification logic lives here.

const API_BASE = ""; // same-origin when served from FastAPI (see main.py mount)

const CATEGORY_QUICK_START = [
  { label: "Wifi / Internet", text: "I'm having a problem connecting to the campus wifi." },
  { label: "Microsoft Teams", text: "I'm having a problem with Microsoft Teams." },
  { label: "VIT Email", text: "I'm having a problem with my VIT email." },
  { label: "AD Account", text: "I'm having a problem with my AD account." },
  { label: "Printer", text: "I'm having a problem with a printer." },
];

// Cosmetic-only label mapping (matches app/config/category_profiles.py
// CATEGORY_DISPLAY_TO_KEY, reversed). Purely for display — never used to
// make any decision.
const CATEGORY_LABELS = {
  wifi_internet: "Wifi/Internet Support",
  ms_teams: "Microsoft Teams Support",
  vit_email: "VIT Email Support",
  ad_account_creation: "AD Account Creation",
  printer_support: "Printer Support",
  general: "General",
};

const GENERIC_QUICK_REPLIES = ["Yes", "No", "Not sure"];

// ---- State ----
let conversationId = null;
let isBusy = false;
let isConversationOver = false;

// ---- DOM refs ----
const el = {
  welcomeState: document.getElementById("welcomeState"),
  chatState: document.getElementById("chatState"),
  categoryChips: document.getElementById("categoryChips"),
  welcomeInput: document.getElementById("welcomeInput"),
  welcomeSendBtn: document.getElementById("welcomeSendBtn"),
  messages: document.getElementById("messages"),
  quickReplies: document.getElementById("quickReplies"),
  completionCard: document.getElementById("completionCard"),
  composerForm: document.getElementById("composerForm"),
  composerInput: document.getElementById("composerInput"),
  composerSendBtn: document.getElementById("composerSendBtn"),
  turnBadge: document.getElementById("turnBadge"),
  newChatBtn: document.getElementById("newChatBtn"),
  statusDot: document.getElementById("statusDot"),
  statusLabel: document.getElementById("statusLabel"),
};

// ---- Init ----
function init() {
  renderCategoryChips();
  checkHealth();

  el.welcomeSendBtn.addEventListener("click", () => {
    const text = el.welcomeInput.value.trim();
    if (text) beginConversation(text);
  });
  el.welcomeInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const text = el.welcomeInput.value.trim();
      if (text) beginConversation(text);
    }
  });

  el.composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = el.composerInput.value.trim();
    if (text) sendTurn(text);
  });

  el.newChatBtn.addEventListener("click", resetConversation);
}

function renderCategoryChips() {
  el.categoryChips.innerHTML = "";
  CATEGORY_QUICK_START.forEach((c) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = c.label;
    chip.addEventListener("click", () => beginConversation(c.text));
    el.categoryChips.appendChild(chip);
  });
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      el.statusDot.classList.add("online");
      el.statusLabel.textContent = "backend online";
    } else {
      throw new Error("unhealthy");
    }
  } catch {
    el.statusDot.classList.add("offline");
    el.statusLabel.textContent = "backend unreachable";
  }
}

// ---- Conversation lifecycle ----
function beginConversation(firstMessage) {
  conversationId = crypto.randomUUID();
  isConversationOver = false;
  el.messages.innerHTML = "";
  el.completionCard.classList.add("hidden");
  el.completionCard.innerHTML = "";
  el.welcomeState.classList.add("hidden");
  el.chatState.classList.remove("hidden");
  el.composerInput.disabled = false;
  el.composerSendBtn.disabled = false;
  sendTurn(firstMessage);
}

function resetConversation() {
  conversationId = null;
  isConversationOver = false;
  el.messages.innerHTML = "";
  el.quickReplies.innerHTML = "";
  el.quickReplies.classList.add("hidden");
  el.completionCard.classList.add("hidden");
  el.completionCard.innerHTML = "";
  el.welcomeInput.value = "";
  el.composerInput.value = "";
  el.chatState.classList.add("hidden");
  el.welcomeState.classList.remove("hidden");
  updateTurnBadge(0);
}

// ---- Sending a turn ----
async function sendTurn(text) {
  if (isBusy || isConversationOver) return;
  isBusy = true;

  addUserMessage(text);
  el.composerInput.value = "";
  el.quickReplies.classList.add("hidden");
  setComposerEnabled(false);

  const typingEl = addTypingIndicator();

  try {
    const res = await fetch(
      `${API_BASE}/conversations/${encodeURIComponent(conversationId)}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      }
    );

    removeTypingIndicator(typingEl);

    if (!res.ok) {
      const body = await safeJson(res);
      addErrorMessage(
        res.status === 503
          ? `The AI provider had a problem: ${body?.detail || "unknown error"}`
          : `Request failed (${res.status}): ${body?.detail || res.statusText}`
      );
      setComposerEnabled(true);
      isBusy = false;
      return;
    }

    const data = await res.json();
    handleResponse(data);
  } catch (err) {
    removeTypingIndicator(typingEl);
    addErrorMessage(`Couldn't reach the server: ${err.message}`);
    setComposerEnabled(true);
  }

  isBusy = false;
}

function handleResponse(data) {
  updateTurnBadge(data.turn);

  switch (data.action) {
    case "ASK_CLARIFICATION":
      addAssistantMessage(data.question, data);
      renderQuickReplies();
      setComposerEnabled(true);
      break;

    case "RECHECK":
      addSystemMessage(
        "Let me double-check that against what you've told me so far…",
        data
      );
      setComposerEnabled(true);
      break;

    case "READY":
      addAssistantMessage(
        "Thanks — I have enough detail to route this. Here's a summary of your ticket:",
        data
      );
      showCompletionCard("ready", data);
      endConversationInput();
      break;

    case "ESCALATE":
      addAssistantMessage(
        "This needs a closer look from a support staff member, so I'm escalating it.",
        data
      );
      showCompletionCard("escalated", data);
      endConversationInput();
      break;

    case "DEFLECTED":
      addAssistantMessage(
        "Good news — I found a knowledge base article that should resolve this without needing a ticket:",
        data
      );
      showCompletionCard("resolved", data);
      endConversationInput();
      break;

    default:
      addErrorMessage(`Unrecognized action from server: ${data.action}`);
      setComposerEnabled(true);
  }
}

function endConversationInput() {
  isConversationOver = true;
  setComposerEnabled(false);
  el.quickReplies.classList.add("hidden");
}

function setComposerEnabled(enabled) {
  el.composerInput.disabled = !enabled;
  el.composerSendBtn.disabled = !enabled;
  if (enabled) el.composerInput.focus();
}

// ---- Rendering ----
function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="bubble user"></div>`;
  row.querySelector(".bubble").textContent = text;
  el.messages.appendChild(row);
  scrollToBottom();
}

function addAssistantMessage(text, decision) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";

  const textEl = document.createElement("div");
  textEl.textContent = text;
  bubble.appendChild(textEl);

  if (decision) {
    const toggle = document.createElement("button");
    toggle.className = "reasoning-toggle";
    toggle.type = "button";
    toggle.textContent = "🔍 Why? (reasoning trace)";

    const panel = document.createElement("div");
    panel.className = "reasoning-panel hidden";
    panel.innerHTML = `
      <div><strong>Reasoning:</strong> ${escapeHtml(decision.reasoning || "—")}</div>
      <div style="margin-top:4px;"><strong>Category:</strong> ${
        decision.category ? escapeHtml(CATEGORY_LABELS[decision.category] || decision.category) : "undetermined"
      } &nbsp; <strong>Completeness:</strong> ${Math.round((decision.completeness_score || 0) * 100)}% &nbsp; <strong>Confidence:</strong> ${Math.round((decision.confidence || 0) * 100)}%</div>
      ${
        decision.affected_fields && decision.affected_fields.length
          ? `<div style="margin-top:4px;">${decision.affected_fields
              .map((f) => `<span class="tag">${escapeHtml(f)}</span>`)
              .join("")}</div>`
          : ""
      }
    `;

    toggle.addEventListener("click", () => panel.classList.toggle("hidden"));
    bubble.appendChild(toggle);
    bubble.appendChild(panel);
  }

  row.appendChild(bubble);
  el.messages.appendChild(row);
  scrollToBottom();
}

function addSystemMessage(text, decision) {
  const row = document.createElement("div");
  row.className = "msg-row system";
  const bubble = document.createElement("div");
  bubble.className = "bubble system";
  bubble.textContent = text;
  row.appendChild(bubble);
  el.messages.appendChild(row);
  if (decision) addAssistantMessage(decision.reasoning || "(no further detail)", decision);
  scrollToBottom();
}

function addErrorMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble error";
  bubble.textContent = `⚠ ${text}`;
  row.appendChild(bubble);
  el.messages.appendChild(row);
  scrollToBottom();
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `<div class="bubble assistant typing-bubble"><span></span><span></span><span></span></div>`;
  el.messages.appendChild(row);
  scrollToBottom();
  return row;
}

function removeTypingIndicator(rowEl) {
  if (rowEl && rowEl.parentNode) rowEl.parentNode.removeChild(rowEl);
}

function renderQuickReplies() {
  el.quickReplies.innerHTML = "";
  GENERIC_QUICK_REPLIES.forEach((label) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = label;
    chip.addEventListener("click", () => sendTurn(label));
    el.quickReplies.appendChild(chip);
  });
  el.quickReplies.classList.remove("hidden");
}

function showCompletionCard(kind, data) {
  el.completionCard.className = `completion-card ${kind}`;
  const title =
    kind === "ready"
      ? "✅ Ticket Ready"
      : kind === "resolved"
      ? "📚 Resolved via Knowledge Base"
      : "🧑‍💼 Escalated to Human Support";
  const categoryLabel = data.category
    ? CATEGORY_LABELS[data.category] || data.category
    : "Undetermined";

  const resolutionBlock =
    kind === "resolved" && data.kb_offered_resolution
      ? `<div class="row" style="display:block;">
            <strong>${escapeHtml(data.kb_matched_title || "Suggested fix")}</strong>
            <pre style="white-space:pre-wrap;margin:6px 0 0 0;font:inherit;">${escapeHtml(
              data.kb_offered_resolution
            )}</pre>
          </div>`
      : "";

  el.completionCard.innerHTML = `
    <h3>${title}</h3>
    ${resolutionBlock}
    <div class="row"><span>Category</span><span>${escapeHtml(categoryLabel)}</span></div>
    <div class="row"><span>Completeness</span><span>${Math.round(
      (data.completeness_score || 0) * 100
    )}%</span></div>
    <div class="row"><span>Confidence</span><span>${Math.round(
      (data.confidence || 0) * 100
    )}%</span></div>
    <div class="row"><span>Conversation turns</span><span>${data.turn}</span></div>
  `;
  el.completionCard.classList.remove("hidden");
}

function updateTurnBadge(turn) {
  el.turnBadge.textContent = `Turn ${turn}`;
}

function scrollToBottom() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

async function safeJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

init();