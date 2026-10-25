// TRO Chatbot frontend — talks ONLY to the backend API (conversation,
// deflection-feedback, duplicate-feedback and audit endpoints). No extraction, scoring,
// clarification or routing logic lives here; it only renders what the
// server decided. Every server value goes through escapeHtml().

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

// Human wording for review-queue reasons (app/services/ticket_pipeline.py,
// app/services/clarification_service.py). Unknown reasons are shown as-is.
const REVIEW_REASON_LABELS = {
  low_classification_confidence: "the automatic routing wasn't confident enough",
  clarification_escalated: "we couldn't gather enough detail in the chat",
};

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
  // A new message supersedes any open KB suggestion (and its feedback buttons).
  el.completionCard.classList.add("hidden");
  el.completionCard.innerHTML = "";
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

  // Agent 4 suspects a duplicate: nothing is merged until the user answers.
  if (data.awaiting_duplicate_confirmation) {
    showDuplicateQuestion(data);
    return;
  }

  // Agents 4-6 already reached a final outcome this turn (ticket created,
  // merged, sent to review, ...): show it and close the conversation.
  if (data.terminal_outcome) {
    addAssistantMessage(outcomeIntro(data), data);
    showOutcomeCard(data);
    endConversationInput();
    return;
  }

  switch (data.action) {
    case "ASK_CLARIFICATION":
      addAssistantMessage(data.question, data);
      if (data.kb_outcome === "WEAK_MATCH" && data.kb_matched_title) {
        addKbHint(data.kb_matched_title);
      }
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
      // Not final yet: the user's feedback decides between SELF_RESOLVED
      // and creating a ticket, so the composer stays usable.
      addAssistantMessage(
        "Good news — I found a knowledge base article that should resolve this without needing a ticket:",
        data
      );
      showCompletionCard("resolved", data);
      addFeedbackButtons();
      setComposerEnabled(true);
      break;

    default:
      addErrorMessage(`Unrecognized action from server: ${data.action}`);
      setComposerEnabled(true);
  }
}

// ---- Deflection feedback ----
function addFeedbackButtons() {
  const row = document.createElement("div");
  row.className = "feedback-row";
  row.innerHTML = `
    <span>Did this fix your issue?</span>
    <button type="button" class="btn-primary" data-resolved="true">That fixed it</button>
    <button type="button" class="btn-secondary" data-resolved="false">Still not working</button>
  `;
  row.querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () =>
      sendDeflectionFeedback(btn.dataset.resolved === "true", btn.textContent)
    )
  );
  el.completionCard.appendChild(row);
}

async function sendDeflectionFeedback(resolved, label) {
  if (isBusy || isConversationOver) return;
  isBusy = true;
  const buttons = el.completionCard.querySelectorAll(".feedback-row button");
  buttons.forEach((b) => (b.disabled = true));
  setComposerEnabled(false);
  addUserMessage(label);
  const typingEl = addTypingIndicator();

  try {
    const res = await fetch(
      `${API_BASE}/conversations/${encodeURIComponent(conversationId)}/deflection-feedback`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resolved }),
      }
    );
    removeTypingIndicator(typingEl);
    const body = await safeJson(res);

    if (!res.ok) {
      addErrorMessage(`Request failed (${res.status}): ${body?.detail || res.statusText}`);
      buttons.forEach((b) => (b.disabled = false));
      setComposerEnabled(true);
    } else if (body.awaiting_duplicate_confirmation) {
      showDuplicateQuestion(body);
    } else if (body.terminal_outcome) {
      addAssistantMessage(outcomeIntro(body), null);
      showOutcomeCard(body);
      endConversationInput();
    } else {
      // No ticket pipeline configured on the server - nothing more to show.
      addAssistantMessage("Thanks for letting us know.", null);
      setComposerEnabled(true);
    }
  } catch (err) {
    removeTypingIndicator(typingEl);
    addErrorMessage(`Couldn't reach the server: ${err.message}`);
    buttons.forEach((b) => (b.disabled = false));
    setComposerEnabled(true);
  }
  isBusy = false;
}

// ---- Suspected duplicate (Agent 4) ----
// The server only sends the candidate's id, title and created time. The
// question is rebuilt here so the time shows in the browser's local time
// (the server's duplicate_question uses UTC and is only a fallback).
function formatLocalTime(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleString(undefined, {
    day: "numeric", month: "short", hour: "numeric", minute: "2-digit",
  });
}

function duplicateQuestionText(data) {
  const c = data.duplicate_candidate;
  if (!c) return data.duplicate_question || "This looks like an existing ticket. Is this the same issue?";
  const label = c.title ? `${c.ticket_id} - ${c.title}` : c.ticket_id;
  const opened = formatLocalTime(c.created_at);
  return `This looks like an existing ticket: ${label}${opened ? ` (opened ${opened})` : ""}. Is this the same issue?`;
}

function showDuplicateQuestion(data) {
  addAssistantMessage(duplicateQuestionText(data), null);
  el.quickReplies.classList.add("hidden");
  el.completionCard.className = "completion-card duplicate";
  el.completionCard.innerHTML = "";
  const row = document.createElement("div");
  row.className = "feedback-row";
  row.innerHTML = `
    <span>Is this the same issue?</span>
    <button type="button" class="btn-primary" data-duplicate="true">Yes, same issue</button>
    <button type="button" class="btn-secondary" data-duplicate="false">No, it's different</button>
  `;
  row.querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () =>
      sendDuplicateFeedback(btn.dataset.duplicate === "true", btn.textContent)
    )
  );
  el.completionCard.appendChild(row);
  el.completionCard.classList.remove("hidden");
  // The server refuses messages until this is answered.
  setComposerEnabled(false);
  scrollToBottom();
}

async function sendDuplicateFeedback(isDuplicate, label) {
  if (isBusy || isConversationOver) return;
  isBusy = true;
  const buttons = el.completionCard.querySelectorAll(".feedback-row button");
  buttons.forEach((b) => (b.disabled = true));
  addUserMessage(label);
  const typingEl = addTypingIndicator();

  try {
    const res = await fetch(
      `${API_BASE}/conversations/${encodeURIComponent(conversationId)}/duplicate-feedback`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_duplicate: isDuplicate }),
      }
    );
    removeTypingIndicator(typingEl);
    const body = await safeJson(res);

    if (!res.ok) {
      addErrorMessage(`Request failed (${res.status}): ${body?.detail || res.statusText}`);
      buttons.forEach((b) => (b.disabled = false));
    } else if (body.terminal_outcome) {
      addAssistantMessage(outcomeIntro(body), null);
      showOutcomeCard(body);
      endConversationInput();
    } else {
      addAssistantMessage("Thanks for letting us know.", null);
      el.completionCard.classList.add("hidden");
      setComposerEnabled(true);
    }
  } catch (err) {
    removeTypingIndicator(typingEl);
    addErrorMessage(`Couldn't reach the server: ${err.message}`);
    buttons.forEach((b) => (b.disabled = false));
  }
  isBusy = false;
}

// ---- Final outcomes (Agents 4-6) ----
function outcomeIntro(data) {
  switch (data.terminal_outcome) {
    case "DISPATCHED":
      return "Your ticket has been created and routed to the right team.";
    case "MERGED":
      return "Thanks for confirming — I've linked your report to the existing ticket.";
    case "PENDING_REVIEW":
    case "ESCALATED":
      return "I've passed this to a specialist for review.";
    case "SELF_RESOLVED":
      return "Great — glad the article sorted it out!";
    default:
      return "Your request has been recorded.";
  }
}

function showOutcomeCard(data) {
  const outcome = data.terminal_outcome;
  let html;

  if (outcome === "DISPATCHED" || outcome === "MERGED") {
    html = ticketCardHtml(data);
    el.completionCard.className = "completion-card ready";
  } else if (outcome === "PENDING_REVIEW" || outcome === "ESCALATED") {
    const reason = data.review_reason
      ? REVIEW_REASON_LABELS[data.review_reason] || data.review_reason
      : "it needs a closer look";
    html = `
      <h3>🧑‍💼 A specialist will review this request</h3>
      <div class="row"><span>Why</span><span>${escapeHtml(reason)}</span></div>
      ${data.review_item_id != null
        ? `<div class="row"><span>Review item</span><span>#${escapeHtml(String(data.review_item_id))}</span></div>`
        : ""}
    `;
    el.completionCard.className = "completion-card escalated";
  } else {
    html = `
      <h3>📚 Resolved via Knowledge Base</h3>
      ${data.kb_matched_title
        ? `<div class="row"><span>Article</span><span>${escapeHtml(data.kb_matched_title)}</span></div>`
        : ""}
      <div class="row"><span>Ticket</span><span>none needed</span></div>
    `;
    el.completionCard.className = "completion-card resolved";
  }

  el.completionCard.innerHTML = html;
  el.completionCard.appendChild(auditTrailBlock());
  el.completionCard.classList.remove("hidden");
  scrollToBottom();
}

function ticketCardHtml(data) {
  const t = data.ticket || {};
  const categoryLabel = CATEGORY_LABELS[t.category] || t.category || "—";
  const merged = data.terminal_outcome === "MERGED";
  const incident = data.correlation && data.correlation.outcome === "INCIDENT_CLUSTER";
  const jiraUrl = safeHttpUrl(t.external_url);

  return `
    <h3>${merged ? "🔗 Linked to an existing ticket" : "✅ Ticket created"}</h3>
    ${incident
      ? `<div class="notice">⚠ Part of an ongoing incident — other users have reported the same problem.</div>`
      : ""}
    ${merged && t.duplicate_of
      ? `<div class="notice">Linked to existing ticket <strong>${escapeHtml(t.duplicate_of)}</strong></div>`
      : ""}
    <div class="row"><span>Ticket ID</span><span>${escapeHtml(t.ticket_id || "—")}</span></div>
    <div class="row"><span>Priority</span><span>${escapeHtml(t.priority || "—")}</span></div>
    <div class="row"><span>Team</span><span>${escapeHtml(t.assignee_group || "—")}</span></div>
    <div class="row"><span>Category</span><span>${escapeHtml(categoryLabel)}</span></div>
    <div class="rationale"><strong>Why this team:</strong> ${escapeHtml(t.routing_rationale || "—")}</div>
    ${jiraUrl
      ? `<a class="jira-link" href="${escapeHtml(jiraUrl)}" target="_blank" rel="noopener">Open in Jira (${escapeHtml(t.external_key || "")}) ↗</a>`
      : ""}
  `;
}

// Only http(s) links are rendered - escapeHtml alone would not stop a
// "javascript:" URL in an href.
function safeHttpUrl(url) {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

// ---- Audit trail ----
function auditTrailBlock() {
  const wrap = document.createElement("div");
  wrap.className = "audit-wrap";
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "reasoning-toggle";
  toggle.textContent = "📋 Show audit trail";
  const panel = document.createElement("div");
  panel.className = "audit-panel hidden";

  toggle.addEventListener("click", async () => {
    const opening = panel.classList.contains("hidden");
    panel.classList.toggle("hidden");
    toggle.textContent = opening ? "📋 Hide audit trail" : "📋 Show audit trail";
    if (opening) await loadAuditTrail(panel);
  });

  wrap.appendChild(toggle);
  wrap.appendChild(panel);
  return wrap;
}

async function loadAuditTrail(panel) {
  panel.textContent = "Loading…";
  try {
    const res = await fetch(
      `${API_BASE}/conversations/${encodeURIComponent(conversationId)}/audit`
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const events = await res.json();
    if (!events.length) {
      panel.textContent = "No audit events recorded.";
      return;
    }
    panel.innerHTML = `
      <table class="audit-table">
        <thead><tr><th>Turn</th><th>Stage</th><th>Decision</th><th>Confidence</th><th>ms</th></tr></thead>
        <tbody>${events
          .map(
            (e) => `<tr>
              <td>${escapeHtml(String(e.turn))}</td>
              <td>${escapeHtml(e.stage)}</td>
              <td>${escapeHtml(e.decision)}</td>
              <td>${e.confidence == null ? "—" : escapeHtml(e.confidence.toFixed(2))}</td>
              <td>${escapeHtml(Number(e.latency_ms || 0).toFixed(1))}</td>
            </tr>`
          )
          .join("")}</tbody>
      </table>`;
  } catch (err) {
    panel.textContent = `Couldn't load the audit trail: ${err.message}`;
  }
}

function addKbHint(title) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  const hint = document.createElement("div");
  hint.className = "kb-hint";
  hint.innerHTML = `💡 You might also check: <strong>${escapeHtml(title)}</strong>`;
  row.appendChild(hint);
  el.messages.appendChild(row);
  scrollToBottom();
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
      ? "📚 Suggested fix from the Knowledge Base"
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
  // innerHTML leaves quotes alone; escape them too so values are safe
  // inside attributes (href="...", value="...").
  return div.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

init();