// TRO review queue page — lists GET /review and lets a specialist approve
// (optionally overriding priority / team) or reject each item. All
// decisions are enforced server-side (the schema validators reject an
// unknown team); this page only renders and submits. Every server value
// goes through escapeHtml().

const API_BASE = ""; // same-origin when served from FastAPI (see main.py mount)

// Team options for the override select. Hard-coded copy of the values in
// app/config/routing.py (CATEGORY_TO_TEAM + DEFAULT_TEAM) - keep in sync
// if that table changes. The server is still the authority: a team not in
// routing.py is rejected with 422.
const TEAMS = [
  "Network Operations",
  "Collaboration Tools Support",
  "Messaging Support",
  "Identity & Access Management",
  "Desktop Support",
  "Service Desk L1",
];
const PRIORITIES = ["P1", "P2", "P3", "P4"];

// Display-only labels (same as app.js).
const CATEGORY_LABELS = {
  wifi_internet: "Wifi/Internet Support",
  ms_teams: "Microsoft Teams Support",
  vit_email: "VIT Email Support",
  ad_account_creation: "AD Account Creation",
  printer_support: "Printer Support",
  general: "General",
};
const REVIEW_REASON_LABELS = {
  low_classification_confidence: "Low classification confidence",
  clarification_escalated: "Escalated during clarification",
};

const el = {
  list: document.getElementById("reviewList"),
  statusFilter: document.getElementById("statusFilter"),
  refreshBtn: document.getElementById("refreshBtn"),
  queueCount: document.getElementById("queueCount"),
};

function init() {
  el.statusFilter.addEventListener("change", loadReviews);
  el.refreshBtn.addEventListener("click", loadReviews);
  loadReviews();
}

async function loadReviews() {
  el.list.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const status = el.statusFilter.value;
    const res = await fetch(`${API_BASE}/review?status=${encodeURIComponent(status)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const items = await res.json();
    el.queueCount.textContent = `${items.length} item${items.length === 1 ? "" : "s"}`;
    el.list.innerHTML = "";
    if (!items.length) {
      el.list.innerHTML = `<div class="empty-state">Nothing here. 🎉</div>`;
      return;
    }
    items.forEach((item) => el.list.appendChild(renderItem(item)));
  } catch (err) {
    el.list.innerHTML = `<div class="empty-state">Couldn't load the review queue: ${escapeHtml(err.message)}</div>`;
  }
}

function renderItem(item) {
  const card = document.createElement("div");
  card.className = "review-card";
  card.style.marginBottom = "12px";

  const c = item.classification;
  const category = c ? c.category : item.detected_category;
  const categoryLabel = CATEGORY_LABELS[category] || category || "undetermined";
  const reason = REVIEW_REASON_LABELS[item.reason] || item.reason;

  card.innerHTML = `
    <h3>Review #${escapeHtml(String(item.id))} · ${escapeHtml(reason)}</h3>
    <div class="review-meta">
      ${escapeHtml(item.status)} · conversation ${escapeHtml(item.conversation_id)}
      · ${escapeHtml(new Date(item.created_at).toLocaleString())}
    </div>
    ${item.user_messages.map((m) => `<blockquote>${escapeHtml(m)}</blockquote>`).join("")}
    <div class="row"><span>Category</span><span>${escapeHtml(categoryLabel)}</span></div>
    <div class="row"><span>Completeness</span><span>${escapeHtml(
      String(Math.round((item.completeness_score || 0) * 100))
    )}%</span></div>
    ${c
      ? `<div class="row"><span>Agent 5 priority / team</span><span>${escapeHtml(c.priority)} · ${escapeHtml(c.assignee_group)}</span></div>
         <div class="row"><span>Agent 5 confidence</span><span>${escapeHtml(c.confidence.toFixed(2))} (${escapeHtml(c.source)})</span></div>
         <div class="rationale" style="font-size:13px;margin-top:6px;"><strong>Rationale:</strong> ${escapeHtml(c.routing_rationale)}</div>`
      : `<div class="row"><span>Agent 5</span><span>not classified yet — runs on approval</span></div>`}
  `;

  if (item.status === "PENDING") {
    card.appendChild(renderActions(item));
  }
  return card;
}

function renderActions(item) {
  const wrap = document.createElement("div");
  const actions = document.createElement("div");
  actions.className = "review-actions";

  const prioritySelect = makeSelect("Keep priority", PRIORITIES);
  const teamSelect = makeSelect("Keep team", TEAMS);
  const approveBtn = button("Approve & dispatch", "btn-primary");
  const rejectBtn = button("Reject", "btn-secondary");
  const result = document.createElement("div");
  result.className = "review-result";

  approveBtn.addEventListener("click", async () => {
    const body = {};
    if (prioritySelect.value) body.priority = prioritySelect.value;
    if (teamSelect.value) body.assignee_group = teamSelect.value;
    await submit(`/review/${item.id}/approve`, body, [approveBtn, rejectBtn], result, (ticket) =>
      `Approved — ticket ${escapeHtml(ticket.ticket_id)} (${escapeHtml(ticket.priority)}, ${escapeHtml(ticket.assignee_group)})`
    );
  });
  rejectBtn.addEventListener("click", async () => {
    await submit(`/review/${item.id}/reject`, null, [approveBtn, rejectBtn], result, () => "Rejected.");
  });

  actions.append(prioritySelect, teamSelect, approveBtn, rejectBtn);
  wrap.append(actions, result);
  return wrap;
}

async function submit(path, body, buttons, resultEl, successText) {
  buttons.forEach((b) => (b.disabled = true));
  resultEl.className = "review-result";
  resultEl.textContent = "Working…";
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await safeJson(res);
    if (!res.ok) {
      const detail = typeof data?.detail === "string" ? data.detail : JSON.stringify(data?.detail);
      resultEl.className = "review-result err";
      resultEl.textContent = `Failed (${res.status}): ${detail || res.statusText}`;
      buttons.forEach((b) => (b.disabled = false));
      return;
    }
    resultEl.className = "review-result ok";
    resultEl.innerHTML = successText(data);
  } catch (err) {
    resultEl.className = "review-result err";
    resultEl.textContent = `Couldn't reach the server: ${err.message}`;
    buttons.forEach((b) => (b.disabled = false));
  }
}

function makeSelect(placeholder, values) {
  const select = document.createElement("select");
  select.innerHTML =
    `<option value="">${escapeHtml(placeholder)}</option>` +
    values.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
  return select;
}

function button(label, cls) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = cls;
  b.textContent = label;
  return b;
}

async function safeJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

// Same helper as app.js (no shared module without a build step).
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

init();
