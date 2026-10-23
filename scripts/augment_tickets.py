"""
Builds the Agent 4 (correlation) evaluation dataset.

Run from the repo root:
    python scripts/augment_tickets.py

Reads the 150 original tickets in data/eval/<category>.json (never
modified) and writes data/eval/tickets_augmented.json. Fully deterministic
(fixed seed); re-running produces an identical file.

Every ticket gets:
  created_at_offset_minutes  minutes relative to "now" (negative = in the
                             past), so time windows still work whenever the
                             pool is loaded (created_at = now + offset)
  status                     "OPEN"
  label_type                 evaluation label, see below
  gold_cluster_id            globally unique incident id, or null
  duplicate_of               ticket_id of the original, or null
  hard_negative_of           gold_cluster_id this negative imitates, or null

label_type values:
  background              ordinary original ticket
  cluster_member          member of an incident cluster (same building,
                          same category, within a short window)
  duplicate               restatement of an earlier ticket (duplicate_of set)
  cross_building_pattern  original PRN cluster_2: the same printer error in
                          two different buildings. Under the "different
                          buildings can't cluster" rule it is not an
                          incident cluster, so it is kept out of positives.
  hard_negative_building  same category/symptom as a cluster, other building, same time
  hard_negative_time      same building/category/symptom, outside the cluster window
  hard_negative_category  same building/time as a cluster, different category

Relabelling of the original duplicate_cluster_id groups (the original field
is kept untouched; per-category ids like "cluster_1" are not unique, so
gold_cluster_id is "<PREFIX>-cluster_N"):
  ADA cluster_1 is one requester following up about the same account
  (Ganesh Iyer), so ADA-014 and ADA-022 are labelled duplicates of ADA-005,
  and ADA-005 itself is background.
  PRN cluster_2 is cross_building_pattern (see above).
  All other groups are cluster_member.

These labels come from this generator, not from real ITSM history.
"""

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config.buildings import normalize_building  # noqa: E402
from app.config.category_profiles import (  # noqa: E402
    CATEGORY_KEY_TO_DISPLAY,
    category_key_to_display,
    normalize_category_label,
)

SEED = 20261023
DATA_DIR = REPO_ROOT / "data" / "eval"
OUT_PATH = DATA_DIR / "tickets_augmented.json"
SOURCE_FILES = [
    "wifi_internet.json",
    "ms_teams.json",
    "vit_email.json",
    "ad_account_creation.json",
    "printer_support.json",
]

H = 60  # minutes per hour
LOOKBACK_MIN = 72 * H
CLUSTER_WINDOW_MIN = 6 * H
BACKGROUND_RANGE = (-96 * H, -30)  # ~25% of background tickets fall outside the 72h lookback
CONFLICT_MARGIN_MIN = CLUSTER_WINDOW_MIN + H

# Hand-picked anchors (hours ago) so that no two clusters with the same
# building and category overlap in time.
EXISTING_CLUSTER_ANCHORS_H = {
    ("WIFI", "cluster_1"): 20, ("WIFI", "cluster_2"): 40,
    ("TEAMS", "cluster_1"): 26, ("TEAMS", "cluster_2"): 50,
    ("EMAIL", "cluster_1"): 14, ("EMAIL", "cluster_2"): 44,
    ("ADA", "cluster_1"): 36, ("ADA", "cluster_2"): 18,
    ("PRN", "cluster_1"): 8, ("PRN", "cluster_2"): 30,
}
EXISTING_SPREAD_MIN = 60  # members of an existing cluster arrive within an hour

FOLLOWUP_DUPLICATES = {"ADA-014": "ADA-005", "ADA-022": "ADA-005"}
CROSS_BUILDING = {("PRN", "cluster_2")}


def _t(ticket_id, category, building, room, priority, subject, message,
       completeness="complete", deflectable=False, missing=None):
    return {
        "ticket_id": ticket_id,
        "category_selected": category,
        "true_category": category,
        "building": building,
        "room": room,
        "priority": priority,
        "subject": subject,
        "message": message,
        "ocr_text": None,
        "completeness_level": completeness,
        "is_deflectable": deflectable,
        "duplicate_cluster_id": None,
        "missing_fields": missing or [],
    }


WIFI, TEAMS, EMAIL, AD, PRN = (
    "Wifi/Internet Support", "Microsoft Teams Support", "VIT Email Support",
    "AD Account Creation", "Printer Support",
)

# --- New incident clusters: (gold_cluster_id, anchor hours ago, spread minutes, tickets)
NEW_CLUSTERS = [
    ("INC-TT-WIFI", 5, 150, [
        _t("AUG-INC1-01", WIFI, "TT", "TT-512", "HIGH",
           "VIT WiFi down on TT 5th floor",
           "Since around 2:10pm nobody on the 5th floor of TT can get on VIT WiFi. Laptops and "
           "phones both see the network but fail right after authenticating. Our lab session in "
           "TT-512 is completely stuck."),
        _t("AUG-INC1-02", WIFI, "TT", "TT-505", "HIGH",
           "no wifi in TT-505",
           "wifi gone in TT-505 for the whole class, VIT WiFi shows up but it just wont connect"),
        _t("AUG-INC1-03", WIFI, "TT", "TT-518", "HIGH",
           "Internet completely out in TT fifth floor labs",
           "All 30 systems in TT-518 lost network access about 20 minutes ago. Every wireless "
           "device fails on VIT WiFi. Can someone check the access points on this floor?"),
        _t("AUG-INC1-04", WIFI, "TT", "TT-501", "HIGH",
           "Can't connect to VIT WiFi, others in my class too",
           "It was working in TT-501 till lunch, now VIT WiFi keeps saying 'can't connect to this "
           "network'. Four of my friends in the same room have the same issue on their phones."),
        _t("AUG-INC1-05", WIFI, "TT", "TT-509", "HIGH",
           "TT 5th floor wireless outage",
           "Reporting a wireless outage on TT 5th floor. Started about 2pm and affects everyone "
           "between TT-509 and TT-515. Mobile hotspot works, campus wifi doesn't."),
    ]),
    ("INC-SJT-PRN", 10, 120, [
        _t("AUG-INC2-01", PRN, "SJT", "SJT-208", "HIGH",
           "SJT-208 lab printer not printing for anyone",
           "Every PC in SJT-208 sends jobs to the shared Canon printer but nothing comes out. The "
           "queue shows 'Error - Printing' for all of us. Started after 9:30am."),
        _t("AUG-INC2-02", PRN, "SJT", "SJT-206", "HIGH",
           "cant print from sjt lab",
           "cant print anything from the sjt-206 lab, it says printer unavailable, same for the "
           "people sitting next to me"),
        _t("AUG-INC2-03", PRN, "SJT", "SJT-210", "HIGH",
           "Shared printer offline on SJT 2nd floor",
           "The department Canon printer outside SJT-210 shows as offline on every faculty laptop "
           "since this morning. We power cycled it and it is still offline."),
        _t("AUG-INC2-04", PRN, "SJT", "SJT-204", "HIGH",
           "Print jobs failing across SJT second floor",
           "Assignment submissions are due today and none of us on the SJT second floor can print. "
           "Jobs sit in the queue and then fail with a 'printer not responding' message."),
        _t("AUG-INC2-05", PRN, "SJT", "SJT-212", "HIGH",
           "Canon printer on 2nd floor unreachable",
           "Our floor's network printer can't be reached from SJT-212 either. Ping times out and "
           "the printer's web page doesn't load. Looks like the whole floor is affected."),
    ]),
    ("INC-GDN-AD", 22, 180, [
        _t("AUG-INC3-01", AD, "GDN", "GDN-115", "HIGH",
           "Lab PCs in GDN rejecting my AD login",
           "None of the PCs in GDN-115 accept my AD login since 11am. The error is 'The trust "
           "relationship between this workstation and the primary domain failed'. Same on two "
           "other machines I tried."),
        _t("AUG-INC3-02", AD, "GDN", "GDN-118", "HIGH",
           "trust relationship error on GDN lab systems",
           "getting 'trust relationship failed' when logging in with my AD id on the GDN-118 lab "
           "systems, everyone in my batch is stuck at the login screen"),
        _t("AUG-INC3-03", AD, "GDN", "GDN-112", "HIGH",
           "Cannot sign in to lab computer in GDN-112",
           "My AD username and password work on my laptop but the lab computer in GDN-112 refuses "
           "them with a domain trust error. Our practical starts in an hour."),
        _t("AUG-INC3-04", AD, "GDN", "GDN-115", "HIGH",
           "AD login failing for whole batch in GDN",
           "Faculty here. The entire batch of 40 students cannot log in to the GDN-115 lab PCs "
           "with their AD accounts. The workstations seem to have dropped off the domain."),
        _t("AUG-INC3-05", AD, "GDN", "GDN-120", "HIGH",
           "domain login error GDN lab",
           "GDN-120 lab PCs also show the primary domain trust error on login, so it's not just "
           "one room"),
    ]),
    ("INC-MGB-EMAIL", 32, 160, [
        _t("AUG-INC4-01", EMAIL, "MGB", "MGB-310", "HIGH",
           "Outlook keeps asking for password on MGB systems",
           "Since this morning Outlook on the faculty desktops in MGB-310 keeps popping up the "
           "password prompt and rejects the correct VIT email password. Webmail works."),
        _t("AUG-INC4-02", EMAIL, "MGB", "MGB-304", "HIGH",
           "Email password prompt loop in MGB",
           "my outlook in MGB-304 asks for my email password every few minutes and never "
           "accepts it, two colleagues on this floor have the same problem"),
        _t("AUG-INC4-03", EMAIL, "MGB", "MGB-318", "HIGH",
           "Can't open mailbox in Outlook, MGB 3rd floor",
           "Outlook on my office PC in MGB-318 won't connect to my VIT mailbox and keeps asking "
           "for credentials. It started around 9am for several of us on the 3rd floor."),
        _t("AUG-INC4-04", EMAIL, "MGB", "MGB-312", "HIGH",
           "VIT email not syncing in Outlook for MGB faculty",
           "None of the faculty systems in MGB-312 can sync email in Outlook today. Repeated "
           "credential prompts, then 'disconnected'. Phones are fine."),
        _t("AUG-INC4-05", EMAIL, "MGB", "MGB-306", "HIGH",
           "Outlook credential prompt won't go away",
           "Outlook in MGB-306 has been stuck on the credential prompt since morning. I've "
           "re-entered my password many times. Others in the department see the same."),
    ]),
]

# --- Hard negatives: (label_type, target gold_cluster_id, offset from target anchor in minutes, ticket)
HARD_NEGATIVES = [
    # different building, same category and symptom, same time
    ("hard_negative_building", "INC-TT-WIFI", 40, _t(
        "AUG-HNB-01", WIFI, "SMV", "SMV-412", "HIGH",
        "VIT WiFi down on SMV 4th floor",
        "Nobody near SMV-412 can connect to VIT WiFi since about 2pm. Phones and laptops see "
        "the network but fail to connect after login.")),
    ("hard_negative_building", "INC-SJT-PRN", 30, _t(
        "AUG-HNB-02", PRN, "MB", "MB-210", "HIGH",
        "MB-210 lab printer not printing for anyone",
        "Every system in the MB-210 lab sends jobs to the shared printer but nothing prints. "
        "The queue shows 'Error - Printing' for everyone.")),
    ("hard_negative_building", "INC-GDN-AD", 60, _t(
        "AUG-HNB-03", AD, "TT", "TT-317", "HIGH",
        "trust relationship error on TT lab PC",
        "The lab PC in TT-317 rejects my AD login with 'The trust relationship between this "
        "workstation and the primary domain failed'.")),
    ("hard_negative_building", "INC-MGB-EMAIL", 45, _t(
        "AUG-HNB-04", EMAIL, "Admin Block", "AB-205", "HIGH",
        "Outlook keeps asking for my email password",
        "Outlook on my desktop in AB-205 keeps showing the password prompt and won't accept my "
        "VIT email password. Webmail works fine.")),
    ("hard_negative_building", "WIFI-cluster_1", 20, _t(
        "AUG-HNB-05", WIFI, "Men's Hostel B-Block", "B-214", "HIGH",
        "wifi down on our floor since 7pm",
        "wifi dropped for the whole floor around 7pm here in B-214, nobody can connect, not even "
        "the login page loads")),
    # same building, category and symptom, outside the cluster window
    ("hard_negative_time", "INC-TT-WIFI", -30 * H, _t(
        "AUG-HNT-01", WIFI, "TT", "TT-507", "MEDIUM",
        "VIT WiFi not connecting on TT 5th floor",
        "VIT WiFi won't connect on my laptop in TT-507, it sees the network but fails after "
        "authenticating.")),
    ("hard_negative_time", "INC-SJT-PRN", -24 * H, _t(
        "AUG-HNT-02", PRN, "SJT", "SJT-208", "MEDIUM",
        "SJT-208 printer not printing",
        "The shared Canon printer in SJT-208 isn't printing my jobs, the queue shows an "
        "error.")),
    ("hard_negative_time", "INC-GDN-AD", -30 * H, _t(
        "AUG-HNT-03", AD, "GDN", "GDN-118", "MEDIUM",
        "GDN lab PC won't accept my AD login",
        "The PC in GDN-118 shows a domain trust relationship error when I log in with my AD "
        "account.")),
    ("hard_negative_time", "INC-MGB-EMAIL", 24 * H, _t(
        "AUG-HNT-04", EMAIL, "MGB", "MGB-310", "MEDIUM",
        "Outlook password prompt again",
        "Outlook in MGB-310 is asking for my VIT email password repeatedly and not accepting "
        "it.")),
    ("hard_negative_time", "TEAMS-cluster_1", -28 * H, _t(
        "AUG-HNT-05", TEAMS, "SJT", "SJT-301", "MEDIUM",
        "Teams audio cut out in class meeting",
        "Audio in our Teams class meeting cut out partway through, I could see video but hear "
        "nothing.")),
    # same building and time, different category
    ("hard_negative_category", "INC-TT-WIFI", 30, _t(
        "AUG-HNC-01", PRN, "TT", "TT-512", "MEDIUM",
        "Printer in TT-512 jamming",
        "The printer in TT-512 jams on every second page. Removed the stuck paper twice.")),
    ("hard_negative_category", "INC-SJT-PRN", 50, _t(
        "AUG-HNC-02", TEAMS, "SJT", "SJT-208", "MEDIUM",
        "Teams not detecting camera on lab PC",
        "Teams on the SJT-208 lab PC says no camera found, the webcam works in the Camera "
        "app.")),
    ("hard_negative_category", "INC-GDN-AD", 40, _t(
        "AUG-HNC-03", WIFI, "GDN", "GDN-120", "MEDIUM",
        "VIT WiFi slow in GDN-120",
        "VIT WiFi is very slow in GDN-120, pages take ages to load even with full signal.")),
    ("hard_negative_category", "INC-MGB-EMAIL", 60, _t(
        "AUG-HNC-04", AD, "MGB", "MGB-304", "MEDIUM",
        "New faculty AD account not created",
        "A new faculty member joined our department in MGB-304 on Monday and their AD account "
        "still hasn't been created.")),
    ("hard_negative_category", "WIFI-cluster_1", 30, _t(
        "AUG-HNC-05", EMAIL, "Men's Hostel D-Block", "D-221", "LOW",
        "Mailbox full, can't receive mail",
        "My VIT mailbox says storage full and new mails are bouncing. How do I free up "
        "space?")),
]

# --- Near-verbatim duplicates: (original ticket_id, minutes after the original)
DUPLICATE_SOURCES = [
    ("WIFI-002", 45), ("TEAMS-005", 90), ("EMAIL-005", 30), ("ADA-003", 180), ("PRN-003", 60),
]
DUP_SUBJECT_PREFIXES = ["Re: ", "Still not fixed: ", "Again - ", "Follow-up: "]
DUP_SUFFIXES = [
    " Submitting this again since I haven't heard back.",
    " Raising this again, still facing the same issue.",
    " Sorry for the repeat ticket, nothing has changed.",
]


def _prefix(ticket_id: str) -> str:
    return ticket_id.split("-")[0]


def _labels(ticket: dict, label_type: str, cluster=None, duplicate_of=None, near=None) -> dict:
    ticket.update({
        "status": "OPEN",
        "label_type": label_type,
        "gold_cluster_id": cluster,
        "duplicate_of": duplicate_of,
        "hard_negative_of": near,
    })
    return ticket


def build(rng: random.Random) -> list[dict]:
    originals = []
    for name in SOURCE_FILES:
        originals.extend(json.loads((DATA_DIR / name).read_text(encoding="utf-8")))
    by_id = {t["ticket_id"]: t for t in originals}

    anchors: dict[str, int] = {}
    out: list[dict] = []
    placed: set[str] = set()

    # 1. existing groups
    for (prefix, cid), hours in EXISTING_CLUSTER_ANCHORS_H.items():
        gold_id = f"{prefix}-{cid}"
        anchors[gold_id] = -hours * H
        members = [t for t in originals if _prefix(t["ticket_id"]) == prefix and t["duplicate_cluster_id"] == cid]
        for t in members:
            t = dict(t)
            t["created_at_offset_minutes"] = anchors[gold_id] + rng.randint(0, EXISTING_SPREAD_MIN)
            if t["ticket_id"] in FOLLOWUP_DUPLICATES:
                _labels(t, "duplicate", duplicate_of=FOLLOWUP_DUPLICATES[t["ticket_id"]])
            elif t["ticket_id"] in FOLLOWUP_DUPLICATES.values():
                _labels(t, "background")  # the ticket the follow-ups restate
            elif (prefix, cid) in CROSS_BUILDING:
                _labels(t, "cross_building_pattern")
            else:
                _labels(t, "cluster_member", cluster=gold_id)
            out.append(t)
            placed.add(t["ticket_id"])
    # follow-ups arrive after the ticket they restate
    first = {t["ticket_id"]: t for t in out}
    for dup_id, orig_id in FOLLOWUP_DUPLICATES.items():
        lo = first[orig_id]["created_at_offset_minutes"] + 5
        first[dup_id]["created_at_offset_minutes"] = max(first[dup_id]["created_at_offset_minutes"], lo)

    # 2. new clusters
    for gold_id, hours, spread, tickets in NEW_CLUSTERS:
        anchors[gold_id] = -hours * H
        for t in tickets:
            t = dict(t, created_at_offset_minutes=anchors[gold_id] + rng.randint(0, spread))
            out.append(_labels(t, "cluster_member", cluster=gold_id))

    # 3. hard negatives
    for label_type, near, delta, t in HARD_NEGATIVES:
        t = dict(t, created_at_offset_minutes=anchors[near] + delta)
        out.append(_labels(t, label_type, near=near))

    # 4. near-verbatim duplicates of ordinary tickets
    for i, (orig_id, after) in enumerate(DUPLICATE_SOURCES, start=1):
        orig = dict(by_id[orig_id])
        orig["created_at_offset_minutes"] = rng.randint(-60 * H, -6 * H)
        out.append(_labels(orig, "background"))
        placed.add(orig_id)
        dup = dict(by_id[orig_id])
        dup.update({
            "ticket_id": f"AUG-DUP-{i:02d}",
            "subject": rng.choice(DUP_SUBJECT_PREFIXES) + orig["subject"],
            "message": orig["message"] + rng.choice(DUP_SUFFIXES),
            "duplicate_cluster_id": None,
            "created_at_offset_minutes": orig["created_at_offset_minutes"] + after,
            # new tickets carry canonical labels only, even if the source used an alias
            "category_selected": category_key_to_display(normalize_category_label(orig["category_selected"])),
            "true_category": category_key_to_display(normalize_category_label(orig["true_category"])),
        })
        out.append(_labels(dup, "duplicate", duplicate_of=orig_id))

    # 5. background: everything else, kept away from labelled tickets that
    #    share its building and category so it can't silently join or
    #    form a cluster
    labelled = list(out)
    for t in originals:
        if t["ticket_id"] in placed:
            continue
        t = dict(t)
        key = (normalize_building(t["building"]), t["true_category"])
        near = [x["created_at_offset_minutes"] for x in labelled
                if (normalize_building(x["building"]), x["true_category"]) == key]
        for _ in range(1000):
            offset = rng.randint(*BACKGROUND_RANGE)
            if all(abs(offset - n) >= CONFLICT_MARGIN_MIN for n in near):
                break
        else:
            raise RuntimeError(f"could not place {t['ticket_id']} away from labelled tickets")
        t["created_at_offset_minutes"] = offset
        out.append(_labels(t, "background"))

    out.sort(key=lambda t: t["created_at_offset_minutes"])
    return out


def validate(tickets: list[dict]):
    ids = [t["ticket_id"] for t in tickets]
    assert len(ids) == len(set(ids)), "duplicate ticket_id"
    by_id = {t["ticket_id"]: t for t in tickets}
    clusters: dict[str, list[dict]] = {}
    for t in tickets:
        if t["label_type"] == "cluster_member":
            clusters.setdefault(t["gold_cluster_id"], []).append(t)
    for cid, members in clusters.items():
        offsets = [m["created_at_offset_minutes"] for m in members]
        assert max(offsets) - min(offsets) <= CLUSTER_WINDOW_MIN, cid
        assert len({normalize_building(m["building"]) for m in members}) == 1, cid
        assert len({m["true_category"] for m in members}) == 1, cid
        assert min(offsets) >= -LOOKBACK_MIN, cid
    canonical = set(CATEGORY_KEY_TO_DISPLAY.values())
    for t in tickets:
        assert t["created_at_offset_minutes"] < 0, t["ticket_id"]
        if t["ticket_id"].startswith("AUG-"):
            assert {t["true_category"], t["category_selected"]} <= canonical, t["ticket_id"]
        if t["duplicate_of"]:
            orig = by_id[t["duplicate_of"]]
            assert orig["created_at_offset_minutes"] < t["created_at_offset_minutes"], t["ticket_id"]
        if t["label_type"] == "hard_negative_time":
            target = clusters[t["hard_negative_of"]]
            gap = min(abs(t["created_at_offset_minutes"] - m["created_at_offset_minutes"]) for m in target)
            assert gap > CLUSTER_WINDOW_MIN, t["ticket_id"]
            assert t["created_at_offset_minutes"] >= -LOOKBACK_MIN, t["ticket_id"]


def main():
    rng = random.Random(SEED)
    tickets = build(rng)
    validate(tickets)
    OUT_PATH.write_text(json.dumps(tickets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for t in tickets:
        counts[t["label_type"]] = counts.get(t["label_type"], 0) + 1
    print(f"Wrote {len(tickets)} tickets to {OUT_PATH.relative_to(REPO_ROOT)}")
    for label, n in sorted(counts.items()):
        print(f"  {label:<24} {n}")
    sizes: dict[str, int] = {}
    for t in tickets:
        if t["gold_cluster_id"]:
            sizes[t["gold_cluster_id"]] = sizes.get(t["gold_cluster_id"], 0) + 1
    print("  clusters: " + ", ".join(f"{c}={n}" for c, n in sorted(sizes.items())))
    outside = sum(1 for t in tickets if t["created_at_offset_minutes"] < -LOOKBACK_MIN)
    print(f"  older than the 72h lookback: {outside}")


if __name__ == "__main__":
    main()
