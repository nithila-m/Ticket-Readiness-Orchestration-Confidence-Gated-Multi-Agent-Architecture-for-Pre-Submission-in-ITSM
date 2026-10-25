"""
Builds extra calibration tickets for Agent 4's duplicate threshold.

Run from the repo root:
    python scripts/build_agent4_calibration_set.py

Reads data/eval/tickets_augmented.json (never modified - it is also the
live seed pool and the Agent 4 eval input) and writes
data/eval/agent4_calibration_extra.json with the same record schema.
Fully deterministic (fixed seed, all text hand-written below); re-running
produces an identical file.

Two kinds of record, 5 categories each:

  duplicate                 CAL-DUP-01..25, 5 per category. A realistic
                            restatement of one label_type="background"
                            ticket (synonyms, reordered details, shorter or
                            longer, one detail dropped) describing the SAME
                            issue. Same building and room, created 10-240
                            min after the original. duplicate_of = its id.
                            Only background tickets are used: a paraphrase
                            of a cluster_member would be ambiguous between
                            "duplicate" and "incident cluster".
  hard_negative_near_miss   CAL-NEG-01..15, 3 per category. Same category
                            and a similar symptom to an existing background
                            ticket, but a DIFFERENT issue. Per category, 1
                            differs by building (same symptom, another
                            building) and 2 differ by the issue itself
                            (device / model / error code / account) in the
                            same building. Also created 10-240 min after
                            the ticket it imitates, so time never separates
                            the pair - only the content does.
                            hard_negative_of = that ticket's id.

These are synthetic labels written for this project, not real ITSM
history. Each record's category, priority and deflectability are copied
from the ticket it is based on.
"""

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config.buildings import normalize_building  # noqa: E402
from app.config.category_profiles import normalize_category_label  # noqa: E402

SEED = 20261025
DATA_DIR = REPO_ROOT / "data" / "eval"
SOURCE_PATH = DATA_DIR / "tickets_augmented.json"
OUT_PATH = DATA_DIR / "agent4_calibration_extra.json"

MIN_OFFSET_MINUTES = 10
MAX_OFFSET_MINUTES = 240
CATEGORIES = ("wifi_internet", "ms_teams", "vit_email", "ad_account_creation", "printer_support")
DUPLICATES_PER_CATEGORY = 5
NEAR_MISSES_PER_CATEGORY = 3

# (original ticket_id, subject, message)
DUPLICATES: list[tuple[str, str, str]] = [
    # wifi_internet
    ("WIFI-021", "Wifi connected, no internet",
     "Since this morning's Windows update my laptop shows VIT WiFi as connected with full "
     "signal but it says 'no internet access' and no page opens. My phone has internet, it's "
     "only the laptop."),
    ("WIFI-005", "laptop wifi disconnects after sleep",
     "every time i close the lid on my windows 11 laptop and open it again the wifi is gone, "
     "have to hit reconnect on VIT WiFi manually. fine after that. going on for a week, just "
     "my laptop"),
    ("WIFI-023", "Phone keeps losing VIT WiFi",
     "My phone disconnects from VIT WiFi every few minutes, started yesterday evening. Forgetting "
     "the network and joining again helps for a while. Laptop is fine on the same wifi."),
    ("WIFI-011", "5GHz band not working on phone",
     "Since morning my phone gets stuck on 'obtaining IP address' when I pick the VIT WiFi 5GHz "
     "network and then times out. The 2.4GHz one connects fine from the same spot."),
    ("WIFI-014", "No wifi networks visible on laptop",
     "My laptop hasn't shown a single wifi network for three days, not even hotspots. I already "
     "reinstalled the driver, ran the troubleshooter and updated Windows, and tried another "
     "place. Other devices connect, so I think the wifi card is dead."),
    # ms_teams
    ("TEAMS-014", "Teams camera black",
     "In Teams meetings my camera light turns on but everyone just sees black. Same on the "
     "desktop app and the browser version, and there's no error message."),
    ("TEAMS-027", "Denied entry to our recurring channel meeting",
     "Our recurring project-team channel call throws me back to the lobby every time with a "
     "'you don't have permission to join' message. Other meetings I join without problems. "
     "Tried desktop and mobile, same thing."),
    ("TEAMS-021", "no incoming audio in teams calls",
     "others can hear me in teams calls but i hear nothing from them, every call. laptop "
     "speakers work for everything else. using the windows desktop app"),
    ("TEAMS-024", "Teams freezes and closes when someone shares screen",
     "As soon as a screen share starts in a meeting, my Teams desktop app (Windows 10) hangs "
     "for around 10 seconds and then crashes. Third meeting in a row this has happened. No "
     "error shown."),
    ("TEAMS-030", "Not getting Teams message notifications",
     "For the last couple of days Teams hasn't shown any desktop notification for new chat "
     "messages. The messages are there when I open the app, and Windows notifications for "
     "Teams are turned on."),
    # vit_email
    ("EMAIL-003", "Email storage full, new mails not arriving",
     "No new emails have come in for two days. My mailbox storage shows completely full on "
     "webmail. Can you increase my quota or help me free up space?"),
    ("EMAIL-014", "Reset link for email says expired straight away",
     "On the webmail password reset page, the reset link I get says it has already expired, "
     "even though I open it within a minute. Requested two more links, same result."),
    ("EMAIL-024", "Outlook desktop not sending mail",
     "Mails I send from Outlook desktop on Windows 11 stay in the Outbox for hours and never go. "
     "The same mail sent from webmail works. It's every email, not one particular message."),
    ("EMAIL-021", "older emails missing",
     "Lots of my emails from last semester have disappeared from my inbox. They aren't in "
     "archive or trash either, though search finds a few. Recent mail is fine."),
    ("EMAIL-011", "Forwarding rule stopped forwarding",
     "My rule that auto-forwards mail from one sender to my personal address stopped working "
     "this week. The rule is still enabled in settings but nothing gets forwarded. Rest of my "
     "mail is fine."),
    # ad_account_creation
    ("ADA-007", "Account request rejected - name mismatch",
     "The AD account request for our new lab assistant was rejected with 'name mismatch with HR "
     "record'. I think her ID card says Sowmiya and HR has Sowmya. What should we do to fix it?"),
    ("ADA-002", "AD account created but can't log in",
     "HR told me my AD account (employee code EMP2291) was created after I joined on Monday as a "
     "project associate, but around 20 hours later I still can't log in to the lab PCs in "
     "TT-214. Is there a delay or is something wrong?"),
    ("ADA-026", "6 staff accounts failed in bulk upload",
     "The HR bulk-upload batch we sent on Tuesday for six new admin staff (ADM-4410 to "
     "ADM-4415) failed with 'data validation error', with no detail on which record or field. "
     "None of them have accounts and they start Monday. Please check the batch log."),
    ("ADA-020", "VPN rejects new faculty account",
     "Our incoming faculty member's new AD account n.subramani works on the portal and lab PCs "
     "but VPN refuses it with a 'group policy restriction' error, both in Cisco AnyConnect and "
     "the web VPN, from two machines in the server room."),
    ("ADA-017", "'Account already exists' for new lab technician",
     "We are trying to create an AD account for our new lab technician Arul Kumar through the HR "
     "self-service portal, but it says an account already exists for his employee ID. He has "
     "never had a VIT account. We resubmitted twice since yesterday afternoon, same error."),
    # printer_support
    ("PRN-002", "Faculty room LaserJet offline",
     "Our HP LaserJet P1108 (network, not USB) in the faculty room shows offline on every PC "
     "since around 9 this morning. It was fine last evening. Haven't restarted it yet."),
    ("PRN-021", "Ricoh on SJT 9th floor down with SC542",
     "The only printer on SJT 9th floor, the Ricoh MP2014 copier, shows error SC542 and won't "
     "print since about 2pm. Power cycling didn't help and several people up here can't print."),
    ("PRN-016", "Canon printing broken after Windows update",
     "Windows updated itself last night and now printing to my USB Canon printer fails with "
     "'print spooler service is not running'. The service won't start manually either. "
     "Restarted twice, no change."),
    ("PRN-011", "OfficeJet missing from printer list after wifi work",
     "Since the overnight Wi-Fi maintenance our HP OfficeJet in the faculty room doesn't appear "
     "on any laptop's printer list anymore. It worked over Wi-Fi yesterday. We don't know which "
     "network to put it back on."),
    ("PRN-024", "Grinding noise after toner change, nothing prints",
     "Toner light was flashing empty so we replaced the cartridge this morning as per the "
     "manual. Still nothing prints and now there's a grinding sound when we send a job."),
]

# (imitated ticket_id, kind, building or None to keep the original's, subject, message)
# kind: "building" = same symptom, other building; "issue" = same building, different issue.
NEAR_MISSES: list[tuple[str, str, str | None, str, str]] = [
    # wifi_internet
    ("WIFI-003", "building", "Men's Hostel B-Block", "Laptop drops wifi every few minutes",
     "Since this afternoon my laptop disconnects from the wifi every 5-10 minutes and I have to "
     "reconnect by hand. My phone stays connected on the same network."),
    ("WIFI-022", "issue", None, "Phones in our room can't join wifi",
     "None of the phones in our room can connect to VIT WiFi since this afternoon - they keep "
     "saying 'authentication problem'. Our laptops connect without any trouble."),
    ("WIFI-030", "issue", None, "MacBook on wifi but websites won't load after update",
     "After installing the macOS update last night my MacBook joins VIT WiFi automatically, but "
     "websites fail with 'DNS_PROBE_FINISHED_NXDOMAIN'. Pinging an IP address works. Only the "
     "MacBook is affected."),
    # ms_teams
    ("TEAMS-006", "building", "SJT", "Can't share screen in Teams, button disabled",
     "During Teams calls the share screen button is greyed out and does nothing when I click it. "
     "Audio, video and chat in the meeting all work."),
    ("TEAMS-003", "issue", None, "Teams doesn't detect my microphone",
     "Teams says 'no microphone detected' in every meeting, so nobody can hear me. The same "
     "headset mic works in Zoom. Camera is fine. Happens in the desktop app and the browser."),
    ("TEAMS-011", "issue", None, "Teams Android app freezes opening chats",
     "The Teams app on my Android phone freezes whenever I open a chat and I have to force "
     "close it. The desktop app on my laptop works normally."),
    # vit_email
    ("EMAIL-002", "building", "MGB", "Emails with attachments won't send",
     "I can receive mail normally, but any email I send with an attachment keeps loading and "
     "never goes out. It happens every time on webmail."),
    ("EMAIL-022", "issue", None, "VIT email not working in Thunderbird",
     "Trying to set up my VIT email in Thunderbird on my Ubuntu laptop and it fails with "
     "'authentication method not supported' after I sign in. Webmail and my phone's mail app "
     "work fine."),
    ("EMAIL-025", "issue", None, "Calendar invites showing the wrong time",
     "Meeting invites I receive by email show up in my calendar 5.5 hours off - a 10am meeting "
     "appears at 4:30am. The location and agenda are there. The rest of my email is fine."),
    # ad_account_creation
    ("ADA-025", "building", "GDN", "password reset otp not coming",
     "trying to reset my password on the self service page but the otp never arrives on my "
     "registered mobile, tried two times"),
    ("ADA-023", "issue", None, "Account locked out for hours, not unlocking",
     "My account k.murali got locked yesterday evening after the wifi on my new laptop kept "
     "retrying an old password. It still says locked this morning, more than 12 hours later, "
     "so it hasn't auto-unlocked. Please unlock it."),
    ("ADA-024", "issue", None, "PhD scholar account blocked by old student ID",
     "We requested an AD account for our new PhD scholar, but the request was rejected because "
     "her employee ID is already linked to her old student account (reg no 19MIS0112) from "
     "when she studied here. How do we get the old account converted or released?"),
    # printer_support
    ("PRN-010", "building", "Men's Hostel B-Block", "common room printer has no power",
     "the printer in our hostel common room won't switch on at all, no lights. checked the "
     "power switch and the cable, still dead. probably the printer itself"),
    ("PRN-018", "issue", None, "Epson inkjet printing faded horizontal bands",
     "The Epson L3150 ink tank printer in our wing prints horizontal faded bands across every "
     "page, mostly in colour. Ran the head cleaning twice from the printer menu, no improvement."),
    ("PRN-006", "issue", None, "Can't add front office network printer, error 0x0000011b",
     "Trying to add the shared network printer in the front office to my laptop and Windows "
     "fails with error 0x0000011b, so I can't print to it at all. The USB printer at the front "
     "desk works."),
]


def load_source(path: Path = SOURCE_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def category_of(ticket: dict) -> str | None:
    return normalize_category_label(ticket.get("true_category")) or normalize_category_label(
        ticket.get("category_selected")
    )


def _record(original: dict, ticket_id: str, subject: str, message: str, offset: int,
            label_type: str, building: str, room: str | None) -> dict:
    return {
        "ticket_id": ticket_id,
        "category_selected": original["category_selected"],
        "true_category": original["true_category"],
        "building": building,
        "room": room,
        "priority": original["priority"],
        "subject": subject,
        "message": message,
        "ocr_text": None,
        "completeness_level": original["completeness_level"],
        "is_deflectable": original["is_deflectable"],
        "duplicate_cluster_id": None,
        "missing_fields": [],
        "created_at_offset_minutes": offset,
        "status": "OPEN",
        "label_type": label_type,
        "gold_cluster_id": None,
        "duplicate_of": None,
        "hard_negative_of": None,
    }


def build_records(source: list[dict]) -> list[dict]:
    by_id = {t["ticket_id"]: t for t in source}
    rng = random.Random(SEED)
    records = []

    for i, (orig_id, subject, message) in enumerate(DUPLICATES, start=1):
        orig = by_id[orig_id]
        offset = orig["created_at_offset_minutes"] + rng.randint(MIN_OFFSET_MINUTES, MAX_OFFSET_MINUTES)
        rec = _record(orig, f"CAL-DUP-{i:02d}", subject, message, offset,
                      "duplicate", orig["building"], orig["room"])
        rec["duplicate_of"] = orig_id
        records.append(rec)

    for i, (orig_id, kind, building, subject, message) in enumerate(NEAR_MISSES, start=1):
        orig = by_id[orig_id]
        offset = orig["created_at_offset_minutes"] + rng.randint(MIN_OFFSET_MINUTES, MAX_OFFSET_MINUTES)
        if kind == "building":
            # Another building means another room; no room is known for it.
            rec = _record(orig, f"CAL-NEG-{i:02d}", subject, message, offset,
                          "hard_negative_near_miss", building, None)
        else:
            rec = _record(orig, f"CAL-NEG-{i:02d}", subject, message, offset,
                          "hard_negative_near_miss", orig["building"], orig["room"])
        rec["hard_negative_of"] = orig_id
        records.append(rec)

    return records


def validate(records: list[dict], source: list[dict]) -> None:
    """Raises ValueError on the first problem found."""
    by_id = {t["ticket_id"]: t for t in source}
    known_buildings = {normalize_building(t["building"]) for t in source}
    ids = [r["ticket_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ticket_id in calibration set")
    clash = set(ids) & set(by_id)
    if clash:
        raise ValueError(f"calibration ids clash with tickets_augmented.json: {sorted(clash)}")

    near_miss_kinds = {orig_id: kind for orig_id, kind, *_ in NEAR_MISSES}
    dup_counts = dict.fromkeys(CATEGORIES, 0)
    neg_counts = dict.fromkeys(CATEGORIES, 0)
    building_negs = dict.fromkeys(CATEGORIES, 0)

    for r in records:
        tid = r["ticket_id"]
        cat = category_of(r)
        if cat not in CATEGORIES:
            raise ValueError(f"{tid}: invalid category {r['true_category']!r}")
        if normalize_building(r["building"]) not in known_buildings:
            raise ValueError(f"{tid}: unknown building {r['building']!r}")

        if r["label_type"] == "duplicate":
            ref = r["duplicate_of"]
            if not tid.startswith("CAL-DUP-") or r["hard_negative_of"] is not None:
                raise ValueError(f"{tid}: malformed duplicate record")
            dup_counts[cat] += 1
        elif r["label_type"] == "hard_negative_near_miss":
            ref = r["hard_negative_of"]
            if not tid.startswith("CAL-NEG-") or r["duplicate_of"] is not None:
                raise ValueError(f"{tid}: malformed near-miss record")
            neg_counts[cat] += 1
        else:
            raise ValueError(f"{tid}: invalid label_type {r['label_type']!r}")

        orig = by_id.get(ref)
        if orig is None:
            raise ValueError(f"{tid}: references unknown ticket {ref!r}")
        if orig["label_type"] != "background":
            raise ValueError(f"{tid}: {ref} is {orig['label_type']}, expected background")
        if category_of(orig) != cat:
            raise ValueError(f"{tid}: category differs from {ref}")
        gap = r["created_at_offset_minutes"] - orig["created_at_offset_minutes"]
        if not MIN_OFFSET_MINUTES <= gap <= MAX_OFFSET_MINUTES:
            raise ValueError(f"{tid}: created {gap} min after {ref}, expected 10-240")
        if r["created_at_offset_minutes"] > 0:
            raise ValueError(f"{tid}: created in the future")

        same_building = normalize_building(r["building"]) == normalize_building(orig["building"])
        if r["label_type"] == "duplicate" or near_miss_kinds[ref] == "issue":
            if not same_building:
                raise ValueError(f"{tid}: must share {ref}'s building")
        else:
            if same_building:
                raise ValueError(f"{tid}: building variant must differ from {ref}'s building")
            building_negs[cat] += 1

    for cat in CATEGORIES:
        if dup_counts[cat] != DUPLICATES_PER_CATEGORY:
            raise ValueError(f"{cat}: {dup_counts[cat]} duplicates, expected {DUPLICATES_PER_CATEGORY}")
        if neg_counts[cat] != NEAR_MISSES_PER_CATEGORY:
            raise ValueError(f"{cat}: {neg_counts[cat]} near-misses, expected {NEAR_MISSES_PER_CATEGORY}")
        if building_negs[cat] != 1:
            raise ValueError(f"{cat}: {building_negs[cat]} building near-misses, expected 1")


def render(records: list[dict]) -> str:
    return json.dumps(records, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    source = load_source()
    records = build_records(source)
    validate(records, source)
    OUT_PATH.write_text(render(records), encoding="utf-8")
    n_dup = sum(r["label_type"] == "duplicate" for r in records)
    print(f"Wrote {len(records)} records ({n_dup} duplicate, {len(records) - n_dup} "
          f"hard_negative_near_miss) to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
