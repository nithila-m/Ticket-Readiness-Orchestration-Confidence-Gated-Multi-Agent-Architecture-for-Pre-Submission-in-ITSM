"""
Manual, read-only smoke test for the Jira Service Management connection -
not part of pytest. Only GET requests; it creates nothing.
Run from the project root: python -m scripts.smoke_test_jira

Needs JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN and JIRA_PROJECT_KEY in
.env (JIRA_ENABLED does not have to be true for this script). Prints
names only - never the token - so you can check that:
  - the credentials work (/myself),
  - JIRA_ISSUE_TYPE exists in the project (createmeta issue types),
  - every value in jira_priority_map is a real priority name,
  - a "Duplicate" issue link type exists (used by link_duplicate).
"""

import sys

import httpx

from app.config.settings import settings


def get(client: httpx.Client, path: str):
    response = client.get(path)
    if response.status_code >= 400:
        print(f"  FAILED {response.status_code} on GET {path}")
        response.raise_for_status()
    return response.json()


def main() -> int:
    missing = [n for n in ("jira_base_url", "jira_email", "jira_api_token") if not getattr(settings, n)]
    if missing:
        print(f"Missing in .env: {', '.join(m.upper() for m in missing)}")
        return 1

    print(f"Jira site: {settings.jira_base_url}  project: {settings.jira_project_key}")
    with httpx.Client(
        base_url=settings.jira_base_url.rstrip("/"),
        auth=(settings.jira_email, settings.jira_api_token),
        headers={"Accept": "application/json"},
        timeout=settings.jira_timeout_seconds,
    ) as client:
        me = get(client, "/rest/api/3/myself")
        print(f"\n[myself] authenticated as: {me.get('displayName')}")

        meta = get(client, f"/rest/api/3/issue/createmeta/{settings.jira_project_key}/issuetypes")
        issue_types = [t.get("name") for t in meta.get("issueTypes", meta.get("values", []))]
        print(f"\n[issue types in {settings.jira_project_key}] {issue_types}")
        ok = settings.jira_issue_type in issue_types
        print(f"  JIRA_ISSUE_TYPE={settings.jira_issue_type!r}: {'OK' if ok else 'NOT FOUND'}")

        priorities = [p.get("name") for p in get(client, "/rest/api/3/priority")]
        print(f"\n[priorities] {priorities}")
        for tro, jira in settings.jira_priority_map.items():
            print(f"  {tro} -> {jira!r}: {'OK' if jira in priorities else 'NOT FOUND'}")

        link_types = [t.get("name") for t in get(client, "/rest/api/3/issueLinkType").get("issueLinkTypes", [])]
        print(f"\n[issue link types] {link_types}")
        print(f"  'Duplicate': {'OK' if 'Duplicate' in link_types else 'NOT FOUND'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
