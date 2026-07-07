"""Trello sync helpers for the review web UI.

Mirrors the card conventions in scripts/trello_sync.py (same name/desc format,
same trello_card_id column) so cards created here are indistinguishable from
ones made by the batch sync. Only implements what the UI needs for single-card,
real-time actions: create-if-missing, comment, archive, reopen. Skill-label
management and the full board sync remain scripts/trello_sync.py's job.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from applypilot.config import APP_DIR

log = logging.getLogger(__name__)

_call_count = 0


def _load_config() -> dict:
    from dotenv import dotenv_values

    values: dict = {}
    for name in (".env", ".trello.env"):
        path = APP_DIR / name
        if path.exists():
            values.update(dotenv_values(path))
    return {
        "api_key": values.get("TRELLO_API_KEY", ""),
        "token": values.get("TRELLO_TOKEN", ""),
        "board_id": values.get("TRELLO_BOARD_ID", ""),
        "list_name": values.get("TRELLO_LIST_NAME", "Job Applications"),
        "review_host": values.get("REVIEW_HOST", "macbeast.local"),
        "review_port": values.get("REVIEW_PORT", "4000"),
    }


def is_configured() -> bool:
    cfg = _load_config()
    return bool(cfg["api_key"] and cfg["token"] and cfg["board_id"])


def build_deactivate_link(job_id: int) -> str:
    """Link back to the review UI's confirm-and-deactivate page for this job."""
    cfg = _load_config()
    return f"http://{cfg['review_host']}:{cfg['review_port']}/deactivate/{job_id}"


def _api(method: str, path: str, payload: dict | None = None) -> dict | None:
    global _call_count
    _call_count += 1
    if _call_count % 80 == 0:
        time.sleep(10)

    cfg = _load_config()
    auth = {"key": cfg["api_key"], "token": cfg["token"]}
    url = f"https://api.trello.com/1/{path.lstrip('/')}"
    if method == "GET":
        resp = requests.get(url, params=auth, timeout=15)
    else:
        resp = requests.request(method, url, params=auth, json=payload, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json() if resp.content else None


def _resolve_attachment_path(raw_path: str | None) -> Path | None:
    """Return the .pdf sibling of a tailored .txt path if it exists, else the raw path."""
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.suffix == ".txt":
        pdf = path.with_suffix(".pdf")
        if pdf.exists():
            return pdf
    return path if path.exists() else None


def _upload_attachment(card_id: str, path: Path) -> None:
    cfg = _load_config()
    url = f"https://api.trello.com/1/cards/{card_id}/attachments"
    with open(path, "rb") as f:
        resp = requests.post(
            url,
            params={"key": cfg["api_key"], "token": cfg["token"]},
            files={"file": (path.name, f, "application/octet-stream")},
            timeout=30,
        )
    resp.raise_for_status()


def _build_card_name(job: dict) -> str:
    company = job.get("company")
    name = f"[{company}] — {job.get('title')}" if company else (job.get("title") or "Untitled")
    return name[:512]


def _build_card_desc(job: dict, job_id: int) -> str:
    link = build_deactivate_link(job_id)
    location = job.get("location") or "—"
    salary = job.get("salary") or "—"
    fit = f"{job['fit_score']}/10" if job.get("fit_score") is not None else "—"
    site = job.get("site") or "—"
    status = job.get("apply_status") or "not applied"
    if status == "applied" and job.get("apply_method"):
        status = f"{status} ({job['apply_method']})"
    if job.get("review_later_at"):
        status = f"{status} | **Review later:** flagged {job['review_later_at'][:10]}"
    discovered = (job.get("discovered_at") or "")[:10]
    url = job.get("url") or ""
    app_url = job.get("application_url") or url
    reasoning = (job.get("score_reasoning") or "").strip()

    docs = " | ".join(filter(None, [
        "Resume: attached" if job.get("tailored_resume_path") else None,
        "Cover letter: attached" if job.get("cover_letter_path") else None,
    ])) or "—"

    return (
        f"[View Job]({url}) | [Apply Now]({app_url})\n\n"
        f"[Deactivate this job in ApplyPilot]({link})\n\n---\n\n"
        f"**Location:** {location}\n"
        f"**Salary:** {salary}\n"
        f"**Fit Score:** {fit} | **Site:** {site} | **Status:** {status}\n"
        f"**Discovered:** {discovered}\n"
        f"**Documents:** {docs}\n\n"
        f"**Score Reasoning:**\n{reasoning}"
    )


def get_or_create_card(job: dict, job_id: int) -> str:
    """Return job['trello_card_id'] if set, else create a card and return its new id.

    Does not persist the new id to the DB -- the caller must save it.
    """
    if job.get("trello_card_id"):
        return job["trello_card_id"]

    cfg = _load_config()
    lists = _api("GET", f"/boards/{cfg['board_id']}/lists") or []
    target = next((l for l in lists if l["name"].lower() == cfg["list_name"].lower()), None)
    if not target:
        raise RuntimeError(f"Trello list '{cfg['list_name']}' not found on configured board.")

    card = _api("POST", "/cards", {
        "idList": target["id"],
        "name": _build_card_name(job),
        "desc": _build_card_desc(job, job_id),
        "pos": "bottom",
    })
    if not card:
        raise RuntimeError("Trello card creation failed.")

    for field in ("tailored_resume_path", "cover_letter_path"):
        path = _resolve_attachment_path(job.get(field))
        if path:
            _upload_attachment(card["id"], path)

    return card["id"]


def update_card_desc(card_id: str, job: dict, job_id: int) -> None:
    """Refresh an existing card's description (e.g. to pick up a new deactivate link)."""
    _api("PUT", f"/cards/{card_id}", {"desc": _build_card_desc(job, job_id)})


def add_comment(card_id: str, text: str) -> None:
    _api("POST", f"/cards/{card_id}/actions/comments", {"text": text})


def set_archived(card_id: str, archived: bool) -> None:
    _api("PUT", f"/cards/{card_id}", {"closed": archived})


def replace_attachment(card_id: str, old_path: str | None, new_path: str) -> None:
    """Remove any existing attachment matching old_path's filename, then attach new_path.

    Used when a job's resume/cover letter is swapped for a different file (e.g. the
    real resume or a generic cover letter) -- keeps the card from accumulating stale
    attachments for the document that's no longer in use.
    """
    resolved_new = _resolve_attachment_path(new_path)
    if not resolved_new:
        return  # new file doesn't exist yet -- nothing to attach

    if old_path:
        old_resolved = _resolve_attachment_path(old_path)
        old_name = old_resolved.name if old_resolved else Path(old_path).with_suffix(".pdf").name
        attachments = _api("GET", f"/cards/{card_id}/attachments") or []
        for att in attachments:
            if att.get("name") == old_name:
                _api("DELETE", f"/cards/{card_id}/attachments/{att['id']}")

    _upload_attachment(card_id, resolved_new)
