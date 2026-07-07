"""Local web UI for browsing tailored resumes and cover letters.

Read-mostly review tool: lists scored jobs in a vim-navigable table and
renders the tailored resume + cover letter PDFs side by side for the
highlighted job. The only write path is marking a job active/inactive,
which also gates the tailor/cover/apply pipeline queries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request

from applypilot import trello
from applypilot.config import (
    APP_DIR,
    COVER_LETTER_DIR,
    GENERIC_COVER_LETTER_PATH,
    REAL_RESUME_PDF_URL,
    RESUME_PATH,
    RESUME_PDF_PATH,
    ensure_dirs,
    load_env,
    resolve_company_name,
)
from applypilot.database import get_connection, init_db
from applypilot.scoring.pdf import convert_cover_letter_to_pdf, convert_to_pdf

log = logging.getLogger(__name__)

_CARD_FIELDS = (
    "company", "title", "location", "salary", "fit_score", "site",
    "apply_status", "apply_method", "discovered_at", "url", "application_url",
    "score_reasoning", "tailored_resume_path", "cover_letter_path", "trello_card_id",
    "review_later_at",
)

# Every TEXT column on jobs -- searched server-side so the client never has to
# hold full_description/description (which can be several KB per row) in memory.
_SEARCHABLE_TEXT_COLUMNS = (
    "url", "company", "title", "salary", "description", "location", "site",
    "strategy", "discovered_at", "full_description", "application_url",
    "detail_scraped_at", "detail_error", "score_reasoning", "scored_at",
    "tailored_resume_path", "tailored_at", "cover_letter_path", "cover_letter_at",
    "applied_at", "apply_status", "apply_error", "agent_id", "last_attempted_at",
    "apply_task_id", "verification_confidence", "trello_card_id",
)


def _row_to_dict(row) -> dict:
    return {
        "id": row["rowid"],
        "url": row["url"],
        "title": row["title"],
        "company": row["company"],
        "site": row["site"],
        "location": row["location"],
        "salary": row["salary"],
        "fit_score": row["fit_score"],
        "score_reasoning": row["score_reasoning"],
        "discovered_at": row["discovered_at"],
        "application_url": row["application_url"],
        "has_direct_url": bool(row["application_url"]),
        "has_resume": bool(row["tailored_resume_path"]),
        "has_cover": bool(row["cover_letter_path"]),
        "applied_at": row["applied_at"],
        "apply_status": row["apply_status"],
        "apply_error": row["apply_error"],
        "detail_error": row["detail_error"],
        "strategy": row["strategy"],
        "active": bool(row["active"] if row["active"] is not None else True),
        "review_later_at": row["review_later_at"],
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/jobs")
    def list_jobs():
        conn = get_connection()
        rows = conn.execute(
            "SELECT rowid, url, title, company, site, location, salary, fit_score, "
            "score_reasoning, discovered_at, application_url, strategy, "
            "apply_error, detail_error, "
            "tailored_resume_path, cover_letter_path, applied_at, apply_status, active, "
            "review_later_at "
            "FROM jobs WHERE fit_score IS NOT NULL "
            "ORDER BY fit_score DESC, title"
        ).fetchall()
        return jsonify([_row_to_dict(r) for r in rows])

    @app.get("/api/jobs/<int:job_id>/detail")
    def job_detail(job_id: int):
        conn = get_connection()
        row = conn.execute(
            "SELECT description, full_description FROM jobs WHERE rowid = ?", (job_id,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(dict(row))

    @app.get("/api/search")
    def search_jobs():
        q = (request.args.get("q") or "").strip()
        if not q:
            return jsonify([])

        conn = get_connection()
        where = " OR ".join(f"{col} LIKE ?" for col in _SEARCHABLE_TEXT_COLUMNS)
        like = f"%{q}%"
        rows = conn.execute(
            f"SELECT rowid FROM jobs WHERE fit_score IS NOT NULL AND ({where})",
            [like] * len(_SEARCHABLE_TEXT_COLUMNS),
        ).fetchall()
        return jsonify([r[0] for r in rows])

    @app.post("/api/jobs/<int:job_id>/active")
    def set_active(job_id: int):
        payload = request.get_json(silent=True) or {}
        active = bool(payload.get("active"))
        result = _apply_active_change(job_id, active)
        return jsonify(result)

    @app.get("/deactivate/<int:job_id>")
    def deactivate_confirm(job_id: int):
        conn = get_connection()
        row = conn.execute(
            "SELECT title, company, active FROM jobs WHERE rowid = ?", (job_id,)
        ).fetchone()
        if row is None:
            return Response("Job not found.", mimetype="text/plain", status=404)
        return Response(_render_confirm_page(job_id, dict(row)), mimetype="text/html")

    @app.post("/deactivate/<int:job_id>")
    def deactivate_confirm_submit(job_id: int):
        result = _apply_active_change(job_id, active=False)
        return Response(_render_done_page(result), mimetype="text/html")

    @app.post("/api/jobs/<int:job_id>/resume/real")
    def use_real_resume(job_id: int):
        if not RESUME_PATH.exists():
            return jsonify({"id": job_id, "error": f"Real resume not found at {RESUME_PATH}"}), 404

        # Fetch the hand-maintained PDF directly instead of converting
        # RESUME_PATH's .txt through the AI-tailored-resume parser/renderer --
        # that pipeline assumes the AI tailoring format and mangles a
        # differently-structured master resume. tailored_resume_path still
        # points at the .txt (for its plain-text content used elsewhere), but
        # its .pdf sibling is now this downloaded file instead of a rendered one.
        try:
            resp = requests.get(REAL_RESUME_PDF_URL, timeout=15)
            resp.raise_for_status()
            RESUME_PDF_PATH.write_bytes(resp.content)
        except Exception as e:
            log.error("Failed to fetch real resume PDF from %s: %s", REAL_RESUME_PDF_URL, e)
            return jsonify({"id": job_id, "error": f"Failed to fetch real resume PDF: {e}"}), 502

        result = _apply_document_override(job_id, "tailored_resume_path", RESUME_PATH)
        return jsonify(result)

    @app.post("/api/jobs/<int:job_id>/cover/generic")
    def use_generic_cover_letter(job_id: int):
        result = _apply_generic_cover_letter(job_id)
        return jsonify(result)

    @app.post("/api/jobs/<int:job_id>/apply")
    def launch_apply(job_id: int):
        payload = request.get_json(silent=True) or {}
        dry_run = bool(payload.get("dry_run", True))
        conn = get_connection()
        row = conn.execute("SELECT url FROM jobs WHERE rowid = ?", (job_id,)).fetchone()
        if row is None or not row["url"]:
            return jsonify({"launched": False, "error": "Job not found"}), 404
        try:
            _launch_apply_in_terminal(job_id, row["url"], dry_run=dry_run)
            return jsonify({"launched": True, "dry_run": dry_run})
        except Exception as e:
            log.error("Failed to launch apply for job %d: %s", job_id, e)
            return jsonify({"launched": False, "error": str(e)}), 500

    @app.post("/api/jobs/<int:job_id>/apply-done")
    def apply_done(job_id: int):
        """Callback hit by the spawned Terminal script once `./ap apply` exits.

        Fires regardless of whether the browser tab that launched it is still
        open. Only syncs Trello for real (non-dry-run) applies that actually
        succeeded -- a dry run's RESULT:APPLIED doesn't mean anything was
        submitted.
        """
        payload = request.get_json(silent=True) or {}
        dry_run = bool(payload.get("dry_run", True))
        conn = get_connection()
        row = conn.execute(
            f"SELECT {', '.join(_CARD_FIELDS)} FROM jobs WHERE rowid = ?", (job_id,)
        ).fetchone()
        if row is None:
            return jsonify({"ok": False, "error": "Job not found"}), 404
        job = dict(row)

        result = {"ok": True}
        if not dry_run and job.get("apply_status") == "applied":
            trello_error = _sync_trello_card_applied(conn, job_id, job)
            if trello_error:
                result["trello_error"] = trello_error
        return jsonify(result)

    @app.post("/api/jobs/<int:job_id>/mark-applied-manually")
    def mark_applied_manually(job_id: int):
        """Mark a job applied when it was submitted outside ApplyPilot entirely
        (e.g. applied directly on the site) -- still records it and syncs Trello,
        just tagged apply_method='manual' instead of 'automated'.
        """
        conn = get_connection()
        row = conn.execute(
            f"SELECT {', '.join(_CARD_FIELDS)} FROM jobs WHERE rowid = ?", (job_id,)
        ).fetchone()
        if row is None:
            return jsonify({"id": job_id, "error": "Job not found"}), 404
        job = dict(row)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE jobs SET apply_status = 'applied', applied_at = ?, "
            "apply_error = NULL, apply_method = 'manual' WHERE rowid = ?",
            (now, job_id),
        )
        conn.commit()

        job["apply_status"] = "applied"
        job["apply_method"] = "manual"
        result = {"id": job_id, "applied_at": now}
        trello_error = _sync_trello_card_manual_apply(conn, job_id, job)
        if trello_error:
            result["trello_error"] = trello_error
        return jsonify(result)

    @app.post("/api/jobs/<int:job_id>/review-later")
    def toggle_review_later(job_id: int):
        """Toggle the review-later flag. NULL = not flagged; a timestamp means
        flagged (and doubles as queue order -- sort the Review column ascending
        to go through flagged jobs in the order they were flagged).
        """
        conn = get_connection()
        row = conn.execute(
            f"SELECT {', '.join(_CARD_FIELDS)} FROM jobs WHERE rowid = ?", (job_id,)
        ).fetchone()
        if row is None:
            return jsonify({"id": job_id, "error": "Job not found"}), 404
        job = dict(row)

        flagging = not job.get("review_later_at")
        now = datetime.now(timezone.utc).isoformat() if flagging else None
        conn.execute(
            "UPDATE jobs SET review_later_at = ? WHERE rowid = ?", (now, job_id)
        )
        conn.commit()

        job["review_later_at"] = now
        result = {"id": job_id, "review_later_at": now}
        trello_error = _sync_trello_card_review_later(conn, job_id, job, flagging)
        if trello_error:
            result["trello_error"] = trello_error
        return jsonify(result)

    @app.get("/api/jobs/<int:job_id>/status")
    def job_status(job_id: int):
        conn = get_connection()
        row = conn.execute(
            "SELECT apply_status, applied_at, apply_error FROM jobs WHERE rowid = ?", (job_id,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(dict(row))

    @app.get("/pdf/resume/<int:job_id>")
    def resume_pdf(job_id: int):
        return _serve_pdf(job_id, "tailored_resume_path")

    @app.get("/pdf/cover/<int:job_id>")
    def cover_pdf(job_id: int):
        return _serve_pdf(job_id, "cover_letter_path")

    def _serve_pdf(job_id: int, column: str) -> Response:
        conn = get_connection()
        row = conn.execute(
            f"SELECT {column} FROM jobs WHERE rowid = ?", (job_id,)
        ).fetchone()
        if row is None or not row[column]:
            return Response("Not generated yet.", mimetype="text/plain", status=404)

        txt_path = Path(row[column])
        pdf_path = txt_path.with_suffix(".pdf")

        # The real resume's PDF comes from REAL_RESUME_PDF_URL (see
        # use_real_resume), never from parsing RESUME_PATH -- skip the
        # txt-vs-pdf staleness regeneration, which would otherwise silently
        # overwrite the downloaded PDF with a broken parser conversion the
        # next time RESUME_PATH's .txt is edited for any other reason.
        if column == "tailored_resume_path" and txt_path == RESUME_PATH:
            if not pdf_path.exists():
                return Response("Not generated yet -- press Shift+R again.", mimetype="text/plain", status=404)
            return Response(pdf_path.read_bytes(), mimetype="application/pdf")

        stale = pdf_path.exists() and txt_path.exists() and txt_path.stat().st_mtime > pdf_path.stat().st_mtime
        if not pdf_path.exists() or stale:
            if not txt_path.exists():
                return Response("Source file missing.", mimetype="text/plain", status=404)
            convert = convert_cover_letter_to_pdf if column == "cover_letter_path" else convert_to_pdf
            try:
                convert(txt_path)
            except Exception as e:
                log.error("On-demand PDF conversion failed for %s: %s", txt_path, e)
                return Response(f"PDF conversion failed: {e}", mimetype="text/plain", status=500)

        return Response(pdf_path.read_bytes(), mimetype="application/pdf")

    return app


def _launch_apply_in_terminal(job_id: int, job_url: str, dry_run: bool = True) -> None:
    """Open a new macOS Terminal window running `./ap apply --url <job_url>`.

    Writes a small shell script and points Terminal.app at its path (via
    osascript) rather than embedding the URL directly in an AppleScript string,
    to avoid a second layer of shell/AppleScript escaping.

    When `dry_run` is False, the script also calls back to this same server's
    `/apply-done` endpoint once `./ap apply` exits, so the caller (which may no
    longer have this request open, e.g. a browser tab) can react to real
    completions -- syncing Trello, advancing the UI's selection, etc.
    """
    if platform.system() != "Darwin":
        raise RuntimeError("Launching a new Terminal window is only supported on macOS.")

    project_root = APP_DIR.parent
    ap_script = project_root / "ap"
    if not ap_script.exists():
        raise RuntimeError(f"Could not find ./ap at {ap_script}")

    log_dir = APP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_path = log_dir / f"apply_launch_{job_id}_{int(time.time())}.sh"

    lines = [
        "#!/bin/bash",
        f"cd {shlex.quote(str(project_root))}",
        f"./ap apply --url {shlex.quote(job_url)}" + ("" if not dry_run else " --dry-run"),
    ]
    if not dry_run:
        callback_url = request.host_url.rstrip("/") + f"/api/jobs/{job_id}/apply-done"
        callback_payload = shlex.quote(json.dumps({"dry_run": False}))
        lines.append(
            f"curl -s -X POST {shlex.quote(callback_url)} "
            f"-H 'Content-Type: application/json' -d {callback_payload} >/dev/null 2>&1"
        )
    lines += ["echo", "echo 'Press Enter to close this window...'", "read"]
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o700)

    # Appending "; exit" makes Terminal close the window automatically once
    # the "Press Enter..." pause is dismissed, instead of dropping back to a
    # bare shell prompt that needs a manual exit/Ctrl-D.
    osa_script = (
        f'tell application "Terminal"\n'
        f'  do script "{script_path}; exit"\n'
        f'  activate\n'
        f'end tell'
    )
    subprocess.Popen(["osascript", "-e", osa_script])


def _apply_active_change(job_id: int, active: bool) -> dict:
    """Shared by the JSON toggle API and the Trello confirm-page POST."""
    conn = get_connection()
    conn.execute("UPDATE jobs SET active = ? WHERE rowid = ?", (1 if active else 0, job_id))
    conn.commit()

    trello_error = _sync_trello_card(conn, job_id, active=active)

    result = {"id": job_id, "active": active}
    if trello_error:
        result["trello_error"] = trello_error
    return result


def _apply_document_override(job_id: int, column: str, new_path) -> dict:
    """Point a job's resume/cover-letter column at a fixed file (real resume)
    instead of the AI-tailored one. Direct overwrite -- the original AI-generated
    file stays on disk, just no longer referenced by this job.
    """
    conn = get_connection()
    row = conn.execute(f"SELECT {column}, trello_card_id FROM jobs WHERE rowid = ?", (job_id,)).fetchone()
    if row is None:
        return {"id": job_id, "error": "Job not found"}

    old_path = row[column]
    return _set_document_path(conn, job_id, column, old_path, str(new_path), row["trello_card_id"])


def _apply_generic_cover_letter(job_id: int) -> dict:
    """Render the shared generic cover letter template for this specific job
    (substituting [Company Name]/[AT_COMPANY]/[Job Title]) into a per-job file
    in COVER_LETTER_DIR, then point cover_letter_path at that rendered copy.

    The shared template can live anywhere (e.g. a symlink into another repo);
    the rendered per-job .txt/.pdf always land in this repo's cover_letters dir.

    `[AT_COMPANY]` covers the " at {company}" clause and `[Company Name]`
    covers bare-noun uses (e.g. "help {company} achieve..."). When the DB's
    `company` value is missing or turns out to be a job board name leaked in
    from a thin listing (see resolve_company_name), the " at ..." clause is
    dropped entirely instead of rendering "at linkedin" / "at None", the
    standalone company line in the header is removed, and bare-noun uses fall
    back to "your team".
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT company, title, url, cover_letter_path, trello_card_id FROM jobs WHERE rowid = ?", (job_id,)
    ).fetchone()
    if row is None:
        return {"id": job_id, "error": "Job not found"}

    if not GENERIC_COVER_LETTER_PATH.exists():
        return {"id": job_id, "error": f"Generic cover letter not found at {GENERIC_COVER_LETTER_PATH}"}

    template = GENERIC_COVER_LETTER_PATH.read_text(encoding="utf-8")
    company = resolve_company_name(row["company"])

    if company:
        rendered = template.replace("[AT_COMPANY]", f" at {company}").replace("[Company Name]", company)
    else:
        rendered = re.sub(r"\n\[Company Name\]\n", "\n", template)
        rendered = rendered.replace("[AT_COMPANY]", "").replace("[Company Name]", "your team")

    rendered = rendered.replace("[Job Title]", row["title"] or "this role")

    url_hash = hashlib.md5(row["url"].encode()).hexdigest()[:8]
    out_path = COVER_LETTER_DIR / f"generic_{url_hash}_CL.txt"
    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")

    # Generate the PDF eagerly (not lazily on first view) so a Trello attachment
    # swap right after this attaches a real PDF instead of falling back to the .txt.
    try:
        convert_cover_letter_to_pdf(out_path)
    except Exception as e:
        log.error("PDF conversion failed for generic cover letter (job %d): %s", job_id, e)

    old_path = row["cover_letter_path"]
    return _set_document_path(conn, job_id, "cover_letter_path", old_path, str(out_path), row["trello_card_id"])


def _set_document_path(conn, job_id: int, column: str, old_path: str | None, new_path: str, card_id: str | None) -> dict:
    """Shared by both override paths: update the DB column, swap the Trello attachment if a card exists."""
    conn.execute(f"UPDATE jobs SET {column} = ? WHERE rowid = ?", (new_path, job_id))
    conn.commit()

    result = {"id": job_id, "path": new_path}
    if card_id and trello.is_configured():
        try:
            trello.replace_attachment(card_id, old_path, new_path)
        except Exception as e:
            log.error("Trello attachment swap failed for job %d: %s", job_id, e)
            result["trello_error"] = str(e)
    return result


def _render_confirm_page(job_id: int, job: dict) -> str:
    title = job.get("title") or "Untitled"
    company = job.get("company") or ""
    heading = f"{title} — {company}" if company else title
    if not job.get("active", True):
        body = f"<p>“{_escape(heading)}” is already marked inactive.</p>"
    else:
        body = f"""
        <p>Deactivate this job?</p>
        <p class="job-title">{_escape(heading)}</p>
        <form method="post" action="/deactivate/{job_id}">
          <button type="submit">Confirm deactivate</button>
        </form>
        """
    return _CONFIRM_PAGE_TEMPLATE.format(body=body)


def _render_done_page(result: dict) -> str:
    if result.get("trello_error"):
        body = f"<p>Job marked inactive locally, but the Trello sync failed:</p><p class=\"err\">{_escape(result['trello_error'])}</p>"
    else:
        body = "<p>Done. This job has been marked inactive.</p>"
    return _CONFIRM_PAGE_TEMPLATE.format(body=body)


def _escape(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )


_CONFIRM_PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ApplyPilot</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 480px; margin: 80px auto; text-align: center; }}
  .job-title {{ font-weight: 600; margin: 12px 0 24px; }}
  button {{ font-size: 15px; padding: 8px 18px; cursor: pointer; }}
  .err {{ color: #c00; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _sync_trello_card(conn, job_id: int, active: bool) -> str | None:
    """Create/update the job's Trello card to reflect an active/inactive toggle.

    Marking inactive: ensure a card exists, comment with the timestamp, archive it.
    Marking active again: ensure a card exists, reopen it, comment with the timestamp.

    Returns an error message on failure, or None on success / when Trello isn't configured.
    """
    if not trello.is_configured():
        return None

    row = conn.execute(
        f"SELECT {', '.join(_CARD_FIELDS)} FROM jobs WHERE rowid = ?", (job_id,)
    ).fetchone()
    if row is None:
        return None
    job = dict(row)

    try:
        card_id = trello.get_or_create_card(job, job_id)
        if card_id != job.get("trello_card_id"):
            conn.execute("UPDATE jobs SET trello_card_id = ? WHERE rowid = ?", (card_id, job_id))
            conn.commit()
        else:
            # Card already existed -- refresh its description so it picks up
            # the deactivate link (or any other desc changes) on every touch.
            trello.update_card_desc(card_id, job, job_id)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if active:
            trello.add_comment(card_id, f"Reactivated via ApplyPilot review UI at {now}.")
            trello.set_archived(card_id, False)
        else:
            trello.add_comment(card_id, f"Marked inactive via ApplyPilot review UI at {now}.")
            trello.set_archived(card_id, True)
        return None
    except Exception as e:
        log.error("Trello sync failed for job %d: %s", job_id, e)
        return str(e)


def _sync_trello_card_applied(conn, job_id: int, job: dict) -> str | None:
    """Comment on the job's Trello card to record a real (non-dry-run) apply.

    Returns an error message on failure, or None on success / when Trello
    isn't configured.
    """
    if not trello.is_configured():
        return None

    try:
        card_id = trello.get_or_create_card(job, job_id)
        if card_id != job.get("trello_card_id"):
            conn.execute("UPDATE jobs SET trello_card_id = ? WHERE rowid = ?", (card_id, job_id))
            conn.commit()
        else:
            trello.update_card_desc(card_id, job, job_id)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        trello.add_comment(card_id, f"Applied via ApplyPilot automated apply at {now}.")
        return None
    except Exception as e:
        log.error("Trello apply sync failed for job %d: %s", job_id, e)
        return str(e)


def _sync_trello_card_manual_apply(conn, job_id: int, job: dict) -> str | None:
    """Comment on the job's Trello card to record a manually-marked apply
    (submitted outside ApplyPilot entirely, e.g. directly on the site).

    Returns an error message on failure, or None on success / when Trello
    isn't configured.
    """
    if not trello.is_configured():
        return None

    try:
        card_id = trello.get_or_create_card(job, job_id)
        if card_id != job.get("trello_card_id"):
            conn.execute("UPDATE jobs SET trello_card_id = ? WHERE rowid = ?", (card_id, job_id))
            conn.commit()
        else:
            trello.update_card_desc(card_id, job, job_id)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        trello.add_comment(card_id, f"Marked as applied manually (outside ApplyPilot) at {now}.")
        return None
    except Exception as e:
        log.error("Trello manual-apply sync failed for job %d: %s", job_id, e)
        return str(e)


def _sync_trello_card_review_later(conn, job_id: int, job: dict, flagging: bool) -> str | None:
    """Comment on and refresh the job's Trello card when the review-later flag
    is toggled either way -- unlike the active/inactive toggle, this always
    creates the card (rather than only touching existing ones) since the user
    wants review-later jobs represented in Trello regardless of prior state.

    Returns an error message on failure, or None on success / when Trello
    isn't configured.
    """
    if not trello.is_configured():
        return None

    try:
        card_id = trello.get_or_create_card(job, job_id)
        if card_id != job.get("trello_card_id"):
            conn.execute("UPDATE jobs SET trello_card_id = ? WHERE rowid = ?", (card_id, job_id))
            conn.commit()
        else:
            trello.update_card_desc(card_id, job, job_id)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if flagging:
            trello.add_comment(card_id, f"Flagged for review later at {now}.")
        else:
            trello.add_comment(card_id, f"Review-later flag cleared at {now}.")
        return None
    except Exception as e:
        log.error("Trello review-later sync failed for job %d: %s", job_id, e)
        return str(e)


def run_server(port: int = 4000) -> None:
    """Start the review web UI, blocking the current thread until Ctrl+C.

    Binds to 0.0.0.0 with debug=True (hot-reload + interactive debugger) --
    reachable from any device on the local network, by request.
    """
    load_env()
    ensure_dirs()
    init_db()

    bind_host = "0.0.0.0"

    app = create_app()
    print(f"ApplyPilot review UI: http://{bind_host}:{port}  (Ctrl+C to stop)")
    app.run(host=bind_host, port=port, debug=True, threaded=True)


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ApplyPilot Review</title>
<style>
  html, body { height: 100%; margin: 0; font-family: -apple-system, Helvetica, Arial, sans-serif; }
  body { display: flex; flex-direction: column; }

  #toolbar { display: flex; align-items: center; gap: 16px; padding: 6px 10px; border-bottom: 1px solid #ccc; background: #fafafa; font-size: 13px; flex-wrap: wrap; }
  #scores label { margin-right: 8px; user-select: none; cursor: pointer; }
  #search { padding: 2px 6px; font-size: 13px; }
  #count { color: #666; margin-left: auto; }

  #table-wrap { overflow-y: auto; overflow-x: auto; border-bottom: 2px solid #333; }
  table { width: 100%; min-width: 100%; border-collapse: collapse; font-size: 13px; table-layout: fixed; }
  th, td { text-align: left; padding: 3px 8px; border-bottom: 1px solid #eee; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  th { position: sticky; top: 0; background: #333; color: #fff; cursor: pointer; user-select: none; }
  th .arrow { opacity: 0.6; font-size: 10px; }
  .resize-handle { position: absolute; top: 0; right: 0; width: 6px; height: 100%; cursor: col-resize; }
  .resize-handle:hover, .resize-handle.resizing { background: rgba(255, 255, 255, 0.35); }
  tr.selected { background: #cfe8ff !important; }
  tr.inactive { color: #999; text-decoration: line-through; }
  tr.status-applied { background: #e6f7e6; }
  tr.status-failed { background: #fbe6e6; }
  tr:hover { background: #f0f6ff; cursor: pointer; }

  #panes { flex: 1; display: flex; min-height: 0; }

  .detail-pane { flex: 0 0 50%; max-width: 50%; display: flex; flex-direction: column; border-right: 1px solid #ccc; min-height: 0; }
  .detail-pane h3 { margin: 0; padding: 4px 8px; background: #eee; font-size: 13px; }
  .detail-body { flex: 1; overflow-y: auto; padding: 10px 14px; font-size: 13px; line-height: 1.5; }
  .detail-body table.kv { border-collapse: collapse; margin-bottom: 12px; }
  .detail-body table.kv th { text-align: left; padding: 2px 10px 2px 0; color: #555; font-weight: 600; white-space: nowrap; vertical-align: top; }
  .detail-body table.kv td { padding: 2px 0; }
  .detail-body h4 { margin: 10px 0 4px; font-size: 13px; }
  .detail-body .reasoning { white-space: pre-wrap; }
  .detail-body td.url-cell { font-family: monospace; font-size: 12px; white-space: normal; word-break: break-all; }
  .detail-body .empty-msg { color: #999; }

  .pdf-stack { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .pane { flex: 1; display: flex; flex-direction: column; border-bottom: 1px solid #ccc; min-height: 0; }
  .pane:last-child { border-bottom: none; }
  .pane h3 { margin: 0; padding: 4px 8px; background: #eee; font-size: 13px; }
  .pane iframe { flex: 1; border: none; width: 100%; }
  .pane .empty { flex: 1; display: flex; align-items: center; justify-content: center; color: #999; }

  #help { padding: 4px 10px; font-size: 12px; color: #666; border-bottom: 1px solid #ccc; }
  a.open-link { text-decoration: none; }
</style>
</head>
<body>
  <div id="help">j/k or &uarr;/&darr;: move &nbsp; x: toggle active/inactive &nbsp; l: toggle review later &nbsp; a: launch apply (dry-run) &nbsp; Shift+A: launch real apply (no dry-run) &nbsp; Shift+M: mark applied manually &nbsp; Shift+R: use real resume &nbsp; Shift+G: use generic cover letter &nbsp; click row to select &nbsp; click a column header to sort</div>
  <div id="toolbar">
    <div id="scores"></div>
    <input id="search" type="text" placeholder="Search all job fields...">
    <label><input type="checkbox" id="hide-inactive"> Hide inactive</label>
    <label><input type="checkbox" id="hide-applied"> Hide applied</label>
    <label><input type="checkbox" id="hide-failed"> Hide failed</label>
    <label><input type="checkbox" id="hide-review-later"> Hide review later</label>
    <div id="trello-status"></div>
    <div id="count"></div>
  </div>
  <div id="table-wrap">
    <table>
      <thead><tr id="header-row"></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <div id="panes">
    <div class="detail-pane">
      <h3>Job Details</h3>
      <div id="detail-body" class="detail-body"><span class="empty-msg">Select a job</span></div>
    </div>
    <div class="pdf-stack">
      <div class="pane">
        <h3>Tailored Resume</h3>
        <div id="resume-holder" class="empty">Select a job</div>
      </div>
      <div class="pane">
        <h3>Cover Letter</h3>
        <div id="cover-holder" class="empty">Select a job</div>
      </div>
    </div>
  </div>

<script>
const ROWS_VISIBLE = 8;

const COLUMNS = [
  { key: 'fit_score', label: 'Score', type: 'number',
    render: j => `<span title="${escapeHtml(j.score_reasoning || 'No reasoning recorded')}">${j.fit_score}</span>` },
  { key: 'title', label: 'Title', type: 'string', render: j => escapeHtml(j.title || '') },
  { key: 'company', label: 'Company', type: 'string', render: j => escapeHtml(j.company || '') },
  { key: 'site', label: 'Site', type: 'string', render: j => escapeHtml(j.site || '') },
  { key: 'location', label: 'Location', type: 'string', render: j => escapeHtml(j.location || '') },
  { key: 'salary', label: 'Salary', type: 'string', render: j => escapeHtml(j.salary || '') },
  { key: 'discovered_at', label: 'Discovered', type: 'string',
    render: j => escapeHtml((j.discovered_at || '').slice(0, 10)) },
  { key: 'has_direct_url', label: 'Direct URL', type: 'bool',
    render: j => j.has_direct_url ? '<span title="Direct apply link available">✓</span>' : '<span title="Falls back to listing URL" style="color:#c66">✗</span>' },
  { key: 'status', label: 'Status', type: 'string', render: j => escapeHtml(statusOf(j)) },
  { key: 'active', label: 'Active', type: 'bool', render: j => j.active ? 'yes' : 'no' },
  { key: 'review_later_at', label: 'Review', type: 'string',
    render: j => j.review_later_at ? `<span title="Flagged ${escapeHtml(j.review_later_at)}">★</span>` : '' },
  { key: null, label: 'Open', type: null,
    render: j => j.url ? `<a class="open-link" href="${escapeHtml(j.url)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()">↗</a>` : '' },
];

let jobs = [];
let visible = [];
let selectedId = null;
let checkedScores = new Set();
let searchIds = null;  // null = no search active; Set of job ids = server-matched results
let searchDebounce = null;
const HIDE_FILTERS_STORAGE_KEY = 'applypilot_hide_filters';

function loadStoredHideFilters() {
  try {
    const raw = localStorage.getItem(HIDE_FILTERS_STORAGE_KEY);
    if (!raw) return null;
    const obj = JSON.parse(raw);
    return (obj && typeof obj === 'object') ? obj : null;
  } catch (e) {
    return null;
  }
}

function saveHideFilters() {
  localStorage.setItem(HIDE_FILTERS_STORAGE_KEY, JSON.stringify({ hideInactive, hideApplied, hideFailed, hideReviewLater }));
}

const storedHideFilters = loadStoredHideFilters();
let hideInactive = storedHideFilters ? !!storedHideFilters.hideInactive : false;
let hideApplied = storedHideFilters ? !!storedHideFilters.hideApplied : false;
let hideFailed = storedHideFilters ? !!storedHideFilters.hideFailed : false;
let hideReviewLater = storedHideFilters ? !!storedHideFilters.hideReviewLater : false;

const SORT_STORAGE_KEY = 'applypilot_sort';

function loadStoredSort() {
  try {
    const raw = localStorage.getItem(SORT_STORAGE_KEY);
    if (!raw) return null;
    const obj = JSON.parse(raw);
    if (!obj || typeof obj.sortKey !== 'string' || (obj.sortDir !== 1 && obj.sortDir !== -1)) return null;
    return COLUMNS.some(c => c.key === obj.sortKey) ? obj : null;
  } catch (e) {
    return null;
  }
}

function saveSort() {
  localStorage.setItem(SORT_STORAGE_KEY, JSON.stringify({ sortKey, sortDir }));
}

const storedSort = loadStoredSort();
let sortKey = storedSort ? storedSort.sortKey : 'fit_score';
let sortDir = storedSort ? storedSort.sortDir : -1;

const SCORES_STORAGE_KEY = 'applypilot_checked_scores';
const SELECTED_JOB_STORAGE_KEY = 'applypilot_selected_job';

function loadStoredSelectedId() {
  try {
    const raw = localStorage.getItem(SELECTED_JOB_STORAGE_KEY);
    if (raw === null) return null;
    const id = JSON.parse(raw);
    return typeof id === 'number' ? id : null;
  } catch (e) {
    return null;
  }
}

function saveSelectedId(id) {
  try {
    localStorage.setItem(SELECTED_JOB_STORAGE_KEY, JSON.stringify(id));
  } catch (e) {}
}

function loadStoredScores() {
  try {
    const raw = localStorage.getItem(SCORES_STORAGE_KEY);
    if (!raw) return null;
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? new Set(arr) : null;
  } catch (e) {
    return null;
  }
}

function saveCheckedScores() {
  localStorage.setItem(SCORES_STORAGE_KEY, JSON.stringify([...checkedScores]));
}

function statusOf(j) {
  if (j.apply_status) return j.apply_status;
  return j.applied_at ? 'applied' : 'pending';
}

function statusClass(j) {
  const s = statusOf(j).toLowerCase();
  if (s === 'applied') return 'status-applied';
  if (s.includes('fail') || s.includes('error') || s === 'captcha' || s === 'login_issue') return 'status-failed';
  return '';
}

async function load() {
  const res = await fetch('/api/jobs');
  jobs = await res.json();

  const scores = [...new Set(jobs.map(j => j.fit_score))].sort((a, b) => b - a);
  const stored = loadStoredScores();
  checkedScores = stored
    ? new Set(scores.filter(s => stored.has(s)))
    : new Set(scores.filter(s => s >= 7));

  const scoresDiv = document.getElementById('scores');
  scoresDiv.innerHTML = '';
  scores.forEach(s => {
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = checkedScores.has(s);
    cb.addEventListener('change', () => {
      if (cb.checked) checkedScores.add(s); else checkedScores.delete(s);
      saveCheckedScores();
      applyFilter();
    });
    label.appendChild(cb);
    label.appendChild(document.createTextNode(' ' + s));
    scoresDiv.appendChild(label);
  });

  buildHeader();
  sizeTableToRows();
  applyFilter();
  const storedId = loadStoredSelectedId();
  if (storedId !== null && visible.some(j => j.id === storedId)) {
    selectRow(storedId);
  } else if (visible.length > 0) {
    selectRow(visible[0].id);
  }
}

let columnWidths = {};  // column index -> px, captured from the rendered layout on first paint
let widthsCaptured = false;

function buildHeader() {
  const headerRow = document.getElementById('header-row');
  headerRow.innerHTML = '';
  COLUMNS.forEach((col, i) => {
    const th = document.createElement('th');
    th.textContent = col.label;
    if (columnWidths[i]) th.style.width = columnWidths[i] + 'px';
    if (col.key) {
      th.addEventListener('click', () => {
        if (sortKey === col.key) sortDir *= -1;
        else { sortKey = col.key; sortDir = 1; }
        saveSort();
        render();
      });
      if (sortKey === col.key) {
        const arrow = document.createElement('span');
        arrow.className = 'arrow';
        arrow.textContent = sortDir === 1 ? ' ▲' : ' ▼';
        th.appendChild(arrow);
      }
    }

    const handle = document.createElement('span');
    handle.className = 'resize-handle';
    handle.addEventListener('click', e => e.stopPropagation());
    handle.addEventListener('mousedown', e => startColumnResize(e, th, i));
    th.appendChild(handle);

    headerRow.appendChild(th);
  });

  if (!widthsCaptured) {
    widthsCaptured = true;
    requestAnimationFrame(() => {
      document.querySelectorAll('#header-row th').forEach((th, i) => {
        columnWidths[i] = th.getBoundingClientRect().width;
      });
    });
  }
}

function startColumnResize(e, th, index) {
  e.preventDefault();
  e.stopPropagation();
  const handle = e.target;
  handle.classList.add('resizing');
  const startX = e.clientX;
  const startWidth = th.getBoundingClientRect().width;

  function onMove(ev) {
    const width = Math.max(40, startWidth + (ev.clientX - startX));
    columnWidths[index] = width;
    th.style.width = width + 'px';
  }
  function onUp() {
    handle.classList.remove('resizing');
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
  }
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

function sizeTableToRows() {
  // Measure header + one row height once data exists, then cap the scroll area to ROWS_VISIBLE rows.
  requestAnimationFrame(() => {
    const wrap = document.getElementById('table-wrap');
    const header = document.querySelector('#header-row');
    const firstRow = document.querySelector('#rows tr');
    const headerH = header ? header.getBoundingClientRect().height : 24;
    const rowH = firstRow ? firstRow.getBoundingClientRect().height : 22;
    wrap.style.maxHeight = (headerH + rowH * ROWS_VISIBLE) + 'px';
  });
}

function applyFilter() {
  visible = jobs.filter(j => {
    if (!checkedScores.has(j.fit_score)) return false;
    if (hideInactive && !j.active) return false;
    if (hideApplied && statusOf(j) === 'applied') return false;
    if (hideFailed && statusOf(j) === 'failed') return false;
    if (hideReviewLater && j.review_later_at) return false;
    if (searchIds !== null && !searchIds.has(j.id)) return false;
    return true;
  });
  render();
}

function compareJobs(a, b) {
  const col = COLUMNS.find(c => c.key === sortKey);
  let va, vb;
  if (sortKey === 'status') { va = statusOf(a); vb = statusOf(b); }
  else { va = a[sortKey]; vb = b[sortKey]; }

  let cmp;
  if (col.type === 'number') cmp = (va ?? -Infinity) - (vb ?? -Infinity);
  else if (col.type === 'bool') cmp = (va ? 1 : 0) - (vb ? 1 : 0);
  else cmp = String(va || '').localeCompare(String(vb || ''));
  return cmp * sortDir;
}

function render() {
  visible.sort(compareJobs);
  buildHeader();

  const tbody = document.getElementById('rows');
  tbody.innerHTML = '';
  visible.forEach(j => {
    const tr = document.createElement('tr');
    tr.dataset.id = j.id;
    if (!j.active) tr.classList.add('inactive');
    const sc = statusClass(j);
    if (sc) tr.classList.add(sc);
    if (j.id === selectedId) tr.classList.add('selected');
    tr.innerHTML = COLUMNS.map(col => `<td>${col.render(j)}</td>`).join('');
    tr.addEventListener('click', () => selectRow(j.id));
    tbody.appendChild(tr);
  });

  document.getElementById('count').textContent = `${visible.length} / ${jobs.length} jobs`;
  if (selectedId !== null && !visible.some(j => j.id === selectedId)) selectedId = null;
  scrollSelectedIntoView();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function selectRow(id) {
  selectedId = id;
  saveSelectedId(id);
  render();
  loadPanes();
}

function selectedIndex() {
  return visible.findIndex(j => j.id === selectedId);
}

function moveSelection(delta) {
  const idx = selectedIndex();
  let next;
  if (idx === -1) next = delta > 0 ? 0 : visible.length - 1;
  else next = Math.min(Math.max(idx + delta, 0), visible.length - 1);
  if (visible[next]) selectRow(visible[next].id);
}

function scrollSelectedIntoView() {
  const el = document.querySelector(`tr[data-id="${selectedId}"]`);
  if (el) el.scrollIntoView({ block: 'nearest' });
}

function loadPanes() {
  const j = visible.find(j => j.id === selectedId);
  const detailBody = document.getElementById('detail-body');
  const resumeHolder = document.getElementById('resume-holder');
  const coverHolder = document.getElementById('cover-holder');
  if (!j) {
    detailBody.innerHTML = '<span class="empty-msg">Select a job</span>';
    resumeHolder.outerHTML = '<div id="resume-holder" class="empty">Select a job</div>';
    coverHolder.outerHTML = '<div id="cover-holder" class="empty">Select a job</div>';
    return;
  }

  detailBody.innerHTML = buildDetailHtml(j, null);

  const jobIdAtRequest = j.id;
  fetch(`/api/jobs/${j.id}/detail`)
    .then(res => res.json())
    .then(extra => {
      if (selectedId !== jobIdAtRequest) return;  // selection moved on before this resolved
      detailBody.innerHTML = buildDetailHtml(j, extra);
    });

  const cacheBust = Date.now();
  resumeHolder.outerHTML = j.has_resume
    ? `<iframe id="resume-holder" src="/pdf/resume/${j.id}?t=${cacheBust}"></iframe>`
    : '<div id="resume-holder" class="empty">Not tailored yet</div>';
  coverHolder.outerHTML = j.has_cover
    ? `<iframe id="cover-holder" src="/pdf/cover/${j.id}?t=${cacheBust}"></iframe>`
    : '<div id="cover-holder" class="empty">No cover letter yet</div>';
}

function buildDetailHtml(j, extra) {
  const appUrl = j.application_url || j.url;
  const docs = [
    j.has_resume ? 'Resume: attached' : null,
    j.has_cover ? 'Cover letter: attached' : null,
  ].filter(Boolean).join(' | ') || '—';

  const links = [];
  if (j.url) links.push(`<a href="${escapeHtml(j.url)}" target="_blank" rel="noopener noreferrer">View Job</a>`);
  if (appUrl) links.push(`<a href="${escapeHtml(appUrl)}" target="_blank" rel="noopener noreferrer">Apply Now</a>`);

  const errors = [];
  if (j.detail_error) errors.push(`<strong>Detail error:</strong> ${escapeHtml(j.detail_error)}`);
  if (j.apply_error) errors.push(`<strong>Apply error:</strong> ${escapeHtml(j.apply_error)}`);
  const errorsHtml = errors.length
    ? `<h4>Errors</h4><div class="reasoning">${errors.join('<br>')}</div>`
    : '';

  const longTextHtml = extra
    ? `
      <h4>Description</h4>
      <div class="reasoning">${escapeHtml(extra.description || '—')}</div>
      <h4>Full Description</h4>
      <div class="reasoning">${escapeHtml(extra.full_description || '—')}</div>
    `
    : '<p class="empty-msg">Loading description…</p>';

  return `
    <table class="kv">
      <tr><th>Company</th><td>${escapeHtml(j.company || '—')}</td></tr>
      <tr><th>Location</th><td>${escapeHtml(j.location || '—')}</td></tr>
      <tr><th>Salary</th><td>${escapeHtml(j.salary || '—')}</td></tr>
      <tr><th>Fit Score</th><td>${j.fit_score != null ? j.fit_score + '/10' : '—'}</td></tr>
      <tr><th>Site</th><td>${escapeHtml(j.site || '—')}</td></tr>
      <tr><th>Strategy</th><td>${escapeHtml(j.strategy || '—')}</td></tr>
      <tr><th>Status</th><td>${escapeHtml(statusOf(j))}</td></tr>
      <tr><th>Discovered</th><td>${escapeHtml((j.discovered_at || '').slice(0, 10) || '—')}</td></tr>
      <tr><th>Documents</th><td>${escapeHtml(docs)}</td></tr>
      <tr><th>--url</th><td class="url-cell">${escapeHtml(j.url || '—')}</td></tr>
      <tr><th>application_url</th><td class="url-cell">${escapeHtml(j.application_url || '—')}</td></tr>
    </table>
    <h4>Score Reasoning</h4>
    <div class="reasoning">${escapeHtml(j.score_reasoning || '—')}</div>
    ${errorsHtml}
    ${longTextHtml}
    <p style="margin-top:12px;">${links.join(' | ') || ''}</p>
  `;
}

async function toggleActive() {
  const j = visible.find(j => j.id === selectedId);
  if (!j) return;
  const newActive = !j.active;
  const res = await fetch(`/api/jobs/${j.id}/active`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ active: newActive }),
  });
  const data = await res.json();
  j.active = data.active;
  applyFilter();
  showStatus(data.trello_error ? `Trello sync failed: ${data.trello_error}` : '');
}

async function toggleReviewLater() {
  const j = visible.find(j => j.id === selectedId);
  if (!j) return;
  const res = await fetch(`/api/jobs/${j.id}/review-later`, { method: 'POST' });
  const data = await res.json();
  if (data.error) {
    showStatus(`Toggle review-later failed: ${data.error}`, true);
    return;
  }
  j.review_later_at = data.review_later_at;
  applyFilter();
  showStatus(data.trello_error ? `Trello sync failed: ${data.trello_error}` : '');
}

async function useRealResume() {
  const j = visible.find(j => j.id === selectedId);
  if (!j) return;
  const res = await fetch(`/api/jobs/${j.id}/resume/real`, { method: 'POST' });
  const data = await res.json();
  j.has_resume = true;
  loadPanes();
  showStatus(data.trello_error ? `Trello sync failed: ${data.trello_error}` : 'Resume set to your real resume.', !!data.trello_error);
}

async function markAppliedManually() {
  const j = visible.find(j => j.id === selectedId);
  if (!j) return;
  const res = await fetch(`/api/jobs/${j.id}/mark-applied-manually`, { method: 'POST' });
  const data = await res.json();
  if (data.error) {
    showStatus(`Mark applied failed: ${data.error}`, true);
    return;
  }
  j.apply_status = 'applied';
  j.applied_at = data.applied_at;
  j.apply_error = null;
  applyFilter();
  showStatus(data.trello_error ? `Trello sync failed: ${data.trello_error}` : 'Marked as applied manually.', !!data.trello_error);
}

async function useGenericCoverLetter() {
  const j = visible.find(j => j.id === selectedId);
  if (!j) return;
  const res = await fetch(`/api/jobs/${j.id}/cover/generic`, { method: 'POST' });
  const data = await res.json();
  j.has_cover = true;
  loadPanes();
  showStatus(data.trello_error ? `Trello sync failed: ${data.trello_error}` : 'Cover letter set to the generic one.', !!data.trello_error);
}

async function launchApply() {
  const j = visible.find(j => j.id === selectedId);
  if (!j) return;
  const res = await fetch(`/api/jobs/${j.id}/apply`, { method: 'POST' });
  const data = await res.json();
  showStatus(
    data.launched ? 'Apply (dry-run) launched in a new Terminal window.' : `Apply launch failed: ${data.error}`,
    !data.launched
  );
}

async function launchRealApply() {
  const j = visible.find(j => j.id === selectedId);
  if (!j) return;
  const res = await fetch(`/api/jobs/${j.id}/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dry_run: false }),
  });
  const data = await res.json();
  if (!data.launched) {
    showStatus(`Apply launch failed: ${data.error}`, true);
    return;
  }
  showStatus('Real apply launched in a new Terminal window.', false);
  pollApplyCompletion(j.id, j.apply_status);
}

function pollApplyCompletion(jobId, initialStatus) {
  const maxTicks = 200; // ~10 minutes at 3s intervals
  let ticks = 0;
  const interval = setInterval(async () => {
    ticks++;
    if (ticks > maxTicks) {
      clearInterval(interval);
      showStatus(`Apply for job ${jobId} is still running -- check manually.`, true);
      return;
    }
    let data;
    try {
      const res = await fetch(`/api/jobs/${jobId}/status`);
      if (!res.ok) return;
      data = await res.json();
    } catch {
      return;
    }
    if (data.apply_status === initialStatus || data.apply_status === 'in_progress') return;

    clearInterval(interval);
    const j = jobs.find(x => x.id === jobId);
    if (j) {
      j.apply_status = data.apply_status;
      j.applied_at = data.applied_at;
      j.apply_error = data.apply_error;
    }
    if (selectedId === jobId) {
      // Figure out the neighbor before re-filtering can remove this job
      // from `visible` (e.g. "Hide applied" is checked).
      const idx = selectedIndex();
      const neighborId = (visible[idx + 1] || visible[idx - 1] || {}).id ?? null;
      applyFilter();
      if (neighborId !== null && visible.some(v => v.id === neighborId)) selectRow(neighborId);
    } else {
      applyFilter();
    }
    showStatus(`Apply finished: ${data.apply_status}`, data.apply_status !== 'applied');
  }, 3000);
}

function showStatus(msg, isError = true) {
  const el = document.getElementById('trello-status');
  el.textContent = msg;
  el.style.color = msg ? (isError ? '#c00' : '#080') : '';
  if (msg) setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 6000);
}

document.getElementById('search').addEventListener('input', (e) => {
  const q = e.target.value.trim();
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => runSearch(q), 250);
});

async function runSearch(q) {
  if (!q) {
    searchIds = null;
    applyFilter();
    return;
  }
  const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
  const ids = await res.json();
  searchIds = new Set(ids);
  applyFilter();
}

document.getElementById('hide-inactive').checked = hideInactive;
document.getElementById('hide-inactive').addEventListener('change', (e) => {
  hideInactive = e.target.checked;
  saveHideFilters();
  applyFilter();
});

document.getElementById('hide-applied').checked = hideApplied;
document.getElementById('hide-applied').addEventListener('change', (e) => {
  hideApplied = e.target.checked;
  saveHideFilters();
  applyFilter();
});

document.getElementById('hide-failed').checked = hideFailed;
document.getElementById('hide-failed').addEventListener('change', (e) => {
  hideFailed = e.target.checked;
  saveHideFilters();
  applyFilter();
});

document.getElementById('hide-review-later').checked = hideReviewLater;
document.getElementById('hide-review-later').addEventListener('change', (e) => {
  hideReviewLater = e.target.checked;
  saveHideFilters();
  applyFilter();
});

document.addEventListener('keydown', (e) => {
  if (document.activeElement && document.activeElement.id === 'search') return;
  if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); moveSelection(1); }
  else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); moveSelection(-1); }
  else if (e.key === 'x') { e.preventDefault(); toggleActive(); }
  else if (e.key === 'l') { e.preventDefault(); toggleReviewLater(); }
  else if (e.key === 'a') { e.preventDefault(); launchApply(); }
  else if (e.key === 'A') { e.preventDefault(); launchRealApply(); }
  else if (e.key === 'M') { e.preventDefault(); markAppliedManually(); }
  else if (e.key === 'R') { e.preventDefault(); useRealResume(); }
  else if (e.key === 'G') { e.preventDefault(); useGenericCoverLetter(); }
});

load();
</script>
</body>
</html>
"""
