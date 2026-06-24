"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from applypilot.config import RESUME_PATH, load_profile
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client

log = logging.getLogger(__name__)

SCORE_MAX_TOKENS = int(os.environ.get("SCORE_MAX_TOKENS", "2048"))

# Jobs whose annual equivalent salary is below this floor are auto-scored 0.
SALARY_FLOOR = 160_000

# Sites/companies exempt from the salary floor (task-based gig platforms).
_SALARY_FLOOR_EXEMPT = frozenset({"mercor"})


def _normalize_salary(salary: str) -> str:
    """Zero-pad sub-100k figures so text-based sort stays numeric.

    "$95k" → "$095k", "$80k-$120k" → "$080k-$120k", "$100k+" unchanged.
    """
    return re.sub(r'\b(\d{1,2})(k)', lambda m: f"{int(m.group(1)):03d}{m.group(2)}", salary, flags=re.IGNORECASE)


def _is_exempt(job: dict) -> bool:
    """Return True if the job is on a task-based platform exempt from the salary floor."""
    site = (job.get("site") or "").lower()
    url = (job.get("url") or "").lower()
    return any(s in site or s in url for s in _SALARY_FLOOR_EXEMPT)


def _max_annual_salary(salary_str: str | None) -> int | None:
    """Parse a salary string and return the maximum annual equivalent in dollars.

    Returns None when the string is absent or too ambiguous to classify
    (e.g. a bare number with no time-unit hint), so the floor rule is skipped.
    """
    if not salary_str:
        return None
    s = salary_str.lower()

    # Pull every number+optional-k from the string (handles ranges like "$120k-$150k")
    raw = re.findall(r'(\d[\d,]*)(\s*k)?', s)
    values = []
    for digits, k in raw:
        try:
            val = float(digits.replace(",", ""))
            if k.strip():
                val *= 1_000
            values.append(val)
        except ValueError:
            continue
    if not values:
        return None

    peak = max(values)

    if re.search(r'/\s*h(?:r|our)?|per\s+hour|hourly', s):
        return int(peak * 2_080)  # 40 hrs/week × 52 weeks
    if re.search(r'/\s*y(?:r|ear)|annual(?:ly)?|per\s+year|per\s+annum', s):
        return int(peak)
    if peak >= 30_000:  # unambiguously annual (nobody makes $30k/hr)
        return int(peak)
    return None  # small bare number with no unit — don't guess


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

SCORING CRITERIA:
- 9-10: Perfect match. Candidate has direct experience in nearly all required skills and qualifications.
- 7-8: Strong match. Candidate has most required skills, minor gaps easily bridged.
- 5-6: Moderate match. Candidate has some relevant skills but missing key requirements.
- 3-4: Weak match. Significant skill gaps, would need substantial ramp-up.
- 1-2: Poor match. Completely different field or experience level.

IMPORTANT FACTORS:
- Weight technical skills heavily (programming languages, frameworks, tools)
- Consider transferable experience (automation, scripting, API work)
- Factor in the candidate's project experience
- Be realistic about experience level vs. job requirements (years of experience, seniority)

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
COMPANY: [company name extracted from the job description, or "Unknown" if not mentioned]
SALARY: [pay range extracted from the job description, e.g. "$120k-$150k/year" or "$75/hr" or "Not specified"]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score]"""


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "company": str|None, "salary": str|None, "keywords": str, "reasoning": str}
    """
    score = 0
    company = None
    salary = None
    keywords = ""
    reasoning = response

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("COMPANY:"):
            val = line.replace("COMPANY:", "").strip()
            company = val if val and val.lower() != "unknown" else None
        elif line.startswith("SALARY:"):
            val = line.replace("SALARY:", "").strip()
            salary = _normalize_salary(val) if val and val.lower() != "not specified" else None
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "company": company, "salary": salary,
            "keywords": keywords, "reasoning": reasoning}


def score_job(resume_text: str, job: dict) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    messages = [
        {"role": "system", "content": SCORE_PROMPT},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=SCORE_MAX_TOKENS, temperature=0.2)
        result = _parse_score_response(response)

        # Salary floor: override score to 0 for sub-$160k roles (exempt gig platforms)
        if not _is_exempt(job):
            effective_salary = result["salary"] or job.get("salary")
            annual = _max_annual_salary(effective_salary)
            if annual is not None and annual < SALARY_FLOOR:
                log.info(
                    "Salary floor: zeroing '%s' (salary=%s, annual≈$%d)",
                    job.get("title", "?")[:50], effective_salary, annual,
                )
                result["score"] = 0

        return result
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "company": None, "salary": None, "keywords": "", "reasoning": f"LLM error: {e}"}


def run_scoring(limit: int = 0, rescore: bool = False) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    results: list[dict] = []

    for job in jobs:
        result = score_job(resume_text, job)
        result["url"] = job["url"]
        completed += 1

        if result["score"] == 0:
            errors += 1

        results.append(result)

        log.info(
            "[%d/%d] score=%d  %s",
            completed, len(jobs), result["score"], job.get("title", "?")[:60],
        )

    # Write scores to DB; COALESCE preserves existing salary from discovery if LLM finds nothing
    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ?, "
            "company = COALESCE(?, company), salary = COALESCE(?, salary) WHERE url = ?",
            (r["score"], f"{r['keywords']}\n{r['reasoning']}", now,
             r["company"], r["salary"], r["url"]),
        )
    conn.commit()

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs (%.1f jobs/sec)", len(results), elapsed, len(results) / elapsed if elapsed > 0 else 0)

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": len(results),
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
