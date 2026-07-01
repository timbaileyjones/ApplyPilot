"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import concurrent.futures
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from applypilot.config import RESUME_PATH, load_profile as _load_profile
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client

log = logging.getLogger(__name__)

SCORE_MAX_TOKENS = int(os.environ.get("SCORE_MAX_TOKENS", "2048"))

# Fallback floor when profile doesn't specify one.
_DEFAULT_SALARY_FLOOR = 160_000

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


# Matches salary patterns in free text: "$120k", "$75/hr", "85,000 CAD", "120,000 - 150,000"
_SALARY_RE = re.compile(
    r'\$\s*\d[\d,]*(?:\.\d+)?\s*k?'           # $120k, $85,000
    r'|\d[\d,]*(?:\.\d+)?\s*k?\s*(?:CAD|USD|AUD|GBP|EUR)'  # 85,000 CAD
    r'|\d[\d,]*\s*(?:-|to|–)\s*\d[\d,]*\s*(?:CAD|USD|AUD|GBP|EUR|k\b)',  # 75,000 - 85,000 CAD
    re.IGNORECASE,
)


def _find_salary_in_description(text: str) -> str | None:
    """Extract the first recognisable salary snippet from a job description."""
    m = _SALARY_RE.search(text or "")
    return m.group(0).strip() if m else None


# ── Country filtering ────────────────────────────────────────────────────

# Ordered list of (compiled regex, ISO-3166-1 alpha-2 code).
# US patterns are first so "US, CA, Remote" is caught before the CA/Canada patterns.
_COUNTRY_PATTERNS: list[tuple[re.Pattern, str]] = [(re.compile(p, re.I), c) for p, c in [
    # United States — explicit markers and all 50 states + DC
    (r'\b(united\s+states|u\.s\.a\.?)\b',                  "US"),
    (r'\busa\b',                                            "US"),
    (r'(?:^|[\s,\-–/])(us)(?:[\s,\-–/]|$)',               "US"),  # bare "US" as a token
    (r'\b(alabama|alaska|arizona|arkansas|colorado|connecticut'
     r'|delaware|florida|georgia|hawaii|idaho|illinois|indiana'
     r'|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts'
     r'|michigan|minnesota|mississippi|missouri|montana|nebraska'
     r'|nevada|new\s+hampshire|new\s+jersey|new\s+mexico|new\s+york'
     r'|north\s+carolina|north\s+dakota|ohio|oklahoma|oregon'
     r'|pennsylvania|rhode\s+island|south\s+carolina|south\s+dakota'
     r'|tennessee|texas|utah|vermont|virginia|washington'
     r'|west\s+virginia|wisconsin|wyoming'
     r'|district\s+of\s+columbia|washington\s+d\.?c\.?)\b',  "US"),
    # Canada — full name, CAN prefix, provinces
    (r'\b(canada|canadian)\b',                              "CA"),
    (r'\bcan\b',                                            "CA"),  # "CAN, Ontario"
    (r'\b(ontario|quebec|québec|british\s+columbia|alberta'
     r'|manitoba|saskatchewan|nova\s+scotia|new\s+brunswick'
     r'|newfoundland|nunavut|yukon|northwest\s+territories'
     r'|prince\s+edward\s+island)\b',                      "CA"),
    # United Kingdom
    (r'\b(united\s+kingdom|england|scotland|wales|northern\s+ireland)\b', "GB"),
    (r'\buk\b',                                             "GB"),
    # Rest of world (alphabetical by code)
    (r'\baustralia\b',                                      "AU"),
    (r'\baustria\b',                                        "AT"),
    (r'\bbrazil\b',                                         "BR"),
    (r'\bchile\b',                                          "CL"),
    (r'\bchina\b',                                          "CN"),
    (r'\bcolombia\b',                                       "CO"),
    (r'\bdenmark\b',                                        "DK"),
    (r'\b(finland|suomi)\b',                               "FI"),
    (r'\bfrance\b',                                         "FR"),
    (r'\b(germany|deutschland)\b',                          "DE"),
    (r'\bhong\s+kong\b',                                    "HK"),
    (r'\bisrael\b',                                         "IL"),
    (r'\bindia\b',                                          "IN"),
    (r'\bireland\b',                                        "IE"),
    (r'\bitaly\b',                                          "IT"),
    (r'\bjapan\b',                                          "JP"),
    (r'\b(south\s+korea|korea)\b',                          "KR"),
    (r'\bmexico\b',                                         "MX"),
    (r'\b(malaysia)\b',                                     "MY"),
    (r'\b(netherlands|holland)\b',                          "NL"),
    (r'\bnew\s+zealand\b',                                  "NZ"),
    (r'\bnigeria\b',                                        "NG"),
    (r'\bnorway\b',                                         "NO"),
    (r'\bpakistan\b',                                       "PK"),
    (r'\bphilippines\b',                                    "PH"),
    (r'\bpoland\b',                                         "PL"),
    (r'\bportugal\b',                                       "PT"),
    (r'\bsaudi\s+arabia\b',                                 "SA"),
    (r'\bsingapore\b',                                      "SG"),
    (r'\bsouth\s+africa\b',                                 "ZA"),
    (r'\bspain\b',                                          "ES"),
    (r'\bsweden\b',                                         "SE"),
    (r'\bswitzerland\b',                                    "CH"),
    (r'\btaiwan\b',                                         "TW"),
    (r'\bthailand\b',                                       "TH"),
    (r'\bvietnam\b',                                        "VN"),
]]

# Aliases used to normalise home_countries profile values → ISO codes
_COUNTRY_ALIASES: dict[str, str] = {
    "us": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US", "united states": "US",
    "ca": "CA", "canada": "CA",
    "gb": "GB", "uk": "GB", "united kingdom": "GB",
    "au": "AU", "australia": "AU",
}


def _detect_job_country(location: str | None) -> str | None:
    """Return ISO country code from a location string, or None if ambiguous."""
    if not location:
        return None
    for pattern, code in _COUNTRY_PATTERNS:
        if pattern.search(location):
            return code
    return None


def _normalize_home_countries(home_countries: list[str]) -> frozenset[str]:
    """Convert profile home_countries list to a frozenset of ISO codes."""
    codes: set[str] = set()
    for hc in home_countries:
        codes.add(_COUNTRY_ALIASES.get(hc.lower(), hc.upper()))
    return frozenset(codes)


# ── Scoring Prompt ────────────────────────────────────────────────────────

def _build_score_prompt(home_country_codes: frozenset[str]) -> str:
    home_str = ", ".join(sorted(home_country_codes))
    return f"""You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

GEOGRAPHIC / WORK-AUTHORIZATION ELIGIBILITY (check this first, even if skills match well):
- The candidate can only work from: {home_str}.
- The LOCATION field is often generic or unreliable (e.g. "Remote" with no country) -- read the full DESCRIPTION text for any region, country, or work-authorization restriction: mentions of specific regions/countries other than the candidate's home countries (APAC, EMEA, LATAM, "based in India", "must reside in the UK", "authorized to work in Canada", visa/sponsorship limited to one country, etc.).
- Do not assume "Remote" means globally open -- many "Remote" listings are region-restricted (e.g. "Remote (APAC only)").
- If the posting restricts eligibility to a region/country that does not include the candidate's home countries, this is a hard disqualifier: score 1-2 regardless of skill match, and state the restriction explicitly in REASONING (e.g. "Restricted to APAC/Malaysia-based candidates; not eligible from the US").

SCORING CRITERIA:
- 9-10: Perfect match. Candidate has direct experience in nearly all required skills and qualifications.
- 7-8: Strong match. Candidate has most required skills, minor gaps easily bridged.
- 5-6: Moderate match. Candidate has some relevant skills but missing key requirements.
- 3-4: Weak match. Significant skill gaps, would need substantial ramp-up.
- 1-2: Poor match. Completely different field or experience level, or fails the geographic/work-authorization eligibility check above.

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


def score_job(resume_text: str, job: dict, salary_floor: int = _DEFAULT_SALARY_FLOOR,
              home_country_codes: frozenset[str] = frozenset({"US"})) -> dict:
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

    # Country pre-check: skip the API call if the job is clearly outside home countries.
    # Unknown/ambiguous location is treated as home country (assumed US).
    detected_country = _detect_job_country(job.get("location"))
    if detected_country is not None and detected_country not in home_country_codes:
        log.info(
            "Country filter: skipping AI for '%s' (location=%s, country=%s)",
            job.get("title", "?")[:50], job.get("location"), detected_country,
        )
        return {"score": 0, "company": None, "salary": None, "keywords": "",
                "reasoning": f"Outside home countries ({detected_country})"}

    # Salary floor pre-check: skip the API call if we can already determine
    # the salary is below the floor — from the DB field or the description text.
    if not _is_exempt(job):
        known_salary = (
            job.get("salary")
            or _find_salary_in_description(job.get("full_description"))
        )
        annual = _max_annual_salary(known_salary)
        if annual is not None and annual < salary_floor:
            log.info(
                "Salary floor: skipping AI for '%s' (salary=%s, annual≈$%d)",
                job.get("title", "?")[:50], known_salary, annual,
            )
            return {"score": 0, "company": None,
                    "salary": _normalize_salary(known_salary), "keywords": "",
                    "reasoning": "Below salary floor"}

    messages = [
        {"role": "system", "content": _build_score_prompt(home_country_codes)},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=SCORE_MAX_TOKENS, temperature=0.2)
        return _parse_score_response(response)
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "company": None, "salary": None, "keywords": "", "reasoning": f"LLM error: {e}"}


def run_scoring(limit: int = 0, rescore: bool = False, workers: int = 1) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    profile = _load_profile()
    salary_floor = int(profile.get("compensation", {}).get("salary_floor", _DEFAULT_SALARY_FLOOR))
    home_country_codes = _normalize_home_countries(profile.get("home_countries", ["US", "USA"]))
    log.info("Salary floor: $%d/year | Home countries: %s", salary_floor, sorted(home_country_codes))
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

    if workers > 1:
        log.info("Scoring %d jobs with %d workers...", len(jobs), workers)
    else:
        log.info("Scoring %d jobs...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    results: list[dict] = []

    now = datetime.now(timezone.utc).isoformat()

    def _score(job):
        result = score_job(resume_text, job, salary_floor=salary_floor,
                           home_country_codes=home_country_codes)
        return job, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_score, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job, result = future.result()
            result["url"] = job["url"]
            completed += 1

            if result["score"] == 0:
                errors += 1

            results.append(result)

            log.info(
                "[%d/%d] score=%d  %s",
                completed, len(jobs), result["score"], job.get("title", "?")[:60],
            )

            # Commit immediately so aborts don't lose progress (main-thread DB access)
            conn.execute(
                "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ?, "
                "company = COALESCE(?, company), salary = COALESCE(?, salary) WHERE url = ?",
                (result["score"], f"{result['keywords']}\n{result['reasoning']}", now,
                 result["company"], result["salary"], result["url"]),
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
