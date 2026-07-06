"""Text-to-PDF conversion for tailored resumes and cover letters.

Parses the structured text resume format, renders via an HTML/CSS template,
and exports to PDF using headless Chromium via Playwright.
"""

import logging
from pathlib import Path

from applypilot.config import TAILORED_DIR

log = logging.getLogger(__name__)


# ── Resume Parser ────────────────────────────────────────────────────────

# Recognized section headers, keyed by canonical name used in `sections`.
# Matched case-insensitively so both the AI tailoring pipeline's strict
# ALL-CAPS convention (EXPERIENCE, TECHNICAL SKILLS, ...) and a hand-written
# master resume's Title Case headers (Work Experience, Technical Skills, ...)
# parse the same way.
_SECTION_ALIASES = {
    "SUMMARY": "SUMMARY",
    "OBJECTIVE": "SUMMARY",
    "PROFESSIONAL SUMMARY": "SUMMARY",
    "TECHNICAL SKILLS": "TECHNICAL SKILLS",
    "SKILLS": "TECHNICAL SKILLS",
    "EXPERIENCE": "EXPERIENCE",
    "WORK EXPERIENCE": "EXPERIENCE",
    "PROFESSIONAL EXPERIENCE": "EXPERIENCE",
    "PROJECTS": "PROJECTS",
    "SELECTED PROJECTS": "PROJECTS",
    "EDUCATION": "EDUCATION",
    "CERTIFICATIONS": "CERTIFICATIONS",
    "CERTIFICATES": "CERTIFICATIONS",
}


def _section_header(line: str) -> str | None:
    """Return the canonical section name if `line` is a recognized header, else None."""
    stripped = line.strip()
    if not stripped or stripped.startswith("-") or stripped.startswith("•") or len(stripped) <= 3:
        return None
    return _SECTION_ALIASES.get(stripped.upper())


def parse_resume(text: str) -> dict:
    """Parse a structured text resume into sections.

    Expects a format with header lines (name, title, location, contact)
    followed by section headers (SUMMARY, TECHNICAL SKILLS, etc. -- see
    _SECTION_ALIASES for recognized spellings/casings).

    Args:
        text: Full resume text.

    Returns:
        {"name": str, "title": str, "location": str, "contact": str, "sections": dict}
    """
    lines = [line.rstrip() for line in text.strip().split("\n")]

    # Header: leading non-blank lines, up to whichever comes first -- a
    # recognized section header, or the first blank line after some header
    # content. (Some resumes go straight into an unlabeled summary paragraph
    # with no SUMMARY marker at all; the blank-line fallback keeps that from
    # being swallowed whole as "header".)
    header_lines: list[str] = []
    body_start = len(lines)
    for i, line in enumerate(lines):
        if _section_header(line) is not None:
            body_start = i
            break
        stripped = line.strip()
        if not stripped:
            if header_lines:
                body_start = i
                break
            continue
        header_lines.append(stripped)

    name = header_lines[0] if len(header_lines) > 0 else ""
    title = header_lines[1] if len(header_lines) > 1 else ""
    # The header may have 3 or 4 lines depending on whether location is included
    location = ""
    contact = ""
    if len(header_lines) > 3:
        location = header_lines[2]
        contact = header_lines[3]
    elif len(header_lines) > 2:
        # Could be location or contact -- check for email/phone indicators
        if "@" in header_lines[2] or "|" in header_lines[2]:
            contact = header_lines[2]
        else:
            location = header_lines[2]

    # Split body into sections by recognized headers. Seed with an implicit
    # SUMMARY so an unlabeled leading paragraph (no SUMMARY marker) still
    # lands somewhere instead of being silently dropped.
    sections: dict[str, str] = {}
    current_section: str | None = "SUMMARY"
    current_lines: list[str] = []

    def _flush() -> None:
        if not current_section:
            return
        text = "\n".join(current_lines).strip()
        if text:
            sections[current_section] = text

    for line in lines[body_start:]:
        canonical = _section_header(line)
        if canonical is not None:
            _flush()
            current_section = canonical
            current_lines = []
        else:
            current_lines.append(line)

    _flush()

    return {
        "name": name,
        "title": title,
        "location": location,
        "contact": contact,
        "sections": sections,
    }


def parse_skills(text: str) -> list[tuple[str, str]]:
    """Parse skills section into (category, value) pairs.

    Args:
        text: The TECHNICAL SKILLS section text.

    Returns:
        List of (category_name, skills_string) tuples.
    """
    skills: list[tuple[str, str]] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            cat, val = line.split(":", 1)
            skills.append((cat.strip(), val.strip()))
    return skills


def parse_entries(text: str) -> list[dict]:
    """Parse experience/project entries from section text.

    Args:
        text: The EXPERIENCE or PROJECTS section text.

    Returns:
        List of {"title": str, "subtitle": str, "bullets": list[str]} dicts.
    """
    entries: list[dict] = []
    lines = text.strip().split("\n")
    current: dict | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") or stripped.startswith("\u2022 "):
            if current:
                current["bullets"].append(stripped[2:].strip())
        elif current is None or (
            not stripped.startswith("-")
            and not stripped.startswith("\u2022")
            and len(current.get("bullets", [])) > 0
        ):
            # New entry
            if current:
                entries.append(current)
            current = {"title": stripped, "subtitle": "", "bullets": []}
        elif current and not current["subtitle"]:
            current["subtitle"] = stripped
        else:
            if current:
                current["bullets"].append(stripped)

    if current:
        entries.append(current)

    return entries


# ── HTML Template ────────────────────────────────────────────────────────

def build_html(resume: dict) -> str:
    """Build professional resume HTML from parsed data.

    Args:
        resume: Parsed resume dict from parse_resume().

    Returns:
        Complete HTML string ready for PDF rendering.
    """
    sections = resume["sections"]

    # Skills
    skills_html = ""
    if "TECHNICAL SKILLS" in sections:
        skills = parse_skills(sections["TECHNICAL SKILLS"])
        rows = ""
        for cat, val in skills:
            rows += f'<div class="skill-row"><span class="skill-cat">{cat}:</span> {val}</div>\n'
        skills_html = f'<div class="section"><div class="section-title">Technical Skills</div>{rows}</div>'

    # Experience
    exp_html = ""
    if "EXPERIENCE" in sections:
        entries = parse_entries(sections["EXPERIENCE"])
        items = ""
        for e in entries:
            bullets = "".join(f"<li>{b}</li>" for b in e["bullets"])
            subtitle = f'<div class="entry-subtitle">{e["subtitle"]}</div>' if e["subtitle"] else ""
            items += f'<div class="entry"><div class="entry-title">{e["title"]}</div>{subtitle}<ul>{bullets}</ul></div>'
        exp_html = f'<div class="section"><div class="section-title">Experience</div>{items}</div>'

    # Projects
    proj_html = ""
    if "PROJECTS" in sections:
        entries = parse_entries(sections["PROJECTS"])
        items = ""
        for e in entries:
            bullets = "".join(f"<li>{b}</li>" for b in e["bullets"])
            subtitle = f'<div class="entry-subtitle">{e["subtitle"]}</div>' if e["subtitle"] else ""
            items += f'<div class="entry"><div class="entry-title">{e["title"]}</div>{subtitle}<ul>{bullets}</ul></div>'
        proj_html = f'<div class="section"><div class="section-title">Projects</div>{items}</div>'

    # Education
    edu_html = ""
    if "EDUCATION" in sections:
        edu_text = sections["EDUCATION"].strip()
        edu_html = f'<div class="section"><div class="section-title">Education</div><div class="edu">{edu_text}</div></div>'

    # Certifications
    cert_html = ""
    if "CERTIFICATIONS" in sections:
        cert_lines = [
            line.strip().lstrip("•").strip()
            for line in sections["CERTIFICATIONS"].split("\n")
            if line.strip()
        ]
        items = "".join(f"<li>{c}</li>" for c in cert_lines)
        cert_html = f'<div class="section"><div class="section-title">Certifications</div><ul>{items}</ul></div>'

    # Summary
    summary_html = ""
    if "SUMMARY" in sections:
        summary_html = f'<div class="section"><div class="section-title">Summary</div><div class="summary">{sections["SUMMARY"].strip()}</div></div>'

    # Contact line parsing
    contact = resume["contact"]
    contact_parts = [p.strip() for p in contact.split("|")] if contact else []
    contact_html = " &nbsp;|&nbsp; ".join(contact_parts)

    # Location line (may be empty)
    location_html = f'<div class="location">{resume["location"]}</div>' if resume["location"] else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: letter;
    margin: 0.35in 0.5in;
}}
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}
body {{
    font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.35;
    color: #1a1a1a;
}}
.header {{
    text-align: center;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 1.5px solid #2a7ab5;
}}
.name {{
    font-size: 18pt;
    font-weight: 700;
    color: #1a3a5c;
    letter-spacing: 0.5px;
}}
.title {{
    font-size: 10.5pt;
    color: #3a6b8c;
    margin: 1px 0;
}}
.location {{
    font-size: 9pt;
    color: #555;
}}
.contact {{
    font-size: 9pt;
    color: #444;
    margin-top: 1px;
}}
.contact a {{
    color: #2c3e50;
    text-decoration: none;
}}
.section {{
    margin-top: 5px;
}}
.section-title {{
    font-size: 10pt;
    font-weight: 700;
    color: #1a3a5c;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1.5px solid #2a7ab5;
    padding-bottom: 1px;
    margin-bottom: 3px;
}}
.summary {{
    font-size: 9.5pt;
    color: #333;
    line-height: 1.4;
}}
.skill-row {{
    font-size: 9.5pt;
    margin: 0;
    line-height: 1.35;
}}
.skill-cat {{
    font-weight: 600;
    color: #1a3a5c;
}}
.entry {{
    margin-bottom: 4px;
    break-inside: avoid;
}}
.entry-title {{
    font-weight: 600;
    font-size: 10pt;
    color: #1a3a5c;
}}
.entry-subtitle {{
    font-size: 9pt;
    color: #4a7a9b;
    font-style: italic;
    margin-bottom: 1px;
}}
ul {{
    margin-left: 14px;
    padding: 0;
}}
li {{
    font-size: 9.5pt;
    margin-bottom: 1px;
    line-height: 1.35;
}}
.edu {{
    font-size: 10pt;
}}
</style>
</head>
<body>
<div class="header">
    <div class="name">{resume['name']}</div>
    <div class="title">{resume['title']}</div>
    {location_html}
    <div class="contact">{contact_html}</div>
</div>
{summary_html}
{skills_html}
{exp_html}
{proj_html}
{edu_html}
{cert_html}
</body>
</html>"""


# ── Cover Letter Template ────────────────────────────────────────────────

LETTER_MARGIN_TOP_IN = 0.75
LETTER_MARGIN_SIDE_IN = 0.9
LETTER_MAX_FONT_PT = 11.0
LETTER_MIN_FONT_PT = 8.0


def build_letter_html(text: str, font_size_pt: float = LETTER_MAX_FONT_PT) -> str:
    """Build simple prose-letter HTML from raw cover letter text.

    Unlike build_html(), this does not assume a resume's ALL-CAPS section
    structure -- it just splits on blank lines and renders each block as a
    paragraph. Using the resume parser on cover-letter prose silently drops
    everything past the first few lines (they get eaten as name/title/
    location/contact), so cover letters need their own renderer.

    line-height and paragraph spacing scale down with font_size_pt so a
    smaller font also buys back the vertical space it needs to fit one page.
    """
    import html as _html

    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    paragraphs_html = "".join(
        f'<p>{_html.escape(block).replace(chr(10), "<br>")}</p>\n' for block in blocks
    )

    line_height = 1.5 - 0.15 * (LETTER_MAX_FONT_PT - font_size_pt)
    para_gap = 10 - 0.6 * (LETTER_MAX_FONT_PT - font_size_pt)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: letter;
    margin: {LETTER_MARGIN_TOP_IN}in {LETTER_MARGIN_SIDE_IN}in;
}}
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}
body {{
    font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
    font-size: {font_size_pt}pt;
    line-height: {line_height};
    color: #1a1a1a;
}}
p {{
    margin-bottom: {para_gap}pt;
}}
</style>
</head>
<body>
{paragraphs_html}
</body>
</html>"""


# ── PDF Renderer ─────────────────────────────────────────────────────────

def render_pdf(html: str, output_path: str) -> None:
    """Render HTML to PDF using Playwright's headless Chromium.

    Args:
        html: Complete HTML string.
        output_path: Path to write the PDF file.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.pdf(
            path=output_path,
            format="Letter",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            print_background=True,
        )
        browser.close()


def render_letter_pdf(text: str, output_path: str) -> None:
    """Render cover letter text to PDF, shrinking the font to fit one page.

    Never drops content -- if it still doesn't fit at LETTER_MIN_FONT_PT,
    renders at that floor size and lets it spill onto a second page rather
    than truncating.
    """
    from playwright.sync_api import sync_playwright

    printable_width_px = round((8.5 - 2 * LETTER_MARGIN_SIDE_IN) * 96)
    printable_height_px = round((11 - 2 * LETTER_MARGIN_TOP_IN) * 96)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": printable_width_px, "height": printable_height_px})

        font_size = LETTER_MAX_FONT_PT
        while True:
            page.set_content(build_letter_html(text, font_size_pt=font_size), wait_until="networkidle")
            content_height = page.evaluate("document.body.scrollHeight")
            if content_height <= printable_height_px or font_size <= LETTER_MIN_FONT_PT:
                break
            font_size = max(font_size - 0.5, LETTER_MIN_FONT_PT)

        page.pdf(
            path=output_path,
            format="Letter",
            margin={
                "top": f"{LETTER_MARGIN_TOP_IN}in", "bottom": f"{LETTER_MARGIN_TOP_IN}in",
                "left": f"{LETTER_MARGIN_SIDE_IN}in", "right": f"{LETTER_MARGIN_SIDE_IN}in",
            },
            print_background=True,
        )
        browser.close()


# ── Public API ───────────────────────────────────────────────────────────

def convert_to_pdf(
    text_path: Path, output_path: Path | None = None, html_only: bool = False
) -> Path:
    """Convert a structured text resume (ALL-CAPS section headers) to PDF.

    Args:
        text_path: Path to the .txt file to convert.
        output_path: Optional override for the output path. Defaults to same
            name with .pdf extension.
        html_only: If True, output HTML instead of PDF.

    Returns:
        Path to the generated PDF (or HTML) file.
    """
    text_path = Path(text_path)
    text = text_path.read_text(encoding="utf-8")
    resume = parse_resume(text)
    html = build_html(resume)

    if html_only:
        out = output_path or text_path.with_suffix(".html")
        out = Path(out)
        out.write_text(html, encoding="utf-8")
        log.info("HTML generated: %s", out)
        return out

    out = output_path or text_path.with_suffix(".pdf")
    out = Path(out)
    render_pdf(html, str(out))
    log.info("PDF generated: %s", out)
    return out


def convert_cover_letter_to_pdf(
    text_path: Path, output_path: Path | None = None, html_only: bool = False
) -> Path:
    """Convert a plain-prose cover letter to PDF.

    Cover letters don't have the resume's ALL-CAPS section structure, so
    convert_to_pdf()'s parser silently drops everything past the first few
    lines. This renders the text as simple paragraphs instead.

    Args:
        text_path: Path to the .txt file to convert.
        output_path: Optional override for the output path. Defaults to same
            name with .pdf extension.
        html_only: If True, output HTML instead of PDF.

    Returns:
        Path to the generated PDF (or HTML) file.
    """
    text_path = Path(text_path)
    text = text_path.read_text(encoding="utf-8")

    if html_only:
        out = output_path or text_path.with_suffix(".html")
        out = Path(out)
        out.write_text(build_letter_html(text), encoding="utf-8")
        log.info("HTML generated: %s", out)
        return out

    out = output_path or text_path.with_suffix(".pdf")
    out = Path(out)
    render_letter_pdf(text, str(out))
    log.info("PDF generated: %s", out)
    return out


def batch_convert(limit: int = 50) -> int:
    """Convert .txt files in TAILORED_DIR that don't have corresponding PDFs.

    Scans for .txt files (excluding _JOB.txt and _REPORT.json), checks if a
    .pdf with the same stem already exists, and converts any that are missing.

    Args:
        limit: Maximum number of files to convert.

    Returns:
        Number of PDFs generated.
    """
    if not TAILORED_DIR.exists():
        log.warning("Tailored directory does not exist: %s", TAILORED_DIR)
        return 0

    txt_files = sorted(TAILORED_DIR.glob("*.txt"))
    # Exclude _JOB.txt and _CL.txt files from resume conversion
    # (they get their own conversion calls)
    candidates = [
        f for f in txt_files
        if not f.name.endswith("_JOB.txt")
    ]

    # Filter to those without a corresponding PDF
    to_convert: list[Path] = []
    for f in candidates:
        pdf_path = f.with_suffix(".pdf")
        if not pdf_path.exists():
            to_convert.append(f)
        if len(to_convert) >= limit:
            break

    if not to_convert:
        log.info("All text files already have PDFs.")
        return 0

    log.info("Converting %d files to PDF...", len(to_convert))
    converted = 0
    for f in to_convert:
        try:
            convert_to_pdf(f)
            converted += 1
        except Exception as e:
            log.error("Failed to convert %s: %s", f.name, e)

    log.info("Done: %d/%d PDFs generated in %s", converted, len(to_convert), TAILORED_DIR)
    return converted
