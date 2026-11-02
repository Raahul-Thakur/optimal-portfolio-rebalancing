"""
Inject submission metadata into paper/main.tex from paper/metadata.json.

The five submission fields (repository URL, archived DOI, ORCID, affiliation,
date) live in ONE place -- paper/metadata.json -- and are written into the
corresponding macros in main.tex by this script.  The operation is idempotent:
run it as often as you like; it always rewrites the macro values to match the
JSON and never accumulates edits.

Workflow
--------
    1. Edit paper/metadata.json (fill in the empty fields).
    2. python scripts/set_metadata.py            # inject values, report readiness
       python scripts/set_metadata.py --build    # inject, then rebuild the PDF

Field handling
--------------
* repo_url, doi, orcid  -> rendered as clickable links (\\url{...}); a bare DOI
  or ORCID iD is expanded to its canonical URL.
* affiliation, date     -> plain text, with LaTeX special characters escaped.
* An empty ("") field is skipped: its placeholder is left untouched, so the
  submission-readiness check still flags it.

This script only edits text macros; it never changes any numerical result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
TEX = os.path.join(PAPER, "main.tex")
META = os.path.join(PAPER, "metadata.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_paper  # noqa: E402  (same directory)


def _escape_text(value: str) -> str:
    """Escape LaTeX-special characters for a plain-text field."""
    for ch, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                    ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("^", r"\textasciicircum{}")):
        value = value.replace(ch, rep)
    return value


def _as_url(value: str) -> str:
    """Wrap a URL-like value in \\url{...} for hyperref rendering."""
    return r"\url{" + value.strip() + "}"


def _doi_url(value: str) -> str:
    v = value.strip()
    if v.lower().startswith(("http://", "https://")):
        return _as_url(v)
    return _as_url("https://doi.org/" + v.lstrip("/"))


def _orcid_url(value: str) -> str:
    v = value.strip()
    if v.lower().startswith(("http://", "https://")):
        return _as_url(v)
    return _as_url("https://orcid.org/" + v.lstrip("/"))


def _set_macro(tex: str, macro: str, latex_value: str) -> str:
    """Replace the body of \\newcommand{\\<macro>}{...} with latex_value."""
    pattern = r"(\\newcommand\{\\" + re.escape(macro) + r"\}\{)(?:[^{}]|\{[^{}]*\})*(\})"
    repl = r"\g<1>" + latex_value.replace("\\", "\\\\") + r"\g<2>"
    new_tex, n = re.subn(pattern, repl, tex)
    if n != 1:
        raise RuntimeError(f"expected exactly one \\newcommand{{\\{macro}}}, "
                           f"found {n}")
    return new_tex


def _set_date(tex: str, latex_value: str) -> str:
    pattern = r"(\\date\{)(?:[^{}]|\{[^{}]*\})*(\})"
    repl = r"\g<1>" + latex_value.replace("\\", "\\\\") + r"\g<2>"
    new_tex, n = re.subn(pattern, repl, tex)
    if n != 1:
        raise RuntimeError(f"expected exactly one \\date{{...}}, found {n}")
    return new_tex


def apply_metadata() -> list:
    """Write metadata.json into main.tex. Returns a list of applied changes."""
    meta = json.load(open(META, encoding="utf-8"))
    tex = open(TEX, encoding="utf-8").read()
    applied = []

    def field(name):
        v = meta.get(name, "")
        return v.strip() if isinstance(v, str) else ""

    if field("repo_url"):
        tex = _set_macro(tex, "coderepo", _as_url(field("repo_url")))
        applied.append(f"coderepo  -> {field('repo_url')}")
    if field("doi"):
        tex = _set_macro(tex, "codedoi", _doi_url(field("doi")))
        applied.append(f"codedoi   -> {field('doi')}")
    if field("orcid"):
        tex = _set_macro(tex, "authororcid", _orcid_url(field("orcid")))
        applied.append(f"authororcid -> {field('orcid')}")
    if field("affiliation"):
        tex = _set_macro(tex, "affiliation", _escape_text(field("affiliation")))
        applied.append(f"affiliation -> {field('affiliation')}")
    if field("date"):
        tex = _set_date(tex, _escape_text(field("date")))
        applied.append(f"date      -> {field('date')}")

    open(TEX, "w", encoding="utf-8", newline="\n").write(tex)
    return applied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="rebuild paper/main.pdf after injecting metadata")
    args = ap.parse_args()

    if not os.path.isfile(META):
        print(f"error: {META} not found", file=sys.stderr)
        return 2

    applied = apply_metadata()
    if applied:
        print("Injected into main.tex:")
        for a in applied:
            print(f"  {a}")
    else:
        print("No non-empty fields in metadata.json; nothing injected.")

    blockers = build_paper.check_submission_ready()
    print("\nSubmission readiness ...")
    if blockers:
        for b in blockers:
            print(f"  STILL PENDING: {b}")
        print("  (fill the matching field in paper/metadata.json)")
    else:
        print("  all submission fields are filled.")

    if args.build:
        print("\nBuilding PDF ...")
        return build_paper.build()
    print("\n(Use --build to regenerate paper/main.pdf.)")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
