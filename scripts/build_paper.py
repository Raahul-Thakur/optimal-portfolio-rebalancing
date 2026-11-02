"""
Build paper/main.pdf from paper/main.tex.

Runs the standard LaTeX/BibTeX cycle (pdflatex, bibtex, pdflatex, pdflatex),
then reports anything that would make the PDF wrong rather than merely ugly:
undefined references, unresolved citations, and missing figures.

Usage
-----
    python scripts/build_paper.py            # build and report
    python scripts/build_paper.py --clean    # remove build products first
    python scripts/build_paper.py --check    # verify consistency, do not build

Requires a TeX distribution on PATH (TeX Live or MiKTeX).  If you do not have
one, upload the repository to Overleaf and compile ``paper/main.tex`` there, or
open ``paper/paper_preview.html`` for a browser-readable version.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
FIGURES = os.path.join(ROOT, "results", "figures")

AUX_SUFFIXES = (".aux", ".bbl", ".blg", ".log", ".out", ".toc",
                ".fls", ".fdb_latexmk", ".synctex.gz")


# --------------------------------------------------------------------------- #
# Static checks (no TeX needed)
# --------------------------------------------------------------------------- #
def check_sources() -> list:
    """Return a list of problems found in the LaTeX sources."""
    tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    bib = open(os.path.join(PAPER, "refs.bib"), encoding="utf-8").read()
    problems = []

    begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
    for env in set(begins) | set(ends):
        if begins.count(env) != ends.count(env):
            problems.append(f"unbalanced environment: {env}")

    figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    for f in figs:
        if not os.path.isfile(os.path.join(FIGURES, f)):
            problems.append(f"figure not found: {f} "
                            f"(run scripts/run_all.py to generate it)")

    on_disk = {f for f in os.listdir(FIGURES) if f.endswith(".png")}
    for orphan in sorted(on_disk - set(figs)):
        problems.append(f"figure generated but never included: {orphan}")

    cited = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    for key in sorted(cited - defined):
        problems.append(f"citation with no bib entry: {key}")
    for key in sorted(defined - cited):
        problems.append(f"bib entry never cited: {key}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex))
    for key in sorted(refs - labels):
        problems.append(f"reference to undefined label: {key}")

    authors = re.findall(r"\\author(?:\[\d+\])?\{([^}]*)\}", tex)
    if not any(a.strip() for a in authors):
        problems.append("the \\author field is empty")

    return problems


def check_submission_ready() -> list:
    """
    Return the reasons this manuscript is not yet ready to send to a venue.

    These are separate from ``check_sources``: the paper compiles correctly with
    placeholders in place, but must not be submitted with them.
    """
    tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    blockers = []

    for macro, what in (("coderepo", "public repository URL"),
                        ("codedoi", "archived DOI"),
                        ("authororcid", "author ORCID")):
        value = re.search(r"\\newcommand\{\\" + macro + r"\}\{([^}]*)\}", tex)
        if value and "TO BE INSERTED" in value.group(1):
            blockers.append(f"{what} is still a placeholder (\\{macro})")

    if "Independent Researcher" in tex:
        blockers.append("affiliation is the placeholder 'Independent Researcher'")

    return blockers


def clean() -> None:
    removed = 0
    for name in os.listdir(PAPER):
        if name.startswith("main") and name.endswith(AUX_SUFFIXES):
            os.remove(os.path.join(PAPER, name))
            removed += 1
    print(f"removed {removed} build product(s)")


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def _run(cmd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)


def build() -> int:
    if shutil.which("pdflatex") is None:
        print("error: pdflatex not found on PATH.\n"
              "Install TeX Live or MiKTeX, compile on Overleaf, or read\n"
              "paper/paper_preview.html instead.", file=sys.stderr)
        return 2

    passes = [
        (["pdflatex", "-interaction=nonstopmode", "main.tex"], "pdflatex (1/3)"),
        (["bibtex", "main"], "bibtex"),
        (["pdflatex", "-interaction=nonstopmode", "main.tex"], "pdflatex (2/3)"),
        (["pdflatex", "-interaction=nonstopmode", "main.tex"], "pdflatex (3/3)"),
    ]
    for cmd, label in passes:
        print(f"  {label} ...")
        proc = _run(cmd)
        # pdflatex returns non-zero on warnings too; a missing PDF is the real
        # failure signal, so we only hard-fail on a fatal TeX error.
        if "Emergency stop" in proc.stdout or "Fatal error" in proc.stdout:
            print(proc.stdout[-3000:], file=sys.stderr)
            return 1

    log = open(os.path.join(PAPER, "main.log"), encoding="utf-8",
               errors="replace").read()
    undefined = re.findall(r"LaTeX Warning: (Reference|Citation) `([^']+)' "
                           r"on page \d+ undefined", log)
    for kind, key in sorted(set(undefined)):
        print(f"  WARNING: undefined {kind.lower()}: {key}")

    pdf = os.path.join(PAPER, "main.pdf")
    if not os.path.isfile(pdf):
        print("error: no PDF was produced", file=sys.stderr)
        return 1

    pages = len(re.findall(rb"/Type\s*/Page[^s]", open(pdf, "rb").read()))
    print(f"\nBuilt {os.path.relpath(pdf, ROOT)} "
          f"({os.path.getsize(pdf) / 1024:.0f} KB, ~{pages} pages)")
    return 1 if undefined else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="remove LaTeX build products before building")
    ap.add_argument("--check", action="store_true",
                    help="run the source checks only; do not invoke LaTeX")
    args = ap.parse_args()

    if args.clean:
        clean()

    print("Checking LaTeX sources ...")
    problems = check_sources()
    for p in problems:
        print(f"  PROBLEM: {p}")
    if not problems:
        print("  sources are consistent: environments balanced, all figures "
              "present, all citations and references resolve.")

    blockers = check_submission_ready()
    print("\nSubmission readiness ...")
    if blockers:
        for b in blockers:
            print(f"  NOT SUBMITTABLE: {b}")
        print("  (the paper still compiles; fix these before sending to a venue)")
    else:
        print("  no placeholders remain.")

    if args.check:
        return 1 if problems else 0

    print("\nBuilding ...")
    return build()


if __name__ == "__main__":
    sys.exit(main())
