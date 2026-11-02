"""
Consistency between the manuscript and the results it reports.

A finished paper drifts from its code the moment a number is edited by hand.
These tests parse ``paper/main.tex`` and check every quoted quantity against
``results/tables/metrics.json``, at the precision the paper displays.  They also
verify the mechanical health of the LaTeX source: balanced environments, every
figure present on disk, every citation resolvable.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import pytest


@pytest.fixture(scope="module")
def tex(project_root) -> str:
    return open(os.path.join(project_root, "paper", "main.tex"),
                encoding="utf-8").read()


@pytest.fixture(scope="module")
def bib(project_root) -> str:
    return open(os.path.join(project_root, "paper", "refs.bib"),
                encoding="utf-8").read()


@pytest.fixture(scope="module")
def metrics(results_dir) -> dict:
    return json.load(open(os.path.join(results_dir, "tables", "metrics.json")))


def _by_strategy(rows):
    return {r["strategy"]: r for r in rows}


def _table_body(tex: str, label: str) -> str:
    """
    The data rows of the tabular carrying the given \\label.

    In this manuscript \\label precedes \\begin{tabular} inside the float, so we
    take the *first* tabular after the label and return what lies between
    \\midrule and \\bottomrule.
    """
    parts = tex.split(r"\label{" + label + "}")
    assert len(parts) == 2, f"label {label} not found exactly once"
    after = parts[1]
    start = after.index(r"\begin{tabular}")
    end = after.index(r"\end{tabular}", start)
    tabular = after[start:end]
    return tabular.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]


def _cells(line: str):
    """
    Split a LaTeX table row on unescaped ampersands.

    A plain ``line.split("&")`` also splits inside the escaped ``\\&`` of
    "Buy \\& hold", which silently drops that row from the comparison.
    """
    return [c.strip() for c in re.split(r"(?<!\\)&", line)]


def _numbers(cell: str):
    """Every plain decimal in a LaTeX table cell, as floats."""
    return [float(m) for m in re.findall(r"[-+]?\d*\.?\d+", cell)]


# --------------------------------------------------------------------------- #
# Mechanical health of the source
# --------------------------------------------------------------------------- #
def test_environments_are_balanced(tex):
    begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", tex))
    ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", tex))
    assert begins == ends


def test_document_is_complete(tex):
    assert tex.rstrip().endswith(r"\end{document}")
    for required in (r"\begin{abstract}", r"\section{Introduction}",
                     r"\section{Conclusion}", r"\bibliography{refs}"):
        assert required in tex


def test_the_paper_has_an_author(tex):
    """A blank \\author is the classic unfinished-preprint giveaway."""
    authors = re.findall(r"\\author(?:\[\d+\])?\{([^}]*)\}", tex)
    assert authors, "no \\author command found"
    assert any(a.strip() for a in authors), "\\author{} is empty"


def test_every_referenced_figure_exists(tex, results_dir):
    figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    assert figs
    for f in figs:
        assert os.path.isfile(os.path.join(results_dir, "figures", f)), f


def test_every_generated_figure_is_used(tex, results_dir):
    """No orphans: a figure worth generating is a figure worth showing."""
    used = set(re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
    on_disk = {f for f in os.listdir(os.path.join(results_dir, "figures"))
               if f.endswith(".png")}
    assert on_disk - used == set(), f"figures never cited in the paper: {on_disk - used}"


def test_every_citation_resolves(tex, bib):
    cited = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert cited - defined == set(), f"missing bib entries: {cited - defined}"


def test_no_unused_bibliography_entries(tex, bib):
    cited = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert defined - cited == set(), f"uncited bib entries: {defined - cited}"


def test_every_label_is_referenced(tex):
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    dangling = {r for r in refs if r not in labels}
    assert dangling == set(), f"\\ref to undefined labels: {dangling}"


# --------------------------------------------------------------------------- #
# The model setup quoted in the text
# --------------------------------------------------------------------------- #
def test_quoted_merton_fraction_matches_the_config(tex, metrics):
    assert metrics["config"]["merton_fraction"] == pytest.approx(0.30, abs=5e-3)
    assert "0.30" in tex


def test_quoted_path_count_matches_the_run(tex, metrics):
    assert metrics["config"]["n_paths"] == 20000
    assert r"20{,}000" in tex


def test_quoted_parameter_count_matches_the_network(tex, metrics):
    assert metrics["ml"]["n_params"] == 4417
    assert r"4{,}417" in tex


def test_quoted_band_matches_the_solver(tex, metrics):
    lo = metrics["dp"]["band_t0_lower"]
    hi = metrics["dp"]["band_t0_upper"]
    assert (lo, hi) == pytest.approx((0.235, 0.365), abs=1e-9)
    assert f"{lo:.3f}" in tex and f"{hi:.3f}" in tex


def test_quoted_validation_mse_matches_training(tex, metrics):
    assert metrics["ml"]["final_val_mse"] == pytest.approx(2.1e-4, rel=0.05)
    assert r"2.1\times10^{-4}" in tex


# --------------------------------------------------------------------------- #
# Performance tables
# --------------------------------------------------------------------------- #
TEX_TO_JSON = {
    "DP optimal": "DP optimal",
    "Buy \\& hold": "Buy & hold",
    "Periodic": "Periodic",
    "ML imitation": "ML imitation",
}


@pytest.mark.parametrize("label,key", [
    ("tab:perf-on", "performance_on_target"),
    ("tab:perf-off", "performance_off_target"),
])
def test_performance_tables_match_the_metrics(tex, metrics, label, key):
    rows = _by_strategy(metrics[key])
    body = _table_body(tex, label)
    seen = 0
    for line in body.splitlines():
        cells = _cells(line)
        if len(cells) < 5 or cells[0] not in TEX_TO_JSON:
            continue
        seen += 1
        row = rows[TEX_TO_JSON[cells[0]]]

        ce = _numbers(cells[1])[0]
        assert ce == pytest.approx(row["certainty_equivalent"], abs=5e-6), cells[0]

        ci_lo, ci_hi = _numbers(cells[2])
        assert ci_lo == pytest.approx(row["ce_ci_low"], abs=5e-6), cells[0]
        assert ci_hi == pytest.approx(row["ce_ci_high"], abs=5e-6), cells[0]

        ewt = _numbers(cells[3])[0]
        assert ewt == pytest.approx(row["mean_terminal_wealth"], abs=5e-5), cells[0]

        turnover = _numbers(cells[4])[0]
        assert turnover == pytest.approx(row["mean_turnover"], abs=5e-5), cells[0]

    assert seen == 4, f"expected 4 strategy rows in {label}, parsed {seen}"


def test_paired_table_matches_the_metrics(tex, metrics):
    lookup = {("on-target", r["vs"]): r for r in metrics["paired_on_target"]}
    lookup.update({("off-target", r["vs"]): r for r in metrics["paired_off_target"]})

    body = _table_body(tex, "tab:paired")
    seen = 0
    for line in body.splitlines():
        cells = _cells(line)
        if len(cells) < 5 or cells[0] not in ("on-target", "off-target"):
            continue
        strategy = TEX_TO_JSON.get(cells[1])
        if strategy is None:
            continue
        seen += 1
        row = lookup[(cells[0], strategy)]

        # Column 3 is the mean advantage in units of 10^{-3}.
        adv = _numbers(cells[2])[0] * 1e-3
        assert adv == pytest.approx(row["mean_utility_advantage"], abs=5e-7), cells

        t_stat = _numbers(cells[3])[0]
        assert t_stat == pytest.approx(row["t_stat"], abs=0.05), cells

        claims_better = cells[4].lower().startswith("yes")
        assert claims_better == row["reference_better_95pct"], cells

    assert seen == 6, f"expected 6 paired rows, parsed {seen}"


# --------------------------------------------------------------------------- #
# The central narrative claims
# --------------------------------------------------------------------------- #
def test_the_paper_is_right_that_ml_is_the_worst_strategy(metrics):
    for key in ("performance_on_target", "performance_off_target"):
        rows = _by_strategy(metrics[key])
        ml = rows["ML imitation"]["certainty_equivalent"]
        assert ml == min(r["certainty_equivalent"] for r in metrics[key])
        assert rows["ML imitation"]["mean_turnover"] == max(
            r["mean_turnover"] for r in metrics[key])


def test_the_paper_is_right_that_dp_is_the_best_strategy(metrics):
    for key in ("performance_on_target", "performance_off_target"):
        rows = _by_strategy(metrics[key])
        best = max(r["certainty_equivalent"] for r in metrics[key])
        assert rows["DP optimal"]["certainty_equivalent"] == pytest.approx(
            best, abs=1e-6)


def test_the_paper_is_right_that_ml_tracks_closely_yet_overtrades(metrics):
    """Low action error, high turnover -- the paper's central lesson."""
    rows = _by_strategy(metrics["performance_on_target"])
    ml, dp = rows["ML imitation"], rows["DP optimal"]
    assert ml["tracking_error"] < 0.01               # accurate in action space
    assert ml["mean_turnover"] > 1000 * dp["mean_turnover"]   # yet churns


def test_the_paper_is_right_about_the_on_target_tie_with_buy_and_hold(metrics):
    """The one comparison the paper reports as *not* significant."""
    row = next(r for r in metrics["paired_on_target"] if r["vs"] == "Buy & hold")
    assert row["reference_better_95pct"] is False
    assert abs(row["t_stat"]) < 2.0


def test_comparative_statics_tables_are_monotone(metrics):
    widths = metrics["band_vs_cost"]["band_width"]
    assert all(b > a for a, b in zip(widths, widths[1:]))
    widths = metrics["band_vs_gamma"]["band_width"]
    assert all(b < a for a, b in zip(widths, widths[1:]))
