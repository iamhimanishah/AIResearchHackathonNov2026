"""C4 figures: every SVG parses as XML, carries at least one data mark and no NaN coordinate;
screening mode reads only selection-half fields."""

import xml.etree.ElementTree as ET

import pytest

from clharness.analysis import attribution as A
from clharness.analysis import figures as F
from clharness.fake import fake_rows


@pytest.fixture(scope="module")
def confirmatory():
    return A.analyze(fake_rows(seed=0), confirmatory=True, n_boot=300)


@pytest.fixture(scope="module")
def screening():
    return A.analyze(fake_rows(seed=0), confirmatory=False, n_boot=150)


def _check_svg(path, marks: bool = True):
    text = path.read_text()
    root = ET.fromstring(text)  # parses as XML
    assert root.tag.endswith("svg"), path
    if marks:
        elems = [e for e in root.iter() if e.tag.split("}")[-1] in ("circle", "path", "rect", "line")]
        assert elems, f"{path}: no data marks"
    for e in root.iter():
        for v in e.attrib.values():
            for word in v.split(","):
                assert not word.strip().lower().startswith("nan"), f"{path}: NaN in {e.attrib}"
    return text


def test_all_five_figures_written_and_valid(confirmatory, tmp_path):
    paths = {p.name: p for p in F.render_all(confirmatory, tmp_path)}
    assert set(paths) == {"frontier.svg", "effects.svg", "claims.svg", "per_task.svg", "cost.svg"}
    for p in paths.values():
        _check_svg(p)


def test_frontier_marks_the_reference_arm_and_the_lr_curve(confirmatory, tmp_path):
    F.render_all(confirmatory, tmp_path)
    text = _check_svg(tmp_path / "frontier.svg")
    assert "reference arm" in text and "curve to beat" in text


def test_frontier_y_axis_is_in_points_not_fractions(confirmatory, tmp_path):
    """The y axis is labelled 'points' — the ticks must be the delta x100, not the raw
    fraction (fake data spans several points, so a tick > 0.5 must exist)."""
    F.render_all(confirmatory, tmp_path)
    root = ET.fromstring((tmp_path / "frontier.svg").read_text())
    y_ticks = [float(e.text) for e in root.iter()
               if e.tag.endswith("text") and e.get("text-anchor") == "end"]
    assert y_ticks and max(abs(v) for v in y_ticks) > 0.5


def test_claims_lists_every_test_with_a_verdict(confirmatory, tmp_path):
    F.render_all(confirmatory, tmp_path)
    text = (tmp_path / "claims.svg").read_text()
    for r in confirmatory["claims"]:
        assert r["verdict"] in text
    assert "no-mitigation-lr5e-05" in text  # the lower-LR second marker names its control


def test_per_task_marks_collapsed_tasks(confirmatory, tmp_path):
    """In the fake rows the collapsed runs are neither promoted nor controls, so nothing is
    marked; a collapsed config that is drawn gets a red ring and a legend note."""
    analysis = dict(confirmatory)
    drawn = [dict(c) for c in analysis["configs_report"]
             if not c["techniques"] or c["key"] in {d["key"] for d in analysis["promotion"]
                                                  if d["promoted"]}]
    drawn[0]["collapsed"] = ["pubmedqa_pqal"]
    analysis["configs_report"] = [dict(c, collapsed=["pubmedqa_pqal"]) if c["key"] == drawn[0]["key"]
                                  else c for c in analysis["configs_report"]]
    F.render_all(analysis, tmp_path)
    text = (tmp_path / "per_task.svg").read_text()
    assert "answer collapse" in text and "pubmedqa_pqal" in text


def test_screening_mode_draws_selection_half_and_says_so(screening, tmp_path):
    paths = {p.name: p for p in F.render_all(screening, tmp_path)}
    for name, p in paths.items():
        # the claims figure is a placeholder in screening mode: no data marks to require
        text = _check_svg(p, marks=name != "claims.svg")
        assert "selection half" in text, name
    # nothing in a screening JSON can be report-half-derived; the claims figure says why it is empty
    assert "confirmatory" in paths["claims.svg"].read_text()


def test_figures_use_only_selection_half_fields_in_screening(screening, tmp_path):
    """A screening JSON carries no report-half numbers; the figures must render anyway."""
    assert all("report" not in k for k in screening)  # the firewall holds in the input
    F.render_all(screening, tmp_path)
    frontier = (tmp_path / "frontier.svg").read_text()
    assert "screening" in frontier


def test_empty_analysis_still_writes_parseable_svgs(tmp_path):
    for p in F.render_all({"effects_retention_selection": {"effects": {}},
                           "alias_chains_retention_selection": {"effects": {}},
                           "configs_selection": []}, tmp_path):
        _check_svg(p, marks=False)
