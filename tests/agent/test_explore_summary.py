"""The research-agent summary page: numbers come from the records, prose from the themes file."""

import json

from clharness.agent import explore, explore_summary
from clharness.config import REPO_ROOT


def _hyp(id_: str, **over):
    base = {"id": id_, "created_at": "2026-11-15T00:00:00+00:00", "author": "model", "origin": "follow_up",
                "question": f"question {id_}?", "claim": "claim", "control": "no-mitigation",
                "prediction": {"direction": "less_forgetting", "size_points": 1.0}, "status": "confirmed",
                "effect_points": 1.5, "threshold_points": 1.2, "acquisition_ok": True}
    return explore.Hypothesis.model_validate({**base, **over})


def test_render_groups_by_theme_and_relabels_costly_confirmations():
    hyps = [_hyp("h001"), _hyp("h002", acquisition_ok=False), _hyp("h003", status="inconclusive", effect_points=0.1)]
    inst = {"name": "x", "hyps": hyps, "rows": [], "model_usd": 0.5, "model_calls": 2, "calls_logged": True}
    themes = [{"title": "A theme", "text": "Words only.", "hypotheses": ["x:h002", "x:h999"]}]
    page = explore_summary.render([inst], [], themes, ["some/file.jsonl"])
    assert "| x | 3 | 1 | 1 | 0 | 1 | 0 | 0 |" in page  # h002 counts as confirmed with cost
    theme, rest = page.split("## Other decided hypotheses")
    assert "### A theme" in theme and "x h002" in theme.split("### A theme")[1]
    assert "**confirmed with cost**" in theme and "+1.5 (needs 1.2)" in theme
    other = rest.split("## Techniques")[0]
    assert "x h001" in other and "x h003" not in other and "x h002" not in other
    assert "x h003" in page.split("## Every hypothesis")[1]


def test_themes_file_holds_no_numbers_and_names_real_hypotheses():
    themes = json.loads((REPO_ROOT / explore_summary.THEMES).read_text())["themes"]
    known = {(f.name.removeprefix("explore-"), h.id)
             for f in (REPO_ROOT / "results/agent").glob("explore-*") if (f / "hypotheses.jsonl").exists()
             for h in explore.load_hypotheses(f / "hypotheses.jsonl")}
    for theme in themes:
        body = theme["text"]
        assert not any(ch.isdigit() for ch in body), f"a number is typed into theme {theme['title']!r}"
        for ref in theme["hypotheses"]:
            assert tuple(ref.split(":", 1)) in known, ref
