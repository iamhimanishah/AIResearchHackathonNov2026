"""`clh figures` — the pitch and report charts (task C4), as self-contained SVGs.

Reads the JSON `clh grid-analysis` writes and draws:

  frontier.svg   headline: protected-set retention change vs legal own-boundary acquisition,
                 one point per configuration; the reference arm, the lower-LR no-mitigation
                 curve ("the curve to beat"), and the Pareto frontier
  effects.svg    forest plot of the technique main effects plus replay x lr_rewarm, then the
                 alias chains under an exploratory heading
  claims.svg     the published-claims tests (PREREG 6.5): delta, interval, verdict, and the
                 same arm against the lower-LR control
  per_task.svg   per-task retention change for the promoted configurations and the controls,
                 with answer-collapse flags (PREREG 8)
  cost.svg       retention gain over no-mitigation vs extra GPU-hours (PREREG 9)

Every chart shows its uncertainty. When the analysis JSON was written without --confirmatory,
the report half was never read, so the charts draw the selection half and say so in their
titles. Plain hand-written SVG — no new dependency. Imports only the standard library, so it
is cheap to call from the page and the agent.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#1d4ed8"
WARN = "#b91c1c"
COLORS = ["#1d4ed8", "#be185d", "#b45309", "#15803d", "#6d28d9", "#0f766e",
          "#374151", "#9f1239", "#0369a1", "#4d7c0f"]
BENCH_LR = 1e-4  # configs/bench/fast.yaml
TASK_SHORT = {"arc_easy": "ARC-E", "hellaswag": "HSwag", "piqa": "PIQA",
              "pubmedqa_pqal": "PubMedQA", "mbpp": "MBPP"}


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo] if math.isfinite(lo) else []
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _nice(lo: float, hi: float, pad: float = 0.08) -> tuple[float, float]:
    m = (hi - lo) * pad or 0.05
    return lo - m, hi + m


class _Axes:
    """Linear x/y scales inside a padded pixel box, with grid lines and tick labels."""

    def __init__(self, xlim: tuple[float, float], ylim: tuple[float, float],
                 box: tuple[float, float, float, float], x_fmt: str = "{:.2f}",
                 y_fmt: str = "{:.1f}"):
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.l, self.t, self.r, self.b = box
        self.x_fmt, self.y_fmt = x_fmt, y_fmt

    def X(self, v: float) -> float:
        return self.l + (v - self.x0) / (self.x1 - self.x0) * (self.r - self.l)

    def Y(self, v: float) -> float:
        return self.t + (1 - (v - self.y0) / (self.y1 - self.y0)) * (self.b - self.t)

    def frame(self, y_ticks: bool = True, x_ticks: bool = True) -> list[str]:
        o = []
        if y_ticks:
            for v in _ticks(self.y0, self.y1):
                o.append(f"<line x1='{self.l}' x2='{self.r}' y1='{self.Y(v):.1f}' "
                         f"y2='{self.Y(v):.1f}' stroke='{GRID}'/>")
                o.append(f"<text x='{self.l - 5}' y='{self.Y(v) + 3.5:.1f}' text-anchor='end' "
                         f"font-size='9' fill='{MUTED}'>{self.y_fmt.format(v)}</text>")
        if x_ticks:
            for v in _ticks(self.x0, self.x1):
                o.append(f"<text x='{self.X(v):.1f}' y='{self.b + 13}' text-anchor='middle' "
                         f"font-size='9' fill='{MUTED}'>{self.x_fmt.format(v)}</text>")
        o.append(f"<rect x='{self.l}' y='{self.t}' width='{self.r - self.l}' "
                 f"height='{self.b - self.t}' fill='none' stroke='{GRID}'/>")
        return o


def _err_h(ax: _Axes, lo: float, hi: float, y_px: float, color: str) -> str:
    """Horizontal 95% interval bar centred at pixel y."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
        return ""
    x0, x1 = ax.X(lo), ax.X(hi)
    return (f"<line x1='{x0:.1f}' x2='{x1:.1f}' y1='{y_px:.1f}' y2='{y_px:.1f}' "
            f"stroke='{color}' stroke-width='1.2'/>"
            f"<line x1='{x0:.1f}' x2='{x0:.1f}' y1='{y_px - 3:.1f}' y2='{y_px + 3:.1f}' stroke='{color}'/>"
            f"<line x1='{x1:.1f}' x2='{x1:.1f}' y1='{y_px - 3:.1f}' y2='{y_px + 3:.1f}' stroke='{color}'/>")


def _err_v(ax: _Axes, lo: float, hi: float, x_px: float, color: str) -> str:
    """Vertical 95% interval bar centred at pixel x."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
        return ""
    y0, y1 = ax.Y(hi), ax.Y(lo)
    return (f"<line x1='{x_px:.1f}' x2='{x_px:.1f}' y1='{y0:.1f}' y2='{y1:.1f}' "
            f"stroke='{color}' stroke-width='1.2'/>"
            f"<line x1='{x_px - 3:.1f}' x2='{x_px + 3:.1f}' y1='{y0:.1f}' y2='{y0:.1f}' stroke='{color}'/>"
            f"<line x1='{x_px - 3:.1f}' x2='{x_px + 3:.1f}' y1='{y1:.1f}' y2='{y1:.1f}' stroke='{color}'/>")


def _dot(ax: _Axes, x: float, y: float, color: str, r: float = 3.2, tip: str = "") -> str:
    tip = f"<title>{_esc(tip)}</title>" if tip else ""
    return f"<circle class='pt' cx='{ax.X(x):.1f}' cy='{ax.Y(y):.1f}' r='{r}' fill='{color}'>{tip}</circle>"


def _svg(width: int, height: int, parts: list[str]) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            f"font-family='ui-monospace, Menlo, monospace' font-size='10' fill='{INK}'>"
            + "".join(parts) + "</svg>\n")


def _title(text: str, x: float = 46) -> str:
    return f"<text x='{x}' y='16' font-size='12' font-weight='bold'>{_esc(text)}</text>"


def _legend(items: list[tuple[str, str]], x: float, y: float) -> str:
    o = []
    for name, color in items:
        o.append(f"<rect x='{x}' y='{y - 7}' width='10' height='3' fill='{color}'/>"
                 f"<text x='{x + 14}' y='{y - 3}' font-size='9' fill='{MUTED}'>{_esc(name)}</text>")
        x += 26 + 6.0 * len(name)
    return "".join(o)


def _empty(title: str, note: str, w: int = 940) -> str:
    return _svg(w, 120, [_title(title), f"<text x='46' y='60' fill='{MUTED}'>{_esc(note)}</text>"])


# ---------------------------------------------------------------------------------------------
# shared reads of the analysis JSON


def _configs(analysis: dict[str, Any], half: str) -> list[dict[str, Any]]:
    return analysis.get(f"configs_{half}", [])


def _is_reference(c: dict[str, Any]) -> bool:
    return c["techniques"] == ["lr_rewarm", "replay"] and c["selection"] is None and c["lr"] is None


def _lr(c: dict[str, Any]) -> float:
    return c["lr"] if c["lr"] is not None else BENCH_LR


def _finite(*vals: float | None) -> bool:
    return all(v is not None and math.isfinite(v) for v in vals)


# ---------------------------------------------------------------------------------------------
# the five figures


def frontier(analysis: dict[str, Any], half: str, tag: str) -> str:
    """The headline chart: retention change (higher better) vs legal own-boundary acquisition
    (lower better), one point per configuration, 95% bars both directions."""
    pts = []
    for c in _configs(analysis, half):
        a = c["acquisition"].get("legal")
        r = c["retention"]
        if a is None or not _finite(a["mean"], a["lo"], a["hi"], r["mean"], r["lo"], r["hi"]):
            continue
        pts.append((c, a, r))
    w, h = 940, 560
    if not pts:
        return _empty(f"frontier — {tag}", "no configurations with legal acquisition")
    xs = [v for _, a, _ in pts for v in (a["lo"], a["hi"])]
    ys = [100 * v for _, _, r in pts for v in (r["lo"], r["hi"])]
    ax = _Axes(_nice(min(xs), max(xs)), _nice(min(ys), max(ys)), (70, 36, w - 16, h - 90),
               y_fmt="{:+.1f}")
    o = [_title(f"frontier — protected-set retention vs legal acquisition, {tag}")] + ax.frame()
    if ax.y0 < 0 < ax.y1:
        o.append(f"<line x1='{ax.l}' x2='{ax.r}' y1='{ax.Y(0):.1f}' y2='{ax.Y(0):.1f}' "
                 f"stroke='{MUTED}' stroke-dasharray='3 3'/>")

    # the curve to beat: no-mitigation at 1e-4, 5e-5 and 3e-5 connected in LR order
    lr_pts = sorted(((c, a, r) for c, a, r in pts if not c["techniques"]), key=lambda t: -_lr(t[0]))
    if len(lr_pts) >= 2:
        d = " ".join(f"{'M' if i == 0 else 'L'}{ax.X(a['mean']):.1f},{ax.Y(100 * r['mean']):.1f}"
                     for i, (_, a, r) in enumerate(lr_pts))
        o.append(f"<path d='{d}' fill='none' stroke='{WARN}' stroke-width='2'/>")

    # Pareto frontier (exploratory view of the same two axes)
    pareto_names = set(analysis.get(f"pareto_{half}", []))
    front = sorted((a["mean"], 100 * r["mean"]) for c, a, r in pts if c["config"] in pareto_names)
    if len(front) >= 2:
        d = " ".join(f"{'M' if i == 0 else 'L'}{ax.X(x):.1f},{ax.Y(y):.1f}"
                     for i, (x, y) in enumerate(front))
        o.append(f"<path d='{d}' fill='none' stroke='{MUTED}' stroke-width='1.2' stroke-dasharray='5 3'/>")

    for c, a, r in pts:
        color = ACCENT if _is_reference(c) else WARN if not c["techniques"] else "#9ca3af"
        o.append(_err_h(ax, a["lo"], a["hi"], ax.Y(100 * r["mean"]), color))
        o.append(_err_v(ax, 100 * r["lo"], 100 * r["hi"], ax.X(a["mean"]), color))
        o.append(_dot(ax, a["mean"], 100 * r["mean"], color, 4.2 if _is_reference(c) else 3,
                      f"{c['config']}: ret {100 * r['mean']:+.1f} pts, acq {a['mean']:.3f}"))
        if _is_reference(c) or not c["techniques"]:
            name = "reference arm" if _is_reference(c) else f"no-mitigation lr{_lr(c):g}"
            o.append(f"<text x='{ax.X(a['mean']) + 7:.1f}' y='{ax.Y(100 * r['mean']) - 5:.1f}' "
                     f"font-size='9' fill='{INK}'>{_esc(name)}</text>")
    o.append(_legend([("reference arm", ACCENT), ("no-mitigation at lower LR (the curve to beat)", WARN),
                      ("other configurations", "#9ca3af"), ("Pareto frontier", MUTED)], ax.l, h - 12))
    o.append(f"<text x='{(ax.l + ax.r) / 2:.0f}' y='{h - 34}' text-anchor='middle' font-size='10' "
             f"fill='{MUTED}'>legal own-boundary perplexity ratio (lower = learns more) &#8594;</text>")
    o.append(f"<text transform='translate(16,{(ax.t + ax.b) / 2:.0f}) rotate(-90)' text-anchor='middle' "
             f"font-size='10' fill='{MUTED}'>{'report' if half == 'report' else 'selection'}-half "
             "mean delta, points &#8594;</text>")
    return _svg(w, h, o)


def effects(analysis: dict[str, Any], half: str, tag: str) -> str:
    """Forest plot: the six main effects plus replay x lr_rewarm (the main fit), then each
    alias chain of the factorial under an exploratory heading. The fit always reads the
    selection half, whatever the mode."""
    fit = analysis.get("effects_retention_selection", {}).get("effects", {})
    chains = {k: v for k, v in analysis.get("alias_chains_retention_selection", {})
              .get("effects", {}).items() if "×" in k}
    mains = sorted(((n, e) for n, e in fit.items() if n != "(baseline)"),
                   key=lambda t: -t[1]["estimate"])
    chain_rows = [((" = ".join([n, *e["aliased_with"]])), e) for n, e in sorted(chains.items())]
    if not mains and not chain_rows:
        return _empty(f"technique effects — selection half ({tag})", "no fitted effects")
    lo = min(e["lo"] for _, e in [*mains, *chain_rows]) * 100
    hi = max(e["hi"] for _, e in [*mains, *chain_rows]) * 100
    x0, x1 = _nice(lo, hi, 0.12)
    w = 940
    row_h, gap = 24, 34
    h = 70 + gap + row_h * (len(mains) + len(chain_rows)) + 50
    ax = _Axes((x0, x1), (0, 1), (280, 44, w - 30, h - 60), x_fmt="{:+.1f}")
    o = [_title(f"technique effects on protected-set retention — selection half ({tag})")]
    o += ax.frame(y_ticks=False)
    if ax.x0 < 0 < ax.x1:
        o.append(f"<line x1='{ax.X(0):.1f}' x2='{ax.X(0):.1f}' y1='{ax.t}' y2='{ax.b}' "
                 f"stroke='{MUTED}' stroke-dasharray='3 3'/>")
    y = ax.t + 14

    def bar(e: dict[str, Any], yp: float, color: str) -> str:
        return (_err_h(ax, e["lo"] * 100, e["hi"] * 100, yp, color)
                + f"<circle class='pt' cx='{ax.X(e['estimate'] * 100):.1f}' cy='{yp}' r='3.4' "
                  f"fill='{color}'/>")

    for name, e in mains:
        alias = f"  = {' = '.join(e['aliased_with'])}" if e.get("aliased_with") else ""
        o.append(f"<text x='{ax.l - 8}' y='{y + 3}' text-anchor='end' font-size='9.5'>"
                 f"{_esc(name + alias)}</text>")
        o.append(bar(e, y, ACCENT))
        y += row_h
    y += gap - 8
    o.append(f"<text x='{ax.l - 8}' y='{y}' text-anchor='end' font-size='9.5' font-weight='bold' "
             f"fill='{WARN}'>alias chains (exploratory)</text>")
    o.append(f"<text x='{ax.l}' y='{y}' font-size='9' fill='{MUTED}'>"
             "each estimate is the sum of the pairs in its chain</text>")
    y += 16
    for name, e in chain_rows:
        o.append(f"<text x='{ax.l - 8}' y='{y + 3}' text-anchor='end' font-size='9' "
                 f"fill='{MUTED}'>{_esc(name)}</text>")
        o.append(bar(e, y, "#b45309"))
        y += row_h
    o.append(f"<text x='{ax.l}' y='{h - 16}' font-size='9' fill='{MUTED}'>"
             "effect on the mean protected-set delta, points (95% intervals; least squares on the "
             "screening design)</text>")
    return _svg(w, h, o)


def claims(analysis: dict[str, Any], half: str, tag: str) -> str:
    """The published-claims tests (PREREG 6.5), one row per test: delta with its interval, a
    zero line, the verdict, and a second marker for the arm against the lower-LR control."""
    w = 1000
    if "claims" not in analysis:
        return _empty(f"published claims — {tag}",
                      "the claims tests read the report half; run grid-analysis --confirmatory")
    rows = analysis["claims"]
    by_id: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    n_lines = len(rows) + 2 * len(by_id)
    h = 60 + 20 * n_lines + 40
    xs = [100 * r[k] for r in rows if r["outcome"] == "retention" for k in ("lo", "hi")
          if _finite(r.get(k))]
    xs += [100 * r["vs_lr_control"][k] for r in rows if r.get("vs_lr_control")
           for k in ("lo", "hi") if _finite(r["vs_lr_control"].get(k))]
    x0, x1 = _nice(min(xs + [0]), max(xs + [0]), 0.1)
    ax = _Axes((x0, x1), (0, 1), (330, 30, w - 230, h - 44), x_fmt="{:+.0f}")
    o = [_title(f"published claims on this bench — {tag} (exploratory, PREREG 6.5)")]
    o += ax.frame(y_ticks=False)
    if ax.x0 < 0 < ax.x1:
        o.append(f"<line x1='{ax.X(0):.1f}' x2='{ax.X(0):.1f}' y1='{ax.t}' y2='{ax.b}' "
                 f"stroke='{MUTED}' stroke-dasharray='3 3'/>")
    y = ax.t + 6
    VERDICT_COLOR = {"reproduced": "#15803d", "opposite": WARN, "reported": MUTED}
    for cid in sorted(by_id):
        tests = by_id[cid]
        o.append(f"<text x='{ax.l - 8}' y='{y + 6}' text-anchor='end' font-size='9.5' "
                 f"font-weight='bold'>#{cid} {_esc(tests[0]['claim'][:56])}</text>")
        o.append(f"<text x='{ax.r + 8}' y='{y + 6}' font-size='8.5' fill='{MUTED}'>"
                 f"{_esc(tests[0]['source'])}</text>")
        y += 18
        for r in tests:
            label = f"{r['arm']} vs {r['comparator']} ({r['outcome']})"
            o.append(f"<text x='{ax.l - 8}' y='{y + 3}' text-anchor='end' font-size='9'>"
                     f"{_esc(label)}</text>")
            if r["outcome"] == "retention" and _finite(r.get("delta"), r.get("lo"), r.get("hi")):
                o.append(_err_h(ax, r["lo"] * 100, r["hi"] * 100, y, INK))
                o.append(f"<circle class='pt' cx='{ax.X(r['delta'] * 100):.1f}' cy='{y}' r='3.2' "
                         f"fill='{INK}'/>")
                lr = r.get("vs_lr_control")
                if lr and _finite(lr.get("delta"), lr.get("lo"), lr.get("hi")):
                    o.append(_err_h(ax, lr["lo"] * 100, lr["hi"] * 100, y + 6, ACCENT))
                    o.append(f"<circle class='pt' cx='{ax.X(lr['delta'] * 100):.1f}' cy='{y + 6}' "
                             f"r='2.6' fill='none' stroke='{ACCENT}' stroke-width='1.4'>"
                             f"<title>vs {_esc(lr['config'])}: {100 * lr['delta']:+.2f} pts</title></circle>")
            else:
                v = r.get("delta")
                txt = "no shared seeds" if not _finite(v) else (
                    f"{v:+.3f} (log ratio)" if r["outcome"] != "retention" else f"{100 * v:+.1f} pts")
                o.append(f"<text x='{ax.X(0) + 6:.1f}' y='{y + 3}' font-size='9' "
                         f"fill='{MUTED}'>{_esc(txt)}</text>")
            o.append(f"<text x='{ax.r + 8}' y='{y + 3}' font-size='9' "
                     f"fill='{VERDICT_COLOR.get(r['verdict'], WARN)}'>{_esc(r['verdict'])}</text>")
            y += 20
        y += 10
    o.append(f"<text x='{ax.l}' y='{h - 12}' font-size='9' fill='{MUTED}'>"
             "filled marker: arm vs comparator (retention deltas in points). Open marker: the same "
             "arm vs the lower-LR no-mitigation control. Acquisition claims are log ratios.</text>")
    return _svg(w, h, o)


def per_task(analysis: dict[str, Any], half: str, tag: str) -> str:
    """Per-task protected-set change for the promoted configurations and the controls. A red
    ring marks a task whose answer distribution collapsed (answer share > 0.90, PREREG 8)."""
    promoted = {d["key"] for d in analysis.get("promotion", []) if d.get("promoted")}
    promoted |= {d["key"] for d in analysis.get("selection_screen", []) if d.get("promoted")}
    chosen = [c for c in _configs(analysis, half)
              if c["key"] in promoted or not c["techniques"] or _is_reference(c)]
    if not chosen:
        return _empty(f"per-task — {tag}", "no promoted configurations yet")
    tasks = sorted({t for c in chosen for t in c["per_task"]})
    w = 960
    plot_bottom = 290
    h = plot_bottom + 44 + 16 * len(chosen)
    ys = [100 * v for c in chosen for d in c["per_task"].values() for v in (d["lo"], d["hi"])
          if _finite(v)]
    ax = _Axes((-0.5, len(tasks) - 0.5), _nice(min(ys + [0]), max(ys + [0]), 0.1),
               (70, 40, w - 20, plot_bottom), x_fmt="{:.0f}", y_fmt="{:+.1f}")
    o = [_title(f"per-task protected-set change — promoted configurations and controls, {tag}")]
    o += ax.frame(x_ticks=False)
    if ax.y0 < 0 < ax.y1:
        o.append(f"<line x1='{ax.l}' x2='{ax.r}' y1='{ax.Y(0):.1f}' y2='{ax.Y(0):.1f}' "
                 f"stroke='{MUTED}' stroke-dasharray='3 3'/>")
    span = 0.7
    for i, c in enumerate(chosen):
        color = COLORS[i % len(COLORS)]
        for ti, t in enumerate(tasks):
            if t not in c["per_task"]:
                continue
            x = ti + (i + 0.5 - len(chosen) / 2) * (span / max(len(chosen), 1))
            pt = c["per_task"][t]
            d = 100 * pt["mean"]
            x_px = ax.X(x)
            if _finite(pt["lo"], pt["hi"]):
                o.append(_err_v(ax, 100 * pt["lo"], 100 * pt["hi"], x_px, color))
            if t in c["collapsed"]:
                o.append(f"<circle class='pt' cx='{ax.X(x):.1f}' cy='{ax.Y(d):.1f}' r='6' "
                         f"fill='none' stroke='{WARN}' stroke-width='1.6'>"
                         f"<title>answer collapse on {_esc(t)}</title></circle>")
            o.append(_dot(ax, x, d, color, 3.4, f"{c['config']} {t}: {d:+.1f} pts"))
    for ti, t in enumerate(tasks):
        o.append(f"<text x='{ax.X(ti):.1f}' y='{ax.b + 14}' text-anchor='middle' font-size='9.5'>"
                 f"{_esc(TASK_SHORT.get(t, t))}</text>")
    for i, c in enumerate(chosen):
        o.append(f"<rect x='{ax.l}' y='{ax.b + 26 + i * 16}' width='10' height='3' "
                 f"fill='{COLORS[i % len(COLORS)]}'/>"
                 f"<text x='{ax.l + 16}' y='{ax.b + 30 + i * 16}' font-size='9'>"
                 f"{_esc(c['config'])}{' (answer collapse)' if c['collapsed'] else ''}</text>")
    o.append(f"<text x='{ax.l}' y='{h - 6}' font-size='9' fill='{MUTED}'>"
             "bars: 95% interval from resampling seeds</text>")
    return _svg(w, h, o)


def cost(analysis: dict[str, Any], half: str, tag: str) -> str:
    """Retention gain over no-mitigation against extra forward tokens, in millions (PREREG 9,
    Amendment 4: forward tokens are measured, a wall-clock difference of short runs is noise).
    The dot is the paired delta vs no-mitigation and the bar is its paired 95% interval, so
    both come from one estimator (red-team RT-26). A configuration with no paired seed falls
    back to the unpaired mean gain from the cost table and has no bar."""
    rows = analysis.get(f"cost_{half}", [])
    by_label = {c["config"]: c for c in _configs(analysis, half)}

    def gain(r: dict[str, Any]) -> float:
        c = by_label.get(r["config"])
        return c["vs_base"]["delta"] if c and c.get("vs_base") else r["retention_gain"]

    pts = [r for r in rows if _finite(gain(r), r["extra_forward_tokens"])]
    w, h = 940, 520
    if not pts:
        return _empty(f"cost — {tag}", "no cost rows")
    xs = [r["extra_forward_tokens"] / 1e6 for r in pts]
    ys = [100 * gain(r) for r in pts]
    ax = _Axes(_nice(min(xs + [0]), max(xs + [0])), _nice(min(ys + [0]), max(ys + [0]), 0.12),
               (70, 36, w - 16, h - 70), x_fmt="{:.0f}", y_fmt="{:+.1f}")
    o = [_title(f"retention gain vs extra forward tokens — {tag}")] + ax.frame()
    if ax.x0 < 0 < ax.x1:
        o.append(f"<line x1='{ax.X(0):.1f}' x2='{ax.X(0):.1f}' y1='{ax.t}' y2='{ax.b}' "
                 f"stroke='{MUTED}' stroke-dasharray='3 3'/>")
    if ax.y0 < 0 < ax.y1:
        o.append(f"<line x1='{ax.l}' x2='{ax.r}' y1='{ax.Y(0):.1f}' y2='{ax.Y(0):.1f}' "
                 f"stroke='{MUTED}' stroke-dasharray='3 3'/>")
    for i, r in enumerate(pts):
        color = COLORS[i % len(COLORS)]
        x, y = r["extra_forward_tokens"] / 1e6, 100 * gain(r)
        c = by_label.get(r["config"])
        if c and c.get("vs_base") and _finite(c["vs_base"]["lo"], c["vs_base"]["hi"]):
            o.append(_err_v(ax, 100 * c["vs_base"]["lo"], 100 * c["vs_base"]["hi"], ax.X(x), color))
        o.append(_dot(ax, x, y, color, 3.6,
                      f"{r['config']}: {y:+.1f} pts for {x:.0f}M extra forward tokens"))
        o.append(f"<text x='{ax.X(x) + 6:.1f}' y='{ax.Y(y) - 4:.1f}' font-size='8.5' "
                 f"fill='{MUTED}'>{_esc(r['config'][:34])}</text>")
    o.append(f"<text x='{(ax.l + ax.r) / 2:.0f}' y='{h - 26}' text-anchor='middle' font-size='10' "
             f"fill='{MUTED}'>extra forward tokens vs no-mitigation, millions &#8594;</text>")
    o.append(f"<text x='{ax.l}' y='{h - 10}' font-size='9' fill='{MUTED}'>"
             "paired retention gain over no-mitigation, points (bars: its 95% paired interval)</text>")
    return _svg(w, h, o)


FIGURES = {"frontier": frontier, "effects": effects, "claims": claims,
           "per_task": per_task, "cost": cost}


def render_all(analysis: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write every figure to `out_dir`. In screening mode (no `confirmatory` key) the report
    half was never read, so every chart draws the selection half and its title says so."""
    confirmatory = "confirmatory" in analysis
    half = "report" if confirmatory else "selection"
    tag = "report half, confirmatory" if confirmatory else "selection half, screening"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in FIGURES.items():
        p = out / f"{name}.svg"
        p.write_text(fn(analysis, half, tag))
        written.append(p)
    return written
