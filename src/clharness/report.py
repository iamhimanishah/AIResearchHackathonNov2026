"""`clh report RUN` — one self-contained HTML page per run, built from analysis.json, items.json
and metrics.jsonl. Inline SVG (no JS libraries, no network), so it opens anywhere.

Sections: headline → eval curves by domain → "the dashboard said fine" (loss vs generation,
degenerate answers) → what was forgotten (item gallery) → recoverable vs genuine → instrument
health → features lost/grown. The demo page for the hackathon is this report with framing.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from clharness.config import REPO_ROOT
from clharness.findings import FINDINGS_CSS, findings, render as render_findings

MC = {"arc_easy/acc_norm": "ARC-Easy", "hellaswag/acc_norm": "HellaSwag", "piqa/acc_norm": "PIQA", "pubmedqa_pqal/acc": "PubMedQA"}
GEN = {"mbpp/pass_at_1": "MBPP (code gen)"}
TASK_LABEL = {**{k.split("/")[0]: v for k, v in MC.items()}, "mbpp": "MBPP (code gen)"}
DOMAIN_COLORS = {"code": "#dbeafe", "biomedical": "#fce7f3", "legal": "#fef3c7", "math": "#dcfce7"}
SERIES_COLORS = ["#1d4ed8", "#be185d", "#b45309", "#15803d", "#6d28d9", "#0f766e", "#374151"]
DEGENERATE_SHARE = 0.90


# ---------------------------------------------------------------- tiny SVG plotting

def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


class Chart:
    """Minimal line chart: x in tokens, several series, shaded domain bands, optional threshold."""

    def __init__(self, width=720, height=280, pad=(44, 12, 46, 40), y_range=None, y_label="", title="", y_fmt="{:.2f}"):
        self.w, self.h = width, height
        self.pl, self.pr, self.pb, self.pt = pad
        self.y_range = y_range
        self.y_label, self.title, self.y_fmt = y_label, title, y_fmt
        self.series: list[tuple[str, list[float], list[float], str, bool]] = []
        self.bands: list[tuple[float, float, str, str]] = []
        self.hlines: list[tuple[float, str, str]] = []
        self.vmarks: list[tuple[float, str]] = []

    def add(self, name, xs, ys, color=None, dashed=False):
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None and y == y]
        if pts:
            self.series.append((name, [p[0] for p in pts], [p[1] for p in pts], color or SERIES_COLORS[len(self.series) % len(SERIES_COLORS)], dashed))

    def band(self, x0, x1, color, label):
        self.bands.append((x0, x1, color, label))

    def hline(self, y, color, label):
        self.hlines.append((y, color, label))

    def vmark(self, x, label):
        self.vmarks.append((x, label))

    def render(self) -> str:
        xs_all = [x for _, xs, _, _, _ in self.series for x in xs] + [b[1] for b in self.bands]
        ys_all = [y for _, _, ys, _, _ in self.series for y in ys] + [h[0] for h in self.hlines]
        if not xs_all or not ys_all:
            return "<div class='empty'>no data</div>"
        x0, x1 = 0.0, max(xs_all) or 1.0
        if self.y_range:
            y0, y1 = self.y_range
        else:
            lo, hi = min(ys_all), max(ys_all)
            m = (hi - lo) * 0.08 or 0.05
            y0, y1 = lo - m, hi + m
        iw, ih = self.w - self.pl - self.pr, self.h - self.pt - self.pb
        X = lambda x: self.pl + (x - x0) / (x1 - x0) * iw  # noqa: E731
        Y = lambda y: self.pt + (1 - (y - y0) / (y1 - y0)) * ih  # noqa: E731
        o = [f"<svg viewBox='0 0 {self.w} {self.h}' class='chart' role='img'>"]
        if self.title:
            o.append(f"<text x='{self.pl}' y='{self.pt - 2}' class='ctitle'>{_esc(self.title)}</text>")
        for bx0, bx1, color, label in self.bands:
            o.append(f"<rect x='{X(bx0):.1f}' y='{self.pt}' width='{X(bx1) - X(bx0):.1f}' height='{ih}' fill='{color}'/>")
            o.append(f"<text x='{(X(bx0) + X(bx1)) / 2:.1f}' y='{self.pt + 12}' class='band' text-anchor='middle'>{_esc(label)}</text>")
        for t in np.linspace(y0, y1, 5):
            o.append(f"<line x1='{self.pl}' x2='{self.w - self.pr}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' class='grid'/>")
            o.append(f"<text x='{self.pl - 4}' y='{Y(t) + 4:.1f}' class='tick' text-anchor='end'>{self.y_fmt.format(t)}</text>")
        for t in np.linspace(x0, x1, 5):
            o.append(f"<text x='{X(t):.1f}' y='{self.h - self.pb + 14}' class='tick' text-anchor='middle'>{t / 1e6:.0f}M</text>")
        o.append(f"<text x='{self.w - self.pr}' y='{self.h - self.pb + 14}' class='tick' text-anchor='end'>tokens</text>")
        if self.y_label:
            o.append(f"<text transform='translate(12,{self.pt + ih / 2:.1f}) rotate(-90)' class='tick' text-anchor='middle'>{_esc(self.y_label)}</text>")
        for y, color, label in self.hlines:
            o.append(f"<line x1='{self.pl}' x2='{self.w - self.pr}' y1='{Y(y):.1f}' y2='{Y(y):.1f}' stroke='{color}' stroke-dasharray='4 3'/>")
            o.append(f"<text x='{self.w - self.pr - 2}' y='{Y(y) - 3:.1f}' class='band' text-anchor='end' fill='{color}'>{_esc(label)}</text>")
        for x, label in self.vmarks:
            o.append(f"<line x1='{X(x):.1f}' x2='{X(x):.1f}' y1='{self.pt}' y2='{self.pt + ih}' stroke='#9ca3af' stroke-dasharray='2 3'/>")
        for name, xs, ys, color, dashed in self.series:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
            o.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'{' stroke-dasharray=\"6 3\"' if dashed else ''}/>")
            if len(xs) <= 200:  # hover markers only for checkpoint-level series; dense series stay a bare path
                for x, y in zip(xs, ys):
                    o.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='2.2' fill='{color}'><title>{_esc(name)}: {self.y_fmt.format(y)} @ {x / 1e6:.0f}M</title></circle>")
        lx = self.pl
        ly = self.h - 4
        for name, _, _, color, _ in self.series:
            o.append(f"<rect x='{lx}' y='{ly - 8}' width='10' height='3' fill='{color}'/>")
            o.append(f"<text x='{lx + 14}' y='{ly - 4}' class='legend'>{_esc(name)}</text>")
            lx += 22 + int(6.2 * len(name))
        o.append("</svg>")
        return "".join(o)


def _strip(correct: str) -> str:
    """Correctness vector as a row of small squares (1 green, 0 red, - grey)."""
    cells = "".join(f"<i class='c{c if c in '10' else 'x'}'></i>" for c in correct)
    return f"<span class='strip' title='{_esc(correct)}'>{cells}</span>"


# ---------------------------------------------------------------- sections

def _domain_bands(a: dict[str, Any]) -> list[tuple[float, float, str]]:
    names = [d["name"] for d in a["config"]["data"]["domains"]]
    ck = a["checkpoints"]
    bands = []
    for di, name in enumerate(names):
        xs = [c["gradient_tokens"] for c in ck if c["domain_index"] == di]
        prev = [c["gradient_tokens"] for c in ck if c["domain_index"] < di] or [0]
        if xs:
            bands.append((max(prev), max(xs), name))
    return bands


def headline(a: dict[str, Any], items: dict[str, Any] | None) -> str:
    r = a["retention"]
    cfg = a["config"]
    end = a["checkpoints"][-1]
    ins = end["instrument"]["13"]
    valid = all(c["instrument"]["13"].get("valid", 1) for c in a["checkpoints"])
    alarms = [(c["index"], t) for c in a["checkpoints"] for t, s in c.get("answers", {}).items() if s and s.get("degenerate")]
    alarm_tasks = sorted({t for _, t in alarms})
    n_lost = sum(t["counts"].get("lost", 0) for t in items["tasks"].values()) if items else None
    cards = [
        ("Protected-set accuracy", f"{r['mc_mean_delta_pts']:+.1f} pts", "mean of 4 multiple-choice evals, end vs base"),
        ("General-text perplexity", f"{(r['ppl_doc_end_over_base'].get('general', float('nan')) - 1) * 100:+.0f}%", "per-document, held-out web text"),
        ("Questions lost", f"{n_lost}" if n_lost is not None else "—", "right at base, wrong at the end and the 2 before it"),
        ("Instrument", "valid" if valid else "TRIPPED", f"SAE FVU drift {ins.get('fvu_delta', 0):+.3f} (limit {cfg['model'].get('sae_fvu_invalid_delta', 0.1)})"),
        ("Degenerate answers", f"{len(alarm_tasks)} task(s)" if alarms else "none", ", ".join(TASK_LABEL.get(t, t) for t in alarm_tasks) or "no eval collapsed to one option"),
    ]
    per = " · ".join(f"{(MC | GEN).get(t, t)} {v:+.1f}" for t, v in r["per_task_delta_pts"].items())
    doms = " → ".join(d["name"] for d in cfg["data"]["domains"])
    o = [f"<p class='sub'>{_esc(cfg['model']['model_id'])} · lr {cfg['optim']['lr']:g} · {doms} · "
         f"{a['gradient_tokens'] / 1e6:.0f}M gradient tokens · {a['n_checkpoints']} checkpoints · {a.get('anomalous_grad_steps', 0)} anomalous steps</p>"]
    o.append("<div class='cards'>" + "".join(f"<div class='card'><div class='k'>{_esc(k)}</div><div class='v'>{_esc(v)}</div><div class='d'>{_esc(d)}</div></div>" for k, v, d in cards) + "</div>")
    o.append(f"<p class='sub'>Per task, end vs base (points): {per}</p>")
    return "".join(o)


def eval_curves(a: dict[str, Any]) -> str:
    ck = a["checkpoints"]
    xs = [c["gradient_tokens"] for c in ck]
    bands = _domain_bands(a)
    ch = Chart(title="Protected-set accuracy through the four domains", y_label="accuracy")
    for b in bands:
        ch.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
    for t, label in MC.items():
        ch.add(label, xs, [c["scores"].get(t) for c in ck])
    g = Chart(title="Code generation (MBPP pass@1) — measured every few checkpoints", y_label="pass@1", height=240)
    for b in bands:
        g.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
    for t, label in GEN.items():
        g.add(label, xs, [c["scores"].get(t) for c in ck], color="#374151")
    p = Chart(title="Held-out perplexity per domain (per document; lower is better)", y_label="perplexity", y_fmt="{:.1f}")
    for b in bands:
        p.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
    for i, d in enumerate(["general", "code", "biomedical", "legal", "math"]):
        p.add(d, xs, [c["ppl_doc"].get(d) for c in ck])
    return (f"<p>Each domain is trained on in turn (shaded). The four multiple-choice evals are the protected set: capabilities "
            f"the model had at the start and was never trained on again.</p>{ch.render()}{g.render()}{p.render()}")


def dashboard_said_fine(a: dict[str, Any], rows: list[dict]) -> str:
    ck = a["checkpoints"]
    bands = _domain_bands(a)
    tr = [(r["train/gradient_tokens"], r["train/loss"]) for r in rows if "train/loss" in r]
    # smooth the loss a little for readability
    o = []
    if tr:
        ch = Chart(title="Training loss on the current domain vs. code generation", y_label="train loss", height=240)
        for b in bands:
            ch.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
        xs, ys = zip(*tr)
        k = max(1, len(ys) // 200)
        sm = [float(np.mean(ys[max(0, i - k): i + 1])) for i in range(len(ys))]
        stride = max(1, len(xs) // 600)  # ~600 points is plenty for a 720px-wide chart
        ch.add("train loss (smoothed)", list(xs)[::stride], sm[::stride], color="#374151")
        o.append(ch.render())
        mb = [(c["gradient_tokens"], c["scores"].get("mbpp/pass_at_1")) for c in ck if c["scores"].get("mbpp/pass_at_1") is not None]
        if len(mb) >= 2:
            o.append(f"<p class='note'>During the code domain the training loss fell steadily while MBPP pass@1 went "
                     f"{mb[0][1]:.3f} → {mb[1][1]:.3f}. The loss curve is the wrong instrument for the question "
                     f"\"did the model get better at code?\"</p>")
    # degenerate-answer alarm
    tasks = [t for t in TASK_LABEL if any(c.get("answers", {}).get(t) for c in ck)]
    if tasks:
        d = Chart(title="Share of items answered with the model's single most common option", y_label="share", y_range=(0, 1.02), height=240)
        for b in bands:
            d.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
        d.hline(DEGENERATE_SHARE, "#b91c1c", "alarm")
        for t in tasks:
            pts = [(c["gradient_tokens"], c["answers"][t]["share_max"]) for c in ck if c.get("answers", {}).get(t)]
            d.add(TASK_LABEL[t], [p[0] for p in pts], [p[1] for p in pts])
        o.append(d.render())
        o.append("<p class='note'>A benchmark whose score sits at the majority-class rate while nearly every answer is the same "
                 "option has stopped reading the question. Accuracy alone does not show it; the answer distribution does. "
                 "(Only checkpoints with recorded answers are plotted.)</p>")
    return "".join(o) or "<p class='empty'>no training-loss rows</p>"


def gallery(items: dict[str, Any], per_task: int = 6, tasks: tuple[str, ...] | None = None) -> str:
    o = []
    doms = items["checkpoint_domain"]
    for task, tk in items["tasks"].items():
        if tasks and task not in tasks:
            continue
        c = tk["counts"]
        lost = [e for e in tk["items"] if e["category"] == "lost"]
        lost.sort(key=lambda e: -e["n_wrong"])  # most decisively lost first
        # one from each causing domain first, then the rest, so the gallery is not all "lost during code"
        seen, ordered = set(), []
        for e in lost:
            if e["first_loss_domain"] not in seen:
                seen.add(e["first_loss_domain"]); ordered.append(e)
        ordered += [e for e in lost if e not in ordered]
        lost = ordered
        by_dom = ", ".join(f"{d}: {n}" for d, n in sorted(tk["lost_by_domain"].items(), key=lambda x: -x[1]))
        o.append(f"<h3>{_esc(TASK_LABEL.get(task, task))} <span class='muted'>— base {tk['acc_base']:.3f} → end {tk['acc_end']:.3f}; "
                 f"lost {c.get('lost', 0)}, recovered {c.get('recovered', 0)}, gained {c.get('gained', 0)}, unstable {c.get('unstable', 0)} of {tk['n_items']}"
                 f"{'; first lost during ' + by_dom if by_dom else ''}</span></h3>")
        if not lost:
            o.append("<p class='empty'>nothing persistently lost</p>")
            continue
        o.append("<div class='gallery'>")
        for e in lost[:per_task]:
            q = e.get("question", "")
            choices = e.get("choices")
            gold = e.get("gold")
            chosen = e.get("chosen") or {}
            ans = ""
            if choices is not None and gold is not None and 0 <= gold < len(choices):
                ans = f"<div class='ans'><b>correct:</b> {_esc(choices[gold])}</div>"
                if chosen:
                    last_k = max(chosen, key=lambda k: int(k))
                    ci = chosen[last_k]
                    if 0 <= ci < len(choices) and ci != gold:
                        ans += f"<div class='ans wrong'><b>model at the end:</b> {_esc(choices[ci])}</div>"
            when = f"lost at checkpoint {e['first_loss_ckpt']} ({e['first_loss_domain']})" if e["first_loss_ckpt"] is not None else ""
            o.append(f"<div class='item'><div class='q'>{_esc(q)}</div>{ans}<div class='meta'>{_esc(when)} · {_strip(e['correct'])}</div></div>")
        o.append("</div>")
    o.append(f"<p class='note'>Strip = correctness at each of the {len(doms)} checkpoints (green right, red wrong). "
             "\"Lost\" = right at base, wrong at the end and at the two checkpoints before it. Showing the earliest-lost items.</p>")
    return "".join(o)


def recoverability(a: dict[str, Any]) -> str:
    base = a["checkpoints"][0]["scores"]
    rows = []
    for c in a["checkpoints"]:
        if c["is_boundary"] and c["index"] > 0 and "recover" in c:
            for t, label in {**MC, **GEN}.items():
                if t in c["scores"] and f"lm_eval/{t}" in c["recover"]:
                    drop = (base[t] - c["scores"][t]) * 100
                    back = (c["recover"][f"lm_eval/{t}"] - c["scores"][t]) * 100
                    rows.append((c["index"], c["domain_index"], label, drop, back))
    if not rows:
        return "<p class='empty'>no recoverability probes</p>"
    names = [d["name"] for d in a["config"]["data"]["domains"]]
    o = ["<table class='tbl'><tr><th>after domain</th><th>eval</th><th>drop from base (pts)</th><th>recovered by 2M-token general re-warm (pts)</th><th>share recovered</th></tr>"]
    for idx, di, label, drop, back in rows:
        share = f"{back / drop * 100:.0f}%" if drop > 0.5 else "—"
        bar = f"<div class='bar'><i style='width:{min(max(drop, 0), 30) / 30 * 100:.0f}%'></i><b style='width:{min(max(back, 0), 30) / 30 * 100:.0f}%'></b></div>"
        o.append(f"<tr><td>{_esc(names[di] if 0 <= di < len(names) else di)}</td><td>{_esc(label)}</td><td>{drop:+.1f}</td><td>{back:+.1f} {bar}</td><td>{share}</td></tr>")
    o.append("</table><p class='note'>At each domain boundary the model is briefly re-warmed on 2M tokens of general text and re-evaluated, "
             "then the weights are restored. The part of the drop that comes back was spurious drift; the rest is genuine forgetting.</p>")
    return "".join(o)


def instrument(a: dict[str, Any]) -> str:
    ck = a["checkpoints"]
    xs = [c["gradient_tokens"] for c in ck]
    bands = _domain_bands(a)
    lim = a["config"]["model"].get("sae_fvu_invalid_delta", 0.1)
    f = Chart(title="SAE reconstruction drift (FVU − base) per layer; instrument invalid above the limit", y_label="FVU Δ", height=220, y_fmt="{:.3f}")
    for b in bands:
        f.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
    f.hline(lim, "#b91c1c", "validity limit")
    for L in ("7", "13", "22"):
        f.add(f"layer {L}", xs, [c["instrument"].get(L, {}).get("fvu_delta") for c in ck])
    s = Chart(title="Representation similarity to the base model (layer 13)", y_label="similarity", height=220, y_range=(0, 1.02))
    for b in bands:
        s.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
    s.add("standardised CKA", xs, [c["instrument"]["13"].get("cka_std") for c in ck])
    s.add("mutual-kNN", xs, [c["instrument"]["13"].get("mknn") for c in ck])
    s.add("readout probe trained on base", xs, [c["instrument"]["13"].get("readout_base_probe_acc") for c in ck])
    s.add("readout probe trained fresh", xs, [c["instrument"]["13"].get("readout_fresh_probe_acc") for c in ck], dashed=True)
    s.add("raw CKA (outlier-dominated)", xs, [c["instrument"]["13"].get("cka_vs_base") for c in ck], color="#9ca3af", dashed=True)
    return (f"{f.render()}{s.render()}<p class='note'>The sparse autoencoders (Gemma Scope 2) were trained on the base model. If their "
            "reconstruction error drifts past the limit, feature-level claims are no longer trustworthy and the report says so. "
            "The gap between the base-trained and fresh readout probes is the part of the representation change that a downstream "
            "reader would have to re-learn.</p>")


def features(a: dict[str, Any], top: int = 12) -> str:
    o = []
    for tag, f in a["features"].items():
        if not f:
            continue
        dc = f["drift_corrected"]
        o.append(f"<h3>{_esc(tag)} <span class='muted'>— {f['n_alive_base']} features alive at base; global firing ratio {f['global_freq_ratio_median']:.2f}; "
                 f"drift-corrected: {dc['n_over_50pct_decay']} lost >50% of their firing rate, {dc['n_over_2x_growth']} more than doubled; "
                 f"top 10% of decayers carry {dc['top10pct_share'] * 100:.0f}% of the lost firing</span></h3>")
        o.append("<div class='cols'><table class='tbl'><tr><th>decayed feature</th><th>freq base</th><th>freq end</th><th>ratio (corrected)</th></tr>")
        for e in f["top_decayed"][:top]:
            o.append(f"<tr><td>{e['feature']}</td><td>{e['freq_base']:.4f}</td><td>{e['freq_end']:.4f}</td><td>{e['ratio_corrected']:.2f}</td></tr>")
        o.append("</table><table class='tbl'><tr><th>grown feature</th><th>freq base</th><th>freq end</th><th>ratio (corrected)</th></tr>")
        for e in f["top_grown"][:top]:
            o.append(f"<tr><td>{e['feature']}</td><td>{e['freq_base']:.4f}</td><td>{e['freq_end']:.4f}</td><td>{e['ratio_corrected']:.2f}</td></tr>")
        o.append("</table></div>")
    o.append("<p class='note'>Firing frequency per SAE feature on a fixed held-out probe set, end vs base, divided by the median change "
             "across all features so a uniform rise in firing does not hide decay. Feature ids index the Gemma Scope 2 dictionaries.</p>")
    return "".join(o)


def domain_feature_section(run_dir: Path, a: dict[str, Any], layer: str = "13", n_examples: int = 3) -> str:
    """Which domain's features decay, at which stage, with names. Needs domain_features.json."""
    p = run_dir / "domain_features.json"
    if not p.exists():
        return "<p class='empty'>run <code>clh features RUN --pass</code> to resolve features by domain</p>"
    d = json.load(open(p))
    lay = d["layers"].get(layer) or next(iter(d["layers"].values()))
    ck = a["checkpoints"]
    tok_by_idx = {c["index"]: c["gradient_tokens"] for c in ck}
    bands = _domain_bands(a)
    o = []
    o.append(f"<p>At the base model, each SAE feature is assigned to the domain whose held-out text it fires on most "
             f"(≥ 50% of its firing on one domain's probe text; the rest are <i>shared</i>). Layer {layer}: "
             + ", ".join(f"{k} {v}" for k, v in lay["set_sizes"].items()) + " features.</p>")
    ch = Chart(title=f"Firing rate of each domain's feature set, relative to base (median over the set, drift-corrected) — layer {layer}",
               y_label="× base firing", height=300, y_fmt="{:.2f}")
    for b in bands:
        ch.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
    ch.hline(1.0, "#9ca3af", "unchanged")
    xs = [tok_by_idx.get(i) for i in lay["pooled_ckpts"]]
    for name, ys in lay["trajectory"].items():
        ch.add(f"{name} features", xs, ys, dashed=(name == "shared"))
    o.append(ch.render())
    o.append("<p class='note'>Measured on the full mixed probe set at every checkpoint, so the domain being trained on can only "
             "push its own features up by changing the model, not by changing the text.</p>")
    # stage table: each set on its own domain's text
    doms = [k for k in lay["set_sizes"] if k != "shared"]
    first = next(iter(lay["stage"].values()))
    o.append("<h3>Each domain's features on their own text, stage by stage</h3>")
    o.append("<table class='tbl'><tr><th>stage</th><th>all features<br><span class='muted'>total firing × base</span></th>" + "".join(f"<th>{_esc(dm)} features ({first['sets'][dm]['n']})<br><span class='muted'>raw × base · vs typical feature · selectivity</span></th>" for dm in doms if dm in first["sets"]) + "</tr>")
    for c, st in lay["stage"].items():
        cells = []
        anyx = next(iter(st["sets"].values()))
        cells.append(f"<td>{anyx['total_firing_ratio']:.2f}</td>")
        for dm in doms:
            x = st["sets"].get(dm)
            if not x:
                cells.append("<td>—</td>"); continue
            cls = " style='color:#b91c1c'" if x["median_ratio"] < 0.7 else (" style='color:#15803d'" if x["median_ratio"] >= 1.1 else "")
            cells.append(f"<td{cls}>{x['median_ratio_raw']:.2f} · {x['median_ratio']:.2f} · {x['selectivity']:.2f}</td>")
        o.append(f"<tr><td>{_esc(st['label'])}</td>{''.join(cells)}</tr>")
    o.append("</table><p class='note'><b>How to read it.</b> <i>raw × base</i>: median change in how often the set's features fire on their own domain's held-out text. "
             "<i>vs typical feature</i>: the same, divided by the median change of <i>all</i> features on that text — below 1 means the set is losing ground to the rest. "
             "<i>selectivity</i>: the share of the set's firing that lands on its own domain's text (1 = fires only there). "
             "Recomputed from saved weights at the boundary checkpoints. Red: the set fires under 0.7× the typical feature.</p>")
    # labelled examples
    o.append("<p class='note'>The single most-decayed features in every domain are document-structure features — end-of-text, line breaks, citation punctuation — "
             "i.e. the model re-learning the training format (notebook T16). The examples below are the most-decayed <i>content</i> features.</p>")
    for dm in doms:
        ex = lay["examples"].get(dm) or []
        if not ex:
            continue
        o.append(f"<h3>Most-decayed {_esc(dm)} content features by the end <span class='muted'>(what they fire on in the base model)</span></h3><div class='gallery'>")
        for e in ex[:n_examples]:
            snippets = "".join(f"<div class='snip'>{_esc(cx['before'][-70:])}<mark>{_esc(cx['token'])}</mark>{_esc(cx['after'][:25])}</div>" for cx in e["contexts"])
            o.append(f"<div class='item'><div class='meta'>feature {e['feature']} · fires on {e['freq_base'] * 100:.2f}% of tokens at base · {e['ratio_end']:.2f}× at the end</div>{snippets or '<div class=muted>no snippet</div>'}</div>")
        o.append("</div>")
    return "".join(o)


CSS = """
body{font:15px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;color:#111827;background:#fff;margin:0}
main{max-width:980px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 8px;padding-top:10px;border-top:1px solid #e5e7eb}h3{font-size:15px;margin:18px 0 6px}
.sub,.note,.muted{color:#6b7280}.note{font-size:13px}.muted{font-weight:normal;font-size:13px}.empty{color:#9ca3af;font-style:italic}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}
.card{border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px}.card .k{font-size:12px;color:#6b7280}.card .v{font-size:22px;font-weight:600;margin:2px 0}.card .d{font-size:12px;color:#6b7280}
.chart{width:100%;height:auto;display:block;margin:6px 0 14px}.ctitle{font-size:13px;font-weight:600;fill:#111827}.tick{font-size:10px;fill:#6b7280}.grid{stroke:#f3f4f6}.band{font-size:10px;fill:#6b7280}.legend{font-size:11px;fill:#374151}
.tbl{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}.tbl th,.tbl td{border-bottom:1px solid #e5e7eb;padding:4px 8px;text-align:left}.tbl th{font-weight:600;color:#374151;background:#f9fafb}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.cols{grid-template-columns:1fr}}
.bar{display:inline-block;width:120px;height:8px;background:#f3f4f6;vertical-align:middle;position:relative;margin-left:8px}.bar i{position:absolute;left:0;top:0;height:8px;background:#fca5a5}.bar b{position:absolute;left:0;top:0;height:8px;background:#86efac}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}.item{border:1px solid #e5e7eb;border-radius:8px;padding:10px;font-size:13px}
.item .q{white-space:pre-wrap;max-height:9.5em;overflow:hidden}.item .ans{margin-top:6px}.item .ans.wrong{color:#b91c1c}.item .meta{color:#6b7280;font-size:12px;margin-top:6px}
.snip{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word;margin:4px 0}.snip mark{background:#fde68a}
.strip{display:inline-block;vertical-align:middle}.strip i{display:inline-block;width:5px;height:9px;margin-right:1px}.strip .c1{background:#86efac}.strip .c0{background:#fca5a5}.strip .cx{background:#e5e7eb}
"""


def build_report(run: str) -> Path:
    run_dir = REPO_ROOT / "results" / "runs" / run
    a = json.load(open(run_dir / "analysis.json"))
    items = json.load(open(run_dir / "items.json")) if (run_dir / "items.json").exists() else None
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    secs = [
        ("Eval curves", eval_curves(a)),
        ("The dashboard said fine", dashboard_said_fine(a, rows)),
        ("What was forgotten", gallery(items) if items else "<p class='empty'>run <code>clh items</code> first</p>"),
        ("Recoverable vs genuine", recoverability(a)),
        ("Instrument health", instrument(a)),
        ("Which features decay, by domain and stage", domain_feature_section(run_dir, a)),
        ("Features lost and gained", features(a)),
    ]
    dfp = run_dir / "domain_features.json"
    dfj = json.load(open(dfp)) if dfp.exists() else None
    body = [f"<h1>Forgetting report: {_esc(run)}</h1>", headline(a, items), render_findings(findings(a, items, dfj), "What this run says")]
    for title, content in secs:
        body.append(f"<h2>{_esc(title)}</h2>{content}")
    page = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{_esc(run)} — forgetting report</title><style>{CSS}{FINDINGS_CSS}</style></head><body><main>{''.join(body)}</main></body></html>"
    out = run_dir / "report.html"
    out.write_text(page)
    return out
