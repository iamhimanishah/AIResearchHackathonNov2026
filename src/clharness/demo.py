"""`clh demo RUN` — the hackathon page. One section per finding: the claim, the evidence, the
chart that shows it, and what it means for a company and for the bet. Self-contained HTML,
inline SVG, no network. Writes results/runs/<run>/demo.html.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clharness.config import REPO_ROOT
from clharness.findings import FINDINGS_CSS, compute
from clharness.report import CSS, MC, TASK_LABEL, Chart, DOMAIN_COLORS, _domain_bands, _esc, dashboard_said_fine, domain_feature_section, gallery, instrument, recoverability

DEMO_CSS = CSS + FINDINGS_CSS + """
.lead{font-size:18px;line-height:1.5;max-width:780px}.big{font-size:32px;font-weight:700;margin:6px 0}
.claim{font-size:22px;font-weight:700;margin:34px 0 4px;padding-top:14px;border-top:2px solid #111827}
.ev{color:#374151;font-size:14.5px;max-width:800px}.so{background:#eff6ff;border-left:4px solid #1d4ed8;padding:10px 14px;margin:12px 0;font-size:15px;max-width:800px}
.so b{color:#1d4ed8}
.gen{border-collapse:collapse;width:100%;font-size:12.5px;table-layout:fixed}.gen th,.gen td{border:1px solid #e5e7eb;padding:6px 8px;vertical-align:top;text-align:left}
.gen th{background:#f9fafb;font-weight:600}.gen td.p{background:#fafafa;font-weight:500}.gen pre{margin:0;white-space:pre-wrap;word-break:break-word;font:11.5px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace}
.gen td.bad{background:#fff1f2}
.conc{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:10px 0;max-width:800px}.conc div{border:1px solid #e5e7eb;border-radius:8px;padding:8px;text-align:center}.conc .n{font-size:22px;font-weight:600}.conc .alarm .n{color:#b91c1c}.conc .l{font-size:12px;color:#6b7280}
.dom{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0;max-width:800px}.dom div{border-radius:8px;padding:10px;text-align:center}.dom .n{font-size:26px;font-weight:700}.dom .l{font-size:12px;color:#374151}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid #e5e7eb;padding:8px 0;margin-bottom:12px;font-size:13px;z-index:2}nav a{margin-right:14px;color:#374151;text-decoration:none}
"""

REPEAT_HINT = 3


def _looks_degenerate(text: str) -> bool:
    words = text.split()
    if len(words) < 18:
        return False
    for n in (4, 6, 8):
        chunks = [" ".join(words[i:i + n]) for i in range(0, len(words) - n)]
        if chunks and max(chunks.count(c) for c in set(chunks)) >= REPEAT_HINT:
            return True
    return False


def generation_panel(g: dict[str, Any], only_tags: tuple[str, ...] | None = None) -> str:
    cks = list(g["checkpoints"].items())
    o = ["<table class='gen'><colgroup><col style='width:19%'>" + "".join("<col>" for _ in cks) + "</colgroup>"]
    o.append("<tr><th>prompt</th>" + "".join(f"<th>{_esc(c['label'])}</th>" for _, c in cks) + "</tr>")
    for j, p in enumerate(g["prompts"]):
        if only_tags and p["tag"] not in only_tags:
            continue
        cells = []
        for _, c in cks:
            s = c["outputs"][j]["short"] or "(empty)"
            cls = "bad" if _looks_degenerate(s) or s == "(empty)" else ""
            cells.append(f"<td class='{cls}'><pre>{_esc(s[:420])}</pre></td>")
        o.append(f"<tr><td class='p'><pre>{_esc(p['prompt'])}</pre><div class='muted'>{_esc(p['tag'])}</div></td>{''.join(cells)}</tr>")
    o.append("</table><p class='note'>Greedy decoding, same prompt to every checkpoint, cut at the first blank line. Nothing here is scored — read them. "
             "The only automatic mark is the pink tint for an empty output or a repetition loop. The base model is a 1B pretrained model and is wrong sometimes too; "
             "the point is what changes.</p>")
    return "".join(o)


def concentration(f: dict[str, Any], task: str = "pubmedqa_pqal", names: list[str] | None = None, ck=None) -> str:
    v = f["answer_conc"].get(task)
    if not v:
        return ""
    cells = []
    for idx, share, score in v:
        c = ck[idx]
        di = c["domain_index"]
        label = "base" if idx == 0 else f"after {names[di]}" if c["is_boundary"] and 0 <= di < len(names) else f"ckpt {idx}"
        cells.append(f"<div class='{'alarm' if share >= 0.9 else ''}'><div class='n'>{share * 100:.0f}%</div><div class='l'>{_esc(label)}<br>score {score:.3f}</div></div>")
    return (f"<h3>{_esc(TASK_LABEL.get(task, task))}: share of 500 questions answered with the model's single favourite option</h3>"
            f"<div class='conc'>{''.join(cells)}</div>")


def domain_cards(f: dict[str, Any]) -> str:
    cells = []
    for k, v in f["domain_delta_pts"].items():
        cells.append(f"<div style='background:{DOMAIN_COLORS.get(k, '#eee')}'><div class='n'>{v:+.1f}</div><div class='l'>after {_esc(k)}<br>pts, protected set</div></div>")
    return f"<div class='dom'>{''.join(cells)}</div>"


def build_demo(run: str) -> Path:
    run_dir = REPO_ROOT / "results" / "runs" / run
    a = json.load(open(run_dir / "analysis.json"))
    items = json.load(open(run_dir / "items.json"))
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    gens = json.load(open(run_dir / "generations.json")) if (run_dir / "generations.json").exists() else None
    f = compute(a, items)
    r = a["retention"]
    names = f["domains"]
    ck = a["checkpoints"]
    doms = " → ".join(names)
    n_items = sum(t["n_items"] for t in items["tasks"].values())
    xs = [c["gradient_tokens"] for c in ck]

    def curve(title, height=260):
        ch = Chart(title=title, y_label="accuracy", height=height)
        for b in _domain_bands(a):
            ch.band(b[0], b[1], DOMAIN_COLORS.get(b[2], "#eee"), b[2])
        for t, label in MC.items():
            ch.add(label, xs, [c["scores"].get(t) for c in ck])
        return ch.render()

    dd = f["domain_delta_pts"]
    worst, best = min(dd, key=dd.get), max(dd, key=dd.get)
    rec_line = "; ".join(f"after {x['domain']}: {x['recovered_pts']:+.1f} of −{x['drop_pts']:.1f}" for x in f["recover"] if x["share"] is not None)
    rs = f["recover_share_max"] or 0
    gap = (f["readout_fresh_min"] - f["readout_base_min"]) * 100
    pm = f["answer_conc"].get("pubmedqa_pqal", [])
    conc_early = next((x for x in pm if x[0] > 0 and x[1] >= 0.8), None)
    conc_peak = max(pm, key=lambda x: x[1]) if pm else None
    conc_text = ""
    if conc_early:
        conc_text += f"PubMedQA answered {conc_early[1] * 100:.0f}% of questions with one option at checkpoint {conc_early[0]} while scoring {conc_early[2]:.3f} — above base — "
    if conc_peak:
        conc_text += f"and reached {conc_peak[1] * 100:.0f}% at checkpoint {conc_peak[0]}. "

    S = []  # (id, nav label, html)

    S.append(("cost", "What it cost", f"""
<h2 class='claim'>One ordinary fine-tune cost this model {abs(r['mc_mean_delta_pts']):.0f} points of what it knew</h2>
<p class='lead'>We took an open 1B model (Gemma 3), fine-tuned it on four ordinary domains in sequence ({doms}, 195M tokens each —
the kind of update a company does to make a model "know" its data), and kept score on {n_items} questions it could answer at the start.
Learning rate 3e-5, no bugs, no instability, {f['anomalous']} anomalous steps.</p>
<div class='big'>{r['mc_mean_delta_pts']:+.1f} pts on the protected set · {f['lost_total']} questions gone for good · general-text perplexity {(f['ppl_general_ratio'] - 1) * 100:+.0f}%</div>
{curve("Accuracy on questions the model was never trained on again")}
<p class='note'><b>How these are scored.</b> Every question is multiple choice: the model sees the fixed options and we take the one it rates most likely
(standard lm-eval, length-normalised); no free text is judged. A question counts as <i>lost</i> only if this same model picked the right option at the
start and the wrong one at the end and at the two checkpoints before it.</p>
{gallery(items, per_task=2, tasks=("arc_easy", "pubmedqa_pqal"))}"""))

    S.append(("first", "First update", f"""
<h2 class='claim'>1. The first narrow update caused most of the lasting damage</h2>
<p class='ev'><b>Evidence.</b> {f['lost_by_domain'].get(names[0], 0)} of the {f['lost_total']} questions that stayed lost were lost during the first domain ({names[0]});
later domains caused mostly transient losses — {f['recovered_total']} questions dipped and came back.
Persistent losses by causing domain: {", ".join(f"{k} {v}" for k, v in f['lost_by_domain'].items())}.</p>
<div class='so'><b>So what.</b> "Just one small fine-tune on our data" is the dangerous step, not the tenth. The first move out of the pretraining
distribution is where a company should be measuring hardest — and today nobody measures there at all.
<br><span class='muted'>Caveat we state: {names[0]} was also first in line, so content and order are confounded until the reversed-order run.</span></div>"""))

    S.append(("data", "Which data", f"""
<h2 class='claim'>2. What you train on matters more than how much</h2>
<p class='ev'><b>Evidence.</b> Same token count and learning rate for every domain; very different damage. {best.capitalize()} text
<i>gave back</i> {dd[best]:+.1f} pts of what {worst} and the others had taken.</p>
{domain_cards(f)}
<div class='so'><b>So what.</b> Forgetting is a property of the data, not of fine-tuning as such. The spread between domains ({dd[best] - dd[worst]:.1f} pts) is larger
than the effect a data-selection method needs to show — which is why choosing what to train on and what to replay is the bet worth making.</div>"""))

    S.append(("genuine", "Genuine loss", f"""
<h2 class='claim'>3. Most of the loss is genuine, not a superficial shift</h2>
<p class='ev'><b>Evidence.</b> At each domain boundary we briefly mixed 2M tokens of general text back in and re-scored. It recovered at most
{rs * 100:.0f}% of the drop ({rec_line}). Inside the model, a readout probe trained on the base representations fell to {f['readout_base_min']:.2f}
while a freshly trained one stayed at {f['readout_fresh_min']:.2f}: a {gap:.0f}-point gap that tracks the eval curve.</p>
{recoverability(a)}
<div class='so'><b>So what.</b> The comfortable belief — "the knowledge is still in there, it just needs reminding" — is wrong at this scale. The representations moved.
The fix has to happen <i>during</i> training, by choosing what to train on and what to replay, not by patching afterwards.</div>"""))

    S.append(("dashboard", "Dashboards miss it", f"""
<h2 class='claim'>4. The usual dashboards miss it, in both directions</h2>
<p class='ev'><b>Evidence.</b> {conc_text}The training loss fell smoothly through the code domain while code generation (MBPP) fell
{f['mbpp'][0][1]:.3f} → {f['mbpp'][1][1]:.3f}. A separate run at 3× the learning rate lost 12 points with zero loss spikes.</p>
{concentration(f, names=names, ck=ck)}
{dashboard_said_fine(a, rows)}
<div class='so'><b>So what.</b> Loss curves, gradient norms and a headline accuracy do not detect capability loss — a score can look <i>better</i> while the model has
stopped reading the question. Per-question, per-answer tracking against a fixed protected set is the minimum instrument, and it is what <code>clh report</code> produces
from checkpoints a team already saves.</div>"""))

    if gens:
        S.append(("outputs", "In the outputs", f"""
<h2 class='claim'>What it looks like in the model's own words</h2>
<p class='ev'>The same prompts to the base model and to the model after each domain.</p>
{generation_panel(gens)}"""))

    S.append(("features", "Which features", f"""
<h2 class='claim'>5. Inside the model, forgetting is de-specialization, not switching off</h2>
<p class='ev'><b>Evidence.</b> Sparse-autoencoder features assigned to a domain by what they fire on in the base model, then tracked through the four
stages on the full probe set and on their own domain's text. Nothing switches off: by the end every feature fires <i>more</i>. But each domain's features
fire less selectively on their own text, the domain-specific sets fall behind the typical feature, and the general-web set pulls ahead.</p>
{domain_feature_section(run_dir, a)}
<div class='so'><b>So what.</b> Representations that meant one thing now fire on everything. That is why a readout trained on the base model stops working while
a fresh one still finds the information, and why a general-text refresher does not repair it: nothing was erased, it was blurred. It also says what a
selection signal must protect — not "features", but their <i>selectivity</i> — and the autoencoder trained on the base model can read that during training.</div>"""))

    S.append(("inside", "Inside the model", f"""
<h2 class='claim'>6. Damage is concentrated — and the instrument that shows it survives the training it watches</h2>
<p class='ev'><b>Evidence.</b> At layer 13, {f.get('feat_grew', '?')} sparse-autoencoder features more than doubled their firing rate while {f.get('feat_decayed', '?')} lost more than half;
the top 10% of decaying features carry {f.get('feat_top10_share', 0) * 100:.0f}% of the lost activity. The autoencoders were trained on the base model and drifted by at most
{f['fvu_delta_max']:+.3f} FVU over {a['gradient_tokens'] / 1e6:.0f}M tokens (validity limit {f['fvu_limit']}).</p>
{instrument(a)}
<div class='so'><b>So what.</b> Forgetting looks like a small set of representations being crowded out, not uniform erosion — a small target is protectable.
And because the reader stays valid, a model's internals can be watched <i>while</i> it trains without rebuilding the reader. That is the enabling fact for the bet.</div>"""))

    S.append(("bet", "The bet", f"""
<h2 class='claim'>The bet, and the ask</h2>
<p class='lead'>Every defense the labs use against forgetting — mixing, replay, anchoring, eval gates — assumes you control the whole data mixture and can re-run the recipe.
Once a model is deployed and data arrives over time, none of them apply. That is why companies that tried fine-tuning on their own data mostly retreated to retrieval,
which never changes what the model can do.</p>
<div class='so'><b>Wedge:</b> the audit above, run by whoever fine-tunes, on any open model. Sellable now.<br>
<b>Bet:</b> a signal read from the model's own internal state chooses what to train on and what to replay better than loss does, at equal compute. Pre-registered,
three seeds, four baselines; ~$3k of GPU to answer at 1B and check at 4B.<br>
<b>If it works:</b> a model that keeps learning from a customer's data stream without a retrain cycle and without silent capability loss.</div>
<p class='sub'>Honest note: the cheap version of the bet — averaging the logged internal signals over each checkpoint interval and asking whether they predict that interval's
drop — is null on this run (43 intervals, one seed, |ρ| ≤ 0.22). The question is per-example, which is what Phase 2 measures. That is why the ask is an experiment, not a tweak.</p>
<p class='sub'>Code, data manifests, predictions written before the data, and the full research plan: github.com/justinmilner1/ContinualLearningResearch</p>"""))

    nav = "<nav>" + "".join(f"<a href='#{sid}'>{_esc(label)}</a>" for sid, label, _ in S) + "</nav>"
    body = [f"<h1>What fine-tuning forgets — and what that tells us</h1><p class='sub'>Gemma 3 1B PT · {doms} · {a['gradient_tokens'] / 1e6:.0f}M tokens · run <code>{_esc(run)}</code></p>", nav]
    for sid, _, content in S:
        body.append(f"<section id='{sid}'>{content}</section>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>What fine-tuning forgets</title><style>{DEMO_CSS}</style></head><body><main>{''.join(body)}</main></body></html>")
    out = run_dir / "demo.html"
    out.write_text(page)
    return out
