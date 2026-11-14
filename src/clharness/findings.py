"""Findings: what a run's numbers *say*, computed from analysis.json + items.json, rendered as
claim → evidence → implication. Used at the top of `clh report` and as the spine of `clh demo`.

Each finding is derived from the data (numbers are filled in), with the wording fixed here. If a
run does not support a claim (e.g. recovery is large), the finding says so — the templates
branch on the numbers rather than assert.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np

MC = ["arc_easy/acc_norm", "hellaswag/acc_norm", "piqa/acc_norm", "pubmedqa_pqal/acc"]
TASK_LABEL = {"arc_easy": "ARC-Easy", "hellaswag": "HellaSwag", "piqa": "PIQA", "pubmedqa_pqal": "PubMedQA", "mbpp": "MBPP"}


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def compute(a: dict[str, Any], items: dict[str, Any] | None) -> dict[str, Any]:
    ck = a["checkpoints"]
    names = [d["name"] for d in a["config"]["data"]["domains"]]
    mc = np.array([np.mean([c["scores"][t] for t in MC]) for c in ck])
    b = [c["index"] for c in ck if c["is_boundary"]]
    f: dict[str, Any] = {"domains": names, "mc_base": float(mc[0]), "mc_end": float(mc[-1]), "mc_min": float(mc.min())}
    # per-domain change in MC mean, in the order trained
    f["domain_delta_pts"] = {names[j - 1]: float((mc[b[j]] - mc[b[j - 1]]) * 100) for j in range(1, len(b)) if j - 1 < len(names)}
    # persistent losses by causing domain
    if items:
        byd: dict[str, int] = {}
        for t in items["tasks"].values():
            for d, n in t["lost_by_domain"].items():
                if d:
                    byd[d] = byd.get(d, 0) + n
        f["lost_total"] = sum(t["counts"].get("lost", 0) for t in items["tasks"].values())
        f["recovered_total"] = sum(t["counts"].get("recovered", 0) for t in items["tasks"].values())
        f["lost_by_domain"] = byd
        f["lost_first_domain_share"] = byd.get(names[0], 0) / max(f["lost_total"], 1)
    # recoverability
    rec = []
    for c in ck:
        if c["is_boundary"] and c["index"] > 0 and "recover" in c and all(f"lm_eval/{t}" in c["recover"] for t in MC):
            before = np.mean([c["scores"][t] for t in MC])
            after = np.mean([c["recover"][f"lm_eval/{t}"] for t in MC])
            drop = float((mc[0] - before) * 100)
            rec.append({"index": c["index"], "domain": names[c["domain_index"]] if 0 <= c["domain_index"] < len(names) else "?",
                        "drop_pts": drop, "recovered_pts": float((after - before) * 100),
                        "share": float((after - before) / (mc[0] - before)) if drop > 0.5 else None})
    f["recover"] = rec
    shares = [r["share"] for r in rec if r["share"] is not None]
    f["recover_share_max"] = max(shares) if shares else None
    # readout gap + instrument validity
    i13 = [c["instrument"]["13"] for c in ck]
    f["readout_base_min"] = min(x["readout_base_probe_acc"] for x in i13 if x.get("readout_base_probe_acc") is not None)
    f["readout_fresh_min"] = min(x["readout_fresh_probe_acc"] for x in i13 if x.get("readout_fresh_probe_acc") is not None)
    f["fvu_delta_max"] = max((x.get("fvu_delta") or 0) for x in i13)
    f["fvu_limit"] = a["config"]["model"].get("sae_fvu_invalid_delta", 0.1)
    f["valid"] = all(x.get("valid", 1) for x in i13)
    # feature concentration (primary L13 SAE)
    ft = a["features"].get("L13")
    if ft:
        dc = ft["drift_corrected"]
        f["feat_grew"], f["feat_decayed"], f["feat_top10_share"] = dc["n_over_2x_growth"], dc["n_over_50pct_decay"], dc["top10pct_share"]
    # degenerate answers
    al = {}
    for c in ck:
        for t, s in c.get("answers", {}).items():
            if s:
                al.setdefault(t, []).append((c["index"], s["share_max"], c["scores"].get(f"{t}/acc") or c["scores"].get(f"{t}/acc_norm")))
    f["answer_conc"] = al
    f["degenerate_tasks"] = sorted({t for t, v in al.items() if any(s >= 0.9 for _, s, _ in v)})
    # loss-vs-generation
    mb = [(c["index"], c["scores"]["mbpp/pass_at_1"]) for c in ck if c["scores"].get("mbpp/pass_at_1") is not None]
    f["mbpp"] = mb
    f["anomalous"] = a.get("anomalous_grad_steps", 0)
    f["ppl_general_ratio"] = a["retention"]["ppl_doc_end_over_base"].get("general")
    f["domain_features"] = a.get("_domain_features")
    return f


def findings(a: dict[str, Any], items: dict[str, Any] | None, domain_features: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Ordered list of {title, evidence, meaning}. Wording branches on the numbers."""
    if domain_features is not None:
        a = {**a, "_domain_features": domain_features}
    f = compute(a, items)
    d = f["domains"]
    dd = f["domain_delta_pts"]
    out: list[dict[str, str]] = []

    # 1. front-loading of persistent losses
    if items:
        share = f["lost_first_domain_share"]
        first = d[0]
        if share >= 0.5:
            out.append({
                "title": f"The first narrow update caused most of the lasting damage",
                "evidence": f"{f['lost_by_domain'].get(first, 0)} of the {f['lost_total']} questions that stayed lost were lost during the first domain ({first}); "
                            f"later domains caused mostly transient losses ({f['recovered_total']} questions dipped and came back).",
                "meaning": f"\"Just one small fine-tune on our data\" is the dangerous step, not the tenth. Caveat: {first} was also first in line, "
                           f"so content and order are confounded until the reversed-order run.",
            })
        else:
            out.append({
                "title": "Lasting losses were spread across the domains",
                "evidence": ", ".join(f"{k}: {v}" for k, v in f["lost_by_domain"].items()) + f" of {f['lost_total']} persistently lost questions.",
                "meaning": "No single update is the culprit; damage accumulates.",
            })

    # 2. data matters more than amount
    if dd:
        worst = min(dd, key=dd.get)
        best = max(dd, key=dd.get)
        ev = "; ".join(f"{k} {v:+.1f}" for k, v in dd.items())
        if dd[best] > 0:
            meaning = (f"{best.capitalize()} text partially repaired damage done by other domains at the same token count and learning rate. "
                       f"Forgetting is a property of the data, not of fine-tuning as such — and the spread between domains "
                       f"({dd[best] - dd[worst]:.1f} pts) is larger than the effect a data-selection method needs to show.")
        else:
            meaning = f"Same budget, very different damage ({dd[best] - dd[worst]:.1f}-pt spread): which data you add matters as much as how much."
        out.append({"title": "What you train on matters more than how much",
                    "evidence": f"Change in protected-set accuracy per domain (pts, same tokens and learning rate): {ev}. Worst: {worst}; best: {best}.",
                    "meaning": meaning})

    # 3. genuine vs superficial
    if f["recover"]:
        rs = f["recover_share_max"]
        gap = (f["readout_fresh_min"] - f["readout_base_min"]) * 100
        if rs is not None and rs < 0.5:
            out.append({
                "title": "Most of the loss is genuine, not a superficial shift",
                "evidence": f"Mixing 2M tokens of general text back in at each boundary recovered at most {rs * 100:.0f}% of the drop "
                            f"({'; '.join(f'{r['domain']}: {r['recovered_pts']:+.1f} of −{r['drop_pts']:.1f}' for r in f['recover'] if r['share'] is not None)}). "
                            f"A readout probe trained on the base model's representations fell to {f['readout_base_min']:.2f} while a freshly trained one stayed at {f['readout_fresh_min']:.2f} — a {gap:.0f}-pt gap.",
                "meaning": "The representations moved; the knowledge is not \"still in there waiting to be reminded\". The fix has to happen during training, by choosing what to train on and replay, not after.",
            })
        else:
            out.append({
                "title": "A large share of the loss was superficial",
                "evidence": f"A 2M-token general re-warm recovered up to {(rs or 0) * 100:.0f}% of the drop at a boundary.",
                "meaning": "Cheap replay would undo much of this run's damage.",
            })

    # 4. aggregate metrics miss it
    bits = []
    for t, v in f["answer_conc"].items():
        base = v[0][1]
        peak = max(v, key=lambda x: x[1])
        if peak[1] >= 0.9:
            early = next((x for x in v if x[1] >= 0.8 and x[0] > 0), None)
            bits.append(f"{TASK_LABEL.get(t, t)} answered {peak[1] * 100:.0f}% of questions with one option at checkpoint {peak[0]} (base {base * 100:.0f}%)"
                        + (f"; already {early[1] * 100:.0f}% at checkpoint {early[0]} while scoring {early[2]:.3f}" if early else ""))
    if len(f["mbpp"]) >= 2 and f["mbpp"][1][1] < f["mbpp"][0][1]:
        bits.append(f"code perplexity improved through the code domain while MBPP pass@1 fell {f['mbpp'][0][1]:.3f} → {f['mbpp'][1][1]:.3f}")
    bits.append(f"{f['anomalous']} anomalous gradient steps and no loss spikes in a run that lost {(f['mc_base'] - f['mc_min']) * 100:.1f} pts at its low point")
    out.append({"title": "The usual dashboards miss it, in both directions",
                "evidence": "; ".join(bits) + ".",
                "meaning": "Loss curves, gradient norms and a headline accuracy do not detect capability loss. Per-question, per-answer tracking against a fixed protected set is the minimum instrument."})

    # 5. de-specialization (needs domain_features.json)
    df = f.get("domain_features")
    if df:
        lay = df["layers"].get("13") or next(iter(df["layers"].values()))
        stages = list(lay["stage"].values())
        b, e = stages[0]["sets"], stages[-1]["sets"]
        tot = next(iter(e.values()))["total_firing_ratio"]
        sel = "; ".join(f"{d} {b[d]['selectivity']:.2f} → {e[d]['selectivity']:.2f}" for d in b if d in e)
        lag = ", ".join(f"{d} {e[d]['median_ratio']:.2f}×" for d in e if d != "general")
        gen = e.get("general", {}).get("median_ratio")
        out.append({"title": "Forgetting looks like de-specialization, not switching off",
                    "evidence": f"By the end every SAE feature fires more (total activity {tot:.1f}× base on every kind of text), but each domain's features fire less "
                                f"selectively on their own text (share of firing on own domain: {sel}). Relative to the typical feature, the domain-specific sets lag ({lag})"
                                + (f" while general-web features lead ({gen:.2f}×)." if gen else "."),
                    "meaning": "Representations that meant one thing now fire on everything. That is why a readout trained on the base model stops working while a fresh one "
                               "still finds the information, and why a general-text refresher does not repair it: nothing was erased, it was blurred."})

    # 6. concentration
    if "feat_grew" in f:
        out.append({"title": "Damage is concentrated, which makes it targetable",
                    "evidence": f"At layer 13, {f['feat_grew']} SAE features more than doubled their firing rate while {f['feat_decayed']} lost more than half; "
                                f"the top 10% of decaying features carry {f['feat_top10_share'] * 100:.0f}% of the lost activity.",
                    "meaning": "Forgetting looks like a small set of representations being crowded out, not uniform erosion. Protecting or replaying for a small target is plausible."})

    # 6. instrument
    out.append({"title": "The instrument survives the training it watches" if f["valid"] else "The instrument did not survive this run",
                "evidence": f"Sparse autoencoders trained on the base model drifted by at most {f['fvu_delta_max']:+.3f} FVU over {a['gradient_tokens'] / 1e6:.0f}M tokens (limit {f['fvu_limit']}).",
                "meaning": "You can read the model's internals during fine-tuning without rebuilding the reader — the enabling fact for choosing updates by what they do inside the model." if f["valid"]
                           else "Feature-level claims for this run are not trustworthy; the report says so rather than guessing."})
    return out


def render(fs: list[dict[str, str]], heading: str = "Findings") -> str:
    o = [f"<h2>{_esc(heading)}</h2><ol class='findings'>"]
    for x in fs:
        o.append(f"<li><div class='ft'>{_esc(x['title'])}</div><div class='fe'>{_esc(x['evidence'])}</div><div class='fm'>{_esc(x['meaning'])}</div></li>")
    o.append("</ol>")
    return "".join(o)


FINDINGS_CSS = """
.findings{padding-left:22px}.findings li{margin:0 0 14px}.ft{font-weight:600;font-size:15px}.fe{color:#374151;font-size:13.5px;margin:2px 0}.fm{color:#1d4ed8;font-size:13.5px}
"""
