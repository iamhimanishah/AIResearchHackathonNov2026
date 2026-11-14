"""Per-run analysis: turn a run directory into one JSON summary (and a short text report).

Reads only harness output (metrics.jsonl, checkpoints/*/{meta,metrics}.json, items_*.npy,
features_*.npz) and produces `results/runs/<run>/analysis.json` with:
  checkpoints      per-checkpoint: tokens, domain, tier-1 scores, per-document and packed ppl,
                   raw + chance-adjusted flips (Harmon et al. 2025), instrument health, similarity
  boundaries       recoverability before/after per metric at each domain boundary
  retention        end-vs-base and end-vs-post-adaptation deltas per metric
  features         per SAE tag: decay concentration, counts of strongly decayed / grown features,
                   top-k decayed and grown feature ids with base/end frequency, both raw and
                   drift-corrected (frequency ratio divided by the global median ratio, so a
                   uniform rise in firing rate — residual norms grow during training — does not
                   mask decay)
The demo/report page and the Phase 1 writeup read this file; nothing here needs a GPU.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from clharness.config import REPO_ROOT

MC_TASKS = ["arc_easy/acc_norm", "hellaswag/acc_norm", "piqa/acc_norm", "pubmedqa_pqal/acc"]
GEN_TASKS = ["mbpp/pass_at_1"]
DOMAINS = ["general", "code", "biomedical", "legal", "math"]


def _load(run_dir: Path):
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    ck = [r for r in rows if "ckpt/index" in r]
    metas = [json.load(open(p)) for p in sorted(glob.glob(str(run_dir / "checkpoints" / "*" / "meta.json")))]
    return rows, ck, metas


def _flips(run_dir: Path, task: str, k: float, i_base: int, i_cur: int) -> dict[str, float] | None:
    fb = run_dir / "checkpoints" / f"ckpt_{i_base:04d}" / f"items_{task}.npy"
    fc = run_dir / "checkpoints" / f"ckpt_{i_cur:04d}" / f"items_{task}.npy"
    if not (fb.exists() and fc.exists()):
        return None
    b, c = np.load(fb), np.load(fc)
    ok = ~(np.isnan(b) | np.isnan(c))
    n = max(int(ok.sum()), 1)
    f10 = float(((b > 0.5) & (c < 0.5) & ok).sum()) / n
    f01 = float(((b < 0.5) & (c > 0.5) & ok).sum()) / n
    a_pre, a_post = float((b[ok] > 0.5).mean()), float((c[ok] > 0.5).mean())
    chance = (1 - a_pre) * (1 - a_post) / (k - 1) if k and k > 1 else 0.0
    return {"f10": f10, "f01": f01, "f10_adj": max(f10 - chance, 0.0), "f01_adj": max(f01 - chance, 0.0), "net": f01 - f10}


DEGENERATE_SHARE = 0.90  # ≥ 90% of items answered with the same option index = alarm


def _answer_stats(run_dir: Path, task: str, i: int) -> dict[str, Any] | None:
    """Distribution of the option the model picked. A benchmark whose score sits at the majority
    rate while every answer is the same option has stopped reading the question (seen on
    PubMedQA through the legal domain of the 200M run); accuracy alone does not show it."""
    fc = run_dir / "checkpoints" / f"ckpt_{i:04d}" / f"choices_{task}.npy"
    fg = run_dir / "checkpoints" / f"ckpt_{i:04d}" / f"gold_{task}.npy"
    if not (fc.exists() and fg.exists()):
        return None
    ch, gd = np.load(fc).astype(int), np.load(fg).astype(int)
    ok = (ch >= 0) & (gd >= 0)
    if not ok.any():
        return None
    ch, gd = ch[ok], gd[ok]
    counts = np.bincount(ch, minlength=int(max(ch.max(), gd.max())) + 1).astype(float)
    p = counts / counts.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    gcounts = np.bincount(gd, minlength=counts.size).astype(float)
    return {
        "share_max": float(p.max()), "mode_option": int(p.argmax()), "entropy_nats": ent,
        "gold_share_max": float(gcounts.max() / gcounts.sum()),
        "degenerate": bool(p.max() >= DEGENERATE_SHARE),  # report also flags a +0.25 rise over base
    }


def _feature_summary(run_dir: Path, tag: str, i_end: int, top_k: int = 25) -> dict[str, Any] | None:
    p0 = run_dir / "checkpoints" / "ckpt_0000" / f"features_{tag}.npz"
    pe = run_dir / "checkpoints" / f"ckpt_{i_end:04d}" / f"features_{tag}.npz"
    if not (p0.exists() and pe.exists()):
        return None
    z0, ze = np.load(p0), np.load(pe)
    f0, fe = z0["freq"].astype(np.float64), ze["freq"].astype(np.float64)
    alive = f0 > 1e-6
    ratio = np.where(alive, fe / np.maximum(f0, 1e-9), np.nan)
    global_ratio = float(np.nanmedian(ratio))  # uniform firing-rate drift; > 1 = model fires more overall
    corrected = ratio / global_ratio
    dec_raw = np.where(alive, f0 - fe, 0.0)
    dec_c = np.where(alive, f0 * (1 - np.minimum(corrected, 1e9)), 0.0)  # drift-corrected lost frequency

    def conc(d: np.ndarray) -> dict[str, float]:
        pos = np.sort(d[d > 0])[::-1]
        if pos.size == 0:
            return {"frac_decayed": 0.0, "top1pct_share": 0.0, "top10pct_share": 0.0}
        return {
            "frac_decayed": float((d > 0).sum() / alive.sum()),
            "top1pct_share": float(pos[: max(1, pos.size // 100)].sum() / pos.sum()),
            "top10pct_share": float(pos[: max(1, pos.size // 10)].sum() / pos.sum()),
        }

    idx_dec = np.argsort(np.where(alive, corrected, np.inf))[:top_k]
    idx_grow = np.argsort(np.where(alive, -corrected, np.inf))[:top_k]
    return {
        "n_alive_base": int(alive.sum()),
        "global_freq_ratio_median": global_ratio,
        "raw": conc(dec_raw),
        "drift_corrected": {**conc(dec_c), "n_over_50pct_decay": int(((corrected < 0.5) & alive).sum()),
                            "n_over_2x_growth": int(((corrected > 2.0) & alive).sum())},
        "near_thr_base": float(z0["near_thr"].mean()), "near_thr_end": float(ze["near_thr"].mean()),
        "top_decayed": [{"feature": int(i), "freq_base": float(f0[i]), "freq_end": float(fe[i]), "ratio_corrected": float(corrected[i])} for i in idx_dec],
        "top_grown": [{"feature": int(i), "freq_base": float(f0[i]), "freq_end": float(fe[i]), "ratio_corrected": float(corrected[i])} for i in idx_grow],
    }


def analyze_run(run: str, post_adapt_ckpt: int = 3) -> dict[str, Any]:
    run_dir = REPO_ROOT / "results" / "runs" / run
    rows, ck, metas = _load(run_dir)
    manifest = json.load(open(REPO_ROOT / "manifests" / "eval-protected-v1.json"))["tasks"]
    kmap = {t: (1.0 / manifest[t]["chance_uniform"] if manifest.get(t, {}).get("chance_uniform") else 0.0) for t in manifest}
    end = len(ck) - 1
    checkpoints = []
    for i, r in enumerate(ck):
        row: dict[str, Any] = {
            "index": i, "step": r["step"], "gradient_tokens": r["ckpt/gradient_tokens"],
            "domain_index": r["ckpt/domain_index"], "is_boundary": bool(r["ckpt/is_boundary"]),
            "scores": {t: r.get(f"eval/lm_eval/{t}") for t in MC_TASKS + GEN_TASKS if f"eval/lm_eval/{t}" in r},
            "ppl": {d: r.get(f"eval/ppl/{d}") for d in DOMAINS if f"eval/ppl/{d}" in r},
            "ppl_doc": {d: r.get(f"eval/ppl_doc/{d}") for d in DOMAINS if f"eval/ppl_doc/{d}" in r},
            "flips": {t.split("/")[0]: _flips(run_dir, t.split("/")[0], kmap.get(t.split("/")[0], 0.0), 0, i) for t in MC_TASKS + GEN_TASKS},
            "answers": {t.split("/")[0]: _answer_stats(run_dir, t.split("/")[0], i) for t in MC_TASKS},
            "instrument": {L: {k: r.get(f"probe/L{L}/{k}") for k in ("fvu", "fvu_delta", "valid", "cka_vs_base", "cka_std", "cka_nomassive", "mknn", "readout_base_probe_acc", "readout_fresh_probe_acc", "massive0_mag_rel", "near_thr_mean")} for L in (7, 13, 22)},
        }
        if r.get("recover/tokens_used"):
            row["recover"] = {k[len("recover/"):]: v for k, v in r.items() if k.startswith("recover/") and "wall" not in k}
        checkpoints.append(row)
    b, e, pa = ck[0], ck[end], ck[min(post_adapt_ckpt, end)]
    retention = {
        "mc_mean_delta_pts": float(np.mean([(e[f"eval/lm_eval/{t}"] - b[f"eval/lm_eval/{t}"]) * 100 for t in MC_TASKS])),
        "per_task_delta_pts": {t: (e[f"eval/lm_eval/{t}"] - b[f"eval/lm_eval/{t}"]) * 100 for t in MC_TASKS + GEN_TASKS if f"eval/lm_eval/{t}" in e},
        "ppl_packed_end_over_postadapt": {d: e[f"eval/ppl/{d}"] / pa[f"eval/ppl/{d}"] for d in DOMAINS if f"eval/ppl/{d}" in e},
        "ppl_doc_end_over_base": {d: e[f"eval/ppl_doc/{d}"] / b[f"eval/ppl_doc/{d}"] for d in DOMAINS if f"eval/ppl_doc/{d}" in e},
    }
    tags = sorted({Path(p).stem[len("features_"):] for p in glob.glob(str(run_dir / "checkpoints" / "ckpt_0000" / "features_*.npz"))})
    features = {tag: _feature_summary(run_dir, tag, end) for tag in tags}
    tr = [r for r in rows if "train/loss" in r]
    out = {
        "run": run, "n_checkpoints": len(ck), "gradient_tokens": e["ckpt/gradient_tokens"],
        "anomalous_grad_steps": tr[-1]["train/anomalous_grad_steps"] if tr else None,
        "config": json.load(open(run_dir / "config.resolved.json")),
        "checkpoints": checkpoints, "retention": retention, "features": features,
    }
    json.dump(out, open(run_dir / "analysis.json", "w"), indent=1, default=float)
    return out


def report(a: dict[str, Any]) -> str:
    L = [f"# {a['run']}: {a['n_checkpoints']} ckpts, {a['gradient_tokens']/1e6:.0f}M tokens, lr {a['config']['optim']['lr']}, anomalous {a['anomalous_grad_steps']}"]
    r = a["retention"]
    L.append(f"MC mean Δ {r['mc_mean_delta_pts']:+.2f} pts | " + " ".join(f"{t.split('/')[0]}:{v:+.1f}" for t, v in r["per_task_delta_pts"].items()))
    L.append("packed ppl end/post-adapt: " + " ".join(f"{d}:{v:.3f}" for d, v in r["ppl_packed_end_over_postadapt"].items()))
    if r["ppl_doc_end_over_base"]:
        L.append("per-doc ppl end/base:      " + " ".join(f"{d}:{v:.3f}" for d, v in r["ppl_doc_end_over_base"].items()))
    e = a["checkpoints"][-1]
    L.append("end flips (adj 1->0 / 0->1): " + " ".join(f"{t}:{f['f10_adj']*100:.1f}/{f['f01_adj']*100:.1f}" for t, f in e["flips"].items() if f))
    for c in a["checkpoints"]:
        if c["is_boundary"] and c["index"] > 0 and "recover" in c:
            rec = c["recover"]
            L.append(f"boundary ckpt {c['index']} recover: " + " ".join(f"{t.split('/')[0][:6]} {c['scores'][t]:.3f}->{rec.get('lm_eval/'+t, float('nan')):.3f}" for t in MC_TASKS + GEN_TASKS if t in c["scores"]))
    L.append("instrument end L13: " + " ".join(f"{k}={v:.3f}" for k, v in e["instrument"][13].items() if v is not None))
    for tag, f in a["features"].items():
        if f:
            dc = f["drift_corrected"]
            L.append(f"features {tag}: global ratio {f['global_freq_ratio_median']:.2f}; drift-corrected: decayed {dc['frac_decayed']*100:.0f}% of {f['n_alive_base']}, top1% carry {dc['top1pct_share']:.2f}, top10% {dc['top10pct_share']:.2f}, >50% decay {dc['n_over_50pct_decay']}, >2x growth {dc['n_over_2x_growth']}")
    return "\n".join(L)
