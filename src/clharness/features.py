"""Domain-resolved SAE feature analysis: which features fire on which domain's text, how each
domain's feature set moves through the training stages, and what the most-decayed features
look like (top-activating snippets from the probe set, so features get names without an
external labelling service).

Two parts:
  `domain_pass(run, ckpts)`  — Mac-friendly forward passes over the probe set, per domain, at the
      base model and saved checkpoints. Writes results/runs/<run>/features_by_domain.npz
      (freq[ckpt, domain, feature] per layer) and feature_contexts_L<L>.json (top-3 snippets per
      feature at the base model).
  `domain_features(run)`     — the analysis: selectivity at base, per-set trajectories across all
      checkpoints (from the per-checkpoint pooled features_*.npz the run already saved), stage
      table at the boundaries, labelled examples. Returns a dict the report/demo render.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clharness.config import REPO_ROOT, RunConfig

PRIMARY_TAGS = {"L7": 7, "L13": 13, "L22": 22}


def domain_pass(run: str, ckpts: list[int], max_seqs: int = 128, top_ctx: int = 3, device: str | None = None) -> Path:
    from clharness.activations import ResidualCapture
    from clharness.data.loader import fixed_chunks, load_tokens
    from clharness.data.manifest import read_manifest
    from clharness.env import pick_device, seed_everything
    from clharness.model import load_model
    from clharness.sae import SAESpec, load_sae
    from clharness.train.loop import autocast_ctx

    run_dir = REPO_ROOT / "results" / "runs" / run
    cfg = RunConfig.model_validate(json.load(open(run_dir / "config.resolved.json")))
    man = read_manifest(cfg.data.manifest)
    seed_everything(cfg.train.seed)
    dev = pick_device(device or "auto")
    lm = load_model(cfg.model.model_id, device=dev, dtype=cfg.model.dtype)
    lm.model.eval()
    ac = autocast_ctx(dev, cfg.train.precision)
    layers = sorted(set(cfg.model.capture_layers) & set(PRIMARY_TAGS.values()))
    saes = {}
    for sc in cfg.model.saes:  # primary SAE per layer = first listed
        if sc.layer in layers:
            saes.setdefault(sc.layer, load_sae(SAESpec(sc.release, sc.sae_id, sc.layer), dev))
    # probe set per domain (same chunks the run's ProbeHook used, first max_seqs of them)
    probe: dict[str, torch.Tensor] = {}
    for dom, entry in man["domains"].items():
        if cfg.data.probe_set in entry["splits"]:
            probe[dom] = fixed_chunks(load_tokens(entry["splits"][cfg.data.probe_set]["tokens"]["path"]), cfg.data.seq_len,
                                      min(max_seqs, cfg.data.probe_max_seqs_per_domain))
    doms = list(probe)
    bos = lm.tokenizer.bos_token_id
    bs = min(cfg.train.probe_batch_size, 4) if dev.startswith("mps") else cfg.train.probe_batch_size
    base_sd = {k: v.detach().clone() for k, v in lm.model.state_dict().items()} if 0 in ckpts else None
    d_sae = {L: saes[L].cfg.d_sae for L in layers}
    freq = {L: np.zeros((len(ckpts), len(doms), d_sae[L]), dtype=np.float32) for L in layers}
    ctx_store: dict[int, dict[str, torch.Tensor]] = {}
    for ci, i in enumerate(ckpts):
        ck = run_dir / "checkpoints" / f"ckpt_{i:04d}"
        if i == 0:
            lm.model.load_state_dict(base_sd)
        else:
            wf = ck / "model.bf16.pt" if (ck / "model.bf16.pt").exists() else ck / "model.pt"
            lm.model.load_state_dict(torch.load(wf, map_location=dev), strict=True)
        fp = float(lm.model.state_dict()["model.layers.13.mlp.down_proj.weight"].float().sum())
        t0 = time.time()
        if i == 0 and top_ctx:
            ctx_store = {L: {"val": torch.full((top_ctx, d_sae[L]), -1.0), "pos": torch.full((top_ctx, d_sae[L]), -1, dtype=torch.long)} for L in layers}
        with torch.no_grad(), ResidualCapture(lm.model, layers) as cap:
            for di, dom in enumerate(doms):
                chunks = probe[dom]
                count = {L: torch.zeros(d_sae[L], dtype=torch.float64) for L in layers}
                n_tok = 0
                for s in range(0, chunks.shape[0], bs):
                    ids = chunks[s : s + bs].to(dev)
                    with ac():
                        lm.model(input_ids=ids, use_cache=False)
                    mask = torch.ones_like(ids, dtype=torch.bool)
                    if cfg.model.exclude_bos_from_stats:
                        mask &= ids != bos
                    flat_mask = mask.reshape(-1)
                    n_tok += int(flat_mask.sum())
                    tok_idx = torch.arange(ids.numel(), device=ids.device)[flat_mask]
                    for L in layers:
                        x_all = cap.acts[L].reshape(-1, cap.acts[L].shape[-1])[flat_mask]
                        for c0 in range(0, x_all.shape[0], 1024):  # bounded encode: [1024, d_sae] fp32 at a time
                            x = x_all[c0 : c0 + 1024].to(saes[L].device, torch.float32)
                            feats = saes[L].encode(x)  # [n, d_sae]
                            count[L] += (feats > 0).sum(0).cpu().double()
                            if i == 0 and top_ctx:
                                k = min(top_ctx, feats.shape[0])
                                v, idx = feats.topk(k, dim=0)  # [k, d_sae]
                                # global token position: (domain, sequence, position) packed as one int
                                gpos = (di * chunks.shape[0] + s) * chunks.shape[1] + tok_idx[c0 : c0 + 1024][idx.reshape(-1)].reshape(idx.shape)
                                st = ctx_store[L]
                                allv = torch.cat([st["val"], v.cpu()]); allp = torch.cat([st["pos"], gpos.cpu()])
                                tv, ti = allv.topk(top_ctx, dim=0)
                                st["val"], st["pos"] = tv, torch.gather(allp, 0, ti)
                            del feats, x
                    cap.clear()
                for L in layers:
                    freq[L][ci, di] = (count[L] / max(n_tok, 1)).numpy()
        print(f"ckpt {i:2d} fingerprint {fp:.4f}: {len(doms)} domains x {probe[doms[0]].shape[0]} seqs in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(run_dir / "features_by_domain.npz", ckpts=np.array(ckpts), domains=np.array(doms),
                        **{f"freq_L{L}": freq[L] for L in layers})
    if ctx_store:
        all_tokens = torch.cat([probe[d].reshape(-1) for d in doms])
        seq_len = cfg.data.seq_len
        for L in layers:
            st = ctx_store[L]
            out: dict[str, list[dict[str, Any]]] = {}
            for f in range(d_sae[L]):
                entries = []
                for r in range(top_ctx):
                    p = int(st["pos"][r, f]); v = float(st["val"][r, f])
                    if p < 0 or v <= 0:
                        continue
                    lo = max(p - 12, (p // seq_len) * seq_len); hi = min(p + 4, (p // seq_len + 1) * seq_len)
                    before = lm.tokenizer.decode(all_tokens[lo:p].tolist())
                    tok = lm.tokenizer.decode(all_tokens[p:p + 1].tolist())
                    after = lm.tokenizer.decode(all_tokens[p + 1:hi].tolist())
                    entries.append({"act": round(v, 2), "domain": doms[p // (seq_len * probe[doms[0]].shape[0])], "before": before, "token": tok, "after": after})
                if entries:
                    out[str(f)] = entries
            json.dump(out, open(run_dir / f"feature_contexts_L{L}.json", "w"))
    return run_dir / "features_by_domain.npz"


def _pooled_freq(run_dir: Path, tag: str) -> tuple[list[int], np.ndarray]:
    files = sorted(glob.glob(str(run_dir / "checkpoints" / "ckpt_*" / f"features_{tag}.npz")))
    idx = [int(Path(f).parent.name.split("_")[1]) for f in files]
    return idx, np.stack([np.load(f)["freq"].astype(np.float64) for f in files])


def domain_features(run: str, min_freq: float = 1e-4, sel_share: float = 0.5, n_examples: int = 4) -> dict[str, Any]:
    run_dir = REPO_ROOT / "results" / "runs" / run
    z = np.load(run_dir / "features_by_domain.npz")
    ckpts = [int(x) for x in z["ckpts"]]
    doms = [str(x) for x in z["domains"]]
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    ck_rows = {int(r["ckpt/index"]): r for r in rows if "ckpt/index" in r}
    cfg = json.load(open(run_dir / "config.resolved.json"))
    train_doms = [d["name"] for d in cfg["data"]["domains"]]
    out: dict[str, Any] = {"ckpts": ckpts, "domains": doms, "layers": {}}
    for key in z.files:
        if not key.startswith("freq_L"):
            continue
        L = int(key[6:])
        F = z[key].astype(np.float64)  # [ckpt, dom, feat]
        f0 = F[0]
        alive = f0.max(0) >= min_freq
        share = f0 / np.maximum(f0.sum(0, keepdims=True), 1e-12)
        top = share.argmax(0)
        selective = alive & (share.max(0) >= sel_share)
        sets = {d: np.where(selective & (top == i))[0] for i, d in enumerate(doms)}
        sets["shared"] = np.where(alive & ~selective)[0]
        # trajectories over ALL checkpoints from the pooled per-checkpoint stats, drift-corrected
        idx, P = _pooled_freq(run_dir, f"L{L}")
        ratio = P / np.maximum(P[0], 1e-9)
        med = np.array([np.nanmedian(ratio[k][P[0] > min_freq]) for k in range(len(idx))])
        traj = {}
        for name, fs in sets.items():
            if fs.size == 0:
                continue
            traj[name] = [float(np.median(ratio[k, fs] / med[k])) for k in range(len(idx))]
        # stage table at the boundary checkpoints: each set measured on its OWN domain's text
        stage = {}
        for ci, c in enumerate(ckpts):
            row = {}
            for i, d in enumerate(doms):
                fs = sets[d]
                if fs.size == 0:
                    continue
                own = F[ci, i, fs] / np.maximum(f0[i, fs], 1e-9)
                al = f0[i] > min_freq
                gmed = np.median(F[ci, i][al] / f0[i][al])
                own_c = own / max(gmed, 1e-9)
                tot = np.maximum(F[ci][:, fs].sum(0), 1e-12)
                row[d] = {"n": int(fs.size), "median_ratio": float(np.median(own_c)), "median_ratio_raw": float(np.median(own)),
                          "global_median_ratio": float(gmed), "total_firing_ratio": float(F[ci, i].sum() / f0[i].sum()),
                          "selectivity": float((F[ci][i, fs] / tot).mean()),
                          "frac_decayed": float((own_c < 0.5).mean()), "frac_grown": float((own_c > 2.0).mean())}
            r = ck_rows.get(c, {})
            di = int(r.get("ckpt/domain_index", -1))
            stage[str(c)] = {"label": "base" if c == 0 else f"after {train_doms[di]}" if 0 <= di < len(train_doms) else f"ckpt {c}", "sets": row}
        # labelled examples: most decayed (own-domain, end vs base) per set
        ctx_path = run_dir / f"feature_contexts_L{L}.json"
        contexts = json.load(open(ctx_path)) if ctx_path.exists() else {}
        examples = {}
        for i, d in enumerate(doms):
            fs = sets[d]
            if fs.size == 0:
                continue
            own_end = F[-1, i, fs] / np.maximum(f0[i, fs], 1e-9)
            gmed = np.median(F[-1, i][f0[i] > min_freq] / f0[i][f0[i] > min_freq])
            own_c = own_end / max(gmed, 1e-9)
            order = np.argsort(own_c)
            ex, structural = [], []
            for j in order:
                f = int(fs[j])
                cx = contexts.get(str(f), [])[:2]
                entry = {"feature": f, "freq_base": float(f0[i, f]), "ratio_end": float(own_c[j]), "contexts": cx}
                tok = (cx[0]["token"] if cx else "").strip()
                # content features first; whitespace / end-of-text / punctuation-only features are format features
                (ex if any(ch.isalnum() for ch in tok) else structural).append(entry)
                if len(ex) >= n_examples and len(structural) >= n_examples:
                    break
            examples[d] = ex[:n_examples]
            examples[d + "_structural"] = structural[:n_examples]
        out["layers"][str(L)] = {"n_alive": int(alive.sum()), "set_sizes": {k: int(v.size) for k, v in sets.items()},
                                 "pooled_ckpts": idx, "trajectory": traj, "stage": stage, "examples": examples}
    json.dump(out, open(run_dir / "domain_features.json", "w"), indent=1)
    return out
