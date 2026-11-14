"""Build a data *set*: per-domain training stream, held-out perplexity slice, and probe sample.

Layout (repo-relative), for set name S:
  data/S/<domain>/{stream,heldout,probe}.jsonl      raw docs  {"id", "text", "meta"}
  data/S/<domain>/{stream,heldout,probe}.tokens.npy int32 1-D token stream (BOS-prefixed docs,
                                                    EOS-separated), packed by the loader
  data/S/general/{heldout,probe}.*                  general web text (no training stream)
  manifests/S.json                                  sources, seeds, counts, sha256 of every file

Tokenisation is Gemma's tokenizer, BOS at the start of every document (the PT model was trained
that way and the SAEs expect it), EOS after every document. The manifest records the tokenizer
hash so a tokenizer change cannot go unnoticed.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

from clharness.config import REPO_ROOT
from clharness.data.manifest import file_entry, sha256_text, write_manifest
from clharness.data.sources import SMOKE_SOURCES, SourceSpec, take

Doc = dict[str, Any]


@dataclass(frozen=True)
class SplitSizes:
    stream_docs: int
    heldout_docs: int
    probe_docs: int


def write_jsonl(docs: list[Doc], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[Doc]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize_docs(docs: list[Doc], tok: PreTrainedTokenizerBase) -> np.ndarray:
    """Concatenate docs as [BOS] tokens... [EOS] per document into one int32 stream."""
    pieces: list[np.ndarray] = []
    bos, eos = tok.bos_token_id, tok.eos_token_id
    for d in docs:
        ids = tok(d["text"], add_special_tokens=False)["input_ids"]
        pieces.append(np.asarray([bos, *ids, eos], dtype=np.int32))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)


def tokenizer_fingerprint(tok: PreTrainedTokenizerBase) -> dict[str, Any]:
    probe = "def f(x):\n    return x**2  # Ünïcödé ∑ test 12345"
    return {
        "name_or_path": str(tok.name_or_path),
        "vocab_size": int(tok.vocab_size),
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "probe_ids_sha256": sha256_text(json.dumps(tok(probe)["input_ids"])),
    }


def _split_docs(docs: list[Doc], sizes: SplitSizes, seed: int) -> dict[str, list[Doc]]:
    """Deterministic disjoint split. Shuffle with `seed`, then carve heldout, probe, stream."""
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    need = sizes.heldout_docs + sizes.probe_docs + sizes.stream_docs
    if len(docs) < need:
        raise ValueError(f"need {need} docs, have {len(docs)}")
    h = idx[: sizes.heldout_docs]
    p = idx[sizes.heldout_docs : sizes.heldout_docs + sizes.probe_docs]
    s = idx[sizes.heldout_docs + sizes.probe_docs : need]
    return {"heldout": [docs[i] for i in h], "probe": [docs[i] for i in p], "stream": [docs[i] for i in s]}


def build_set(
    set_name: str,
    sources: dict[str, SourceSpec],
    sizes: dict[str, SplitSizes],
    tok: PreTrainedTokenizerBase,
    seed: int,
    seq_len: int,
    skip: int = 0,
) -> dict[str, Any]:
    """Pull docs from every source, split, tokenize, write files + manifest. Returns manifest."""
    out_root = REPO_ROOT / "data" / set_name
    manifest: dict[str, Any] = {
        "set": set_name,
        "seed": seed,
        "seq_len": seq_len,
        "tokenizer": tokenizer_fingerprint(tok),
        "domains": {},
    }
    for domain, spec in sources.items():
        sz = sizes[domain]
        n_docs = sz.stream_docs + sz.heldout_docs + sz.probe_docs
        docs = take(spec, n_docs, skip=skip)
        for j, d in enumerate(docs):
            d["id"] = f"{set_name}/{domain}/{skip + j}"
        parts = _split_docs(docs, sz, seed)
        entry: dict[str, Any] = {"source": spec.to_dict(), "skip": skip, "n_docs_pulled": n_docs, "splits": {}}
        for split, sdocs in parts.items():
            if not sdocs:
                continue
            jsonl = out_root / domain / f"{split}.jsonl"
            npy = out_root / domain / f"{split}.tokens.npy"
            write_jsonl(sdocs, jsonl)
            toks = tokenize_docs(sdocs, tok)
            np.save(npy, toks)
            entry["splits"][split] = {
                "n_docs": len(sdocs),
                "n_tokens": int(toks.shape[0]),
                "doc_ids": [d["id"] for d in sdocs],
                "jsonl": file_entry(jsonl),
                "tokens": file_entry(npy),
            }
        manifest["domains"][domain] = entry
    write_manifest(manifest, REPO_ROOT / "manifests" / f"{set_name}.json")
    return manifest


# Smoke set: small enough to build in ~a minute and to keep the jsonl in git.
SMOKE_SIZES: dict[str, SplitSizes] = {
    "general": SplitSizes(stream_docs=64, heldout_docs=24, probe_docs=24),  # small stream: recoverability re-warm text
    "code": SplitSizes(stream_docs=160, heldout_docs=24, probe_docs=24),
    "biomedical": SplitSizes(stream_docs=160, heldout_docs=24, probe_docs=24),
    "legal": SplitSizes(stream_docs=160, heldout_docs=24, probe_docs=24),
    "math": SplitSizes(stream_docs=160, heldout_docs=24, probe_docs=24),
}


def build_smoke_set(tok: PreTrainedTokenizerBase, seed: int = 1234, seq_len: int = 256) -> dict[str, Any]:
    return build_set("smoke", SMOKE_SOURCES, SMOKE_SIZES, tok, seed=seed, seq_len=seq_len)


# --- real corpora (deliverable 3): token-budgeted, streaming, one domain at a time -------------
# Docs arrive in fixed source order. Each doc is assigned to heldout / probe / stream by a seeded
# RNG (so held-out and probe are unbiased samples of the same distribution as the stream, not the
# first-N docs), until each split's token budget is met. jsonl is written incrementally; tokens
# are accumulated per split and saved once. The manifest stores counts + a hash of the doc-id
# list (full lists for 400k docs would bloat it); the jsonl sha256 covers the content.


@dataclass(frozen=True)
class TokenBudgets:
    stream_tokens: int
    heldout_tokens: int
    probe_tokens: int


REAL_BUDGETS: dict[str, TokenBudgets] = {
    # 200M/domain lets the pilot sweep tokens-per-domain downward without rebuilding.
    # general: 50M stream for Phase 3's "generic replay" baseline; heldout/probe like the rest.
    "general": TokenBudgets(stream_tokens=50_000_000, heldout_tokens=2_000_000, probe_tokens=1_000_000),
    "code": TokenBudgets(stream_tokens=200_000_000, heldout_tokens=2_000_000, probe_tokens=1_000_000),
    "biomedical": TokenBudgets(stream_tokens=200_000_000, heldout_tokens=2_000_000, probe_tokens=1_000_000),
    "legal": TokenBudgets(stream_tokens=200_000_000, heldout_tokens=2_000_000, probe_tokens=1_000_000),
    "math": TokenBudgets(stream_tokens=200_000_000, heldout_tokens=2_000_000, probe_tokens=1_000_000),
}


def _pick_split(rng: random.Random, remaining: dict[str, int]) -> str | None:
    """Sample a split proportional to remaining budget among unfilled splits (None = all full)."""
    open_ = {k: v for k, v in remaining.items() if v > 0}
    if not open_:
        return None
    r = rng.random() * sum(open_.values())
    for k, v in open_.items():
        r -= v
        if r <= 0:
            return k
    return next(iter(open_))


def build_real_domain(
    set_name: str,
    domain: str,
    spec: SourceSpec,
    budgets: TokenBudgets,
    tok: PreTrainedTokenizerBase,
    seed: int,
    seq_len: int,
    tokenize_batch: int = 256,
    log_every_tokens: int = 10_000_000,
) -> dict[str, Any]:
    """Build one domain of a real set; merge its entry into manifests/<set>.json. Returns entry."""
    from clharness.data.sources import iter_source

    out_dir = REPO_ROOT / "data" / set_name / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {"stream": budgets.stream_tokens, "heldout": budgets.heldout_tokens, "probe": budgets.probe_tokens}
    remaining = dict(targets)
    rng = random.Random(f"{seed}:{set_name}:{domain}")
    files = {k: open(out_dir / f"{k}.jsonl", "w", encoding="utf-8") for k in targets if targets[k] > 0}
    pieces: dict[str, list[np.ndarray]] = {k: [] for k in files}
    n_docs = {k: 0 for k in files}
    id_hash = {k: hashlib.sha256() for k in files}
    bos, eos = tok.bos_token_id, tok.eos_token_id
    n_seen = 0
    next_log = log_every_tokens
    t0 = time.time()

    def flush(batch: list[tuple[int, Doc]]) -> None:
        enc = tok([d["text"] for _, d in batch], add_special_tokens=False)["input_ids"]
        for (src_idx, d), ids in zip(batch, enc):
            split = _pick_split(rng, remaining)
            if split is None:
                return
            d["id"] = f"{set_name}/{domain}/{src_idx}"
            arr = np.asarray([bos, *ids, eos], dtype=np.int32)
            pieces[split].append(arr)
            remaining[split] -= len(arr)
            n_docs[split] += 1
            id_hash[split].update(d["id"].encode())
            files[split].write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")

    batch: list[tuple[int, Doc]] = []
    for src_idx, doc in enumerate(iter_source(spec)):
        n_seen += 1
        batch.append((src_idx, doc))
        if len(batch) >= tokenize_batch:
            flush(batch)
            batch = []
            done = sum(targets.values()) - sum(max(v, 0) for v in remaining.values())
            if done >= next_log:
                print(f"  {domain}: {done/1e6:.0f}M tokens, {n_seen} docs seen, {time.time()-t0:.0f}s", flush=True)
                next_log += log_every_tokens
            if all(v <= 0 for v in remaining.values()):
                break
    if batch and any(v > 0 for v in remaining.values()):
        flush(batch)
    for f in files.values():
        f.close()
    short = {k: v for k, v in remaining.items() if v > 0}
    if short:
        raise RuntimeError(f"{domain}: source exhausted with budgets unmet: {short}")

    entry: dict[str, Any] = {"source": spec.to_dict(), "n_docs_seen": n_seen, "budgets": targets, "splits": {}}
    for split in files:
        toks = np.concatenate(pieces[split])
        npy = out_dir / f"{split}.tokens.npy"
        np.save(npy, toks)
        entry["splits"][split] = {
            "n_docs": n_docs[split],
            "n_tokens": int(toks.shape[0]),
            "doc_ids_sha256": id_hash[split].hexdigest(),
            "jsonl": file_entry(out_dir / f"{split}.jsonl"),
            "tokens": file_entry(npy),
        }
    man_path = REPO_ROOT / "manifests" / f"{set_name}.json"
    manifest = json.load(open(man_path)) if man_path.exists() else {
        "set": set_name, "seed": seed, "seq_len": seq_len, "tokenizer": tokenizer_fingerprint(tok), "domains": {},
    }
    if manifest["seed"] != seed or manifest["seq_len"] != seq_len:
        raise ValueError(f"manifest {man_path} was built with seed={manifest['seed']} seq_len={manifest['seq_len']}")
    manifest["domains"][domain] = entry
    write_manifest(manifest, man_path)
    print(f"  {domain}: done in {time.time()-t0:.0f}s; " + ", ".join(
        f"{k}={v['n_tokens']/1e6:.1f}M tok/{v['n_docs']} docs" for k, v in entry["splits"].items()))
    return entry
