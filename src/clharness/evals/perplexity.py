"""Held-out perplexity on fixed token slices. Cheap, continuous, run at every checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel


@dataclass
class PerplexityResult:
    name: str
    n_seqs: int
    n_tokens: int  # predicted tokens (positions 1..T-1)
    nll: float  # mean per-token negative log-likelihood (nats)
    ppl: float

    def to_dict(self) -> dict:
        return {"n_seqs": self.n_seqs, "n_tokens": self.n_tokens, "nll": self.nll, "ppl": self.ppl}


@torch.no_grad()
def perplexity_on_chunks(
    model: PreTrainedModel, chunks: torch.Tensor, batch_size: int, device: str, name: str
) -> PerplexityResult:
    """chunks: [n, seq_len] int64, no padding. Loss over positions 1..seq_len-1 (BOS never predicted)."""
    model.eval()
    total_nll = 0.0
    total_tok = 0
    for i in range(0, chunks.shape[0], batch_size):
        ids = chunks[i : i + batch_size].to(device)
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
        tgt = ids[:, 1:]
        flat, ft = logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)
        for j in range(0, flat.shape[0], 1024):  # chunked fp32 softmax (262k vocab)
            total_nll += float(torch.nn.functional.cross_entropy(flat[j : j + 1024].float(), ft[j : j + 1024], reduction="sum"))
        total_tok += tgt.numel()
    mean = total_nll / total_tok
    return PerplexityResult(name=name, n_seqs=int(chunks.shape[0]), n_tokens=total_tok, nll=mean, ppl=math.exp(mean))


@torch.no_grad()
def perplexity_on_docs(
    model: PreTrainedModel, docs: torch.Tensor, doc_mask: torch.Tensor, batch_size: int, device: str, name: str
) -> PerplexityResult:
    """Per-document perplexity: docs [n, seq_len] int64 with BOS at position 0, right-padded;
    doc_mask [n, seq_len] bool marks real tokens. Loss over real positions 1..len-1. Unlike the
    packed variant this does not depend on the stream's packing format (research-notes T16)."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, docs.shape[0], batch_size):
        ids = docs[i : i + batch_size].to(device)
        am = doc_mask[i : i + batch_size].to(device)
        logits = model(input_ids=ids, attention_mask=am.long(), use_cache=False).logits[:, :-1]
        tgt = ids[:, 1:]
        valid = am[:, 1:]
        flat, ft, fv = logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), valid.reshape(-1)
        for j in range(0, flat.shape[0], 1024):
            l = torch.nn.functional.cross_entropy(flat[j : j + 1024].float(), ft[j : j + 1024], reduction="none")
            total_nll += float((l * fv[j : j + 1024]).sum())
        total_tok += int(valid.sum())
    mean = total_nll / max(total_tok, 1)
    return PerplexityResult(name=name, n_seqs=int(docs.shape[0]), n_tokens=total_tok, nll=mean, ppl=math.exp(mean))


def docs_from_jsonl(path, tokenizer, seq_len: int, max_docs: int, min_tokens: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """First `max_docs` held-out documents with >= min_tokens tokens, as [n, seq_len] right-padded
    tensors (BOS at 0, truncated to seq_len) plus a bool mask. Deterministic in file order."""
    import json

    rows, masks = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ids = tokenizer(json.loads(line)["text"], add_special_tokens=False)["input_ids"][: seq_len - 1]
            if len(ids) < min_tokens:
                continue
            seq = [tokenizer.bos_token_id] + ids
            pad = seq_len - len(seq)
            rows.append(seq + [tokenizer.pad_token_id or 0] * pad)
            masks.append([True] * len(seq) + [False] * pad)
            if len(rows) >= max_docs:
                break
    return torch.tensor(rows, dtype=torch.int64), torch.tensor(masks, dtype=torch.bool)
