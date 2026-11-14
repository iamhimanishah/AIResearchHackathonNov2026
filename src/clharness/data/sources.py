"""Document sources: each yields dicts {"text": str, "meta": {...}} in a deterministic order.

Deterministic order matters: manifests record (source, skip, take), and rebuilding must produce
identical bytes. Sources therefore never shuffle; sampling is done downstream with a seed.

SMOKE_SOURCES (deliverable 1) stream the first few hundred docs; REAL_SOURCES (deliverable 3) read
fixed shard lists at pinned revisions. Datasets whose HF repo
is a loading *script* (pile-of-law, ncbi/pubmed, casehold) cannot be loaded by `datasets>=4`;
we read their raw files directly instead.
"""

from __future__ import annotations

import json
import lzma
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

from huggingface_hub import hf_hub_download

from clharness.env import hf_token

Doc = dict[str, Any]


@dataclass(frozen=True)
class SourceSpec:
    """Serialisable description of a source; goes into the manifest verbatim."""

    name: str
    kind: str  # hf_streaming | hf_file_jsonl_xz | hf_file_parquet
    repo_id: str
    text_field: str
    config: str | None = None
    split: str = "train"
    data_files: list[str] = field(default_factory=list)
    revision: str | None = None  # pin for reproducibility once chosen
    filters: dict[str, Any] = field(default_factory=dict)
    min_chars: int = 200
    max_chars: int | None = None  # truncate long docs (recorded in manifest)
    strip_header_re: str | None = None  # regex removed from the start of every doc (e.g. StarCoder <reponame> lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "repo_id": self.repo_id,
            "text_field": self.text_field,
            "config": self.config,
            "split": self.split,
            "data_files": list(self.data_files),
            "revision": self.revision,
            "filters": dict(self.filters),
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "strip_header_re": self.strip_header_re,
        }


def _clip(text: str, spec: "SourceSpec") -> str:
    if spec.strip_header_re is not None:
        text = re.sub(spec.strip_header_re, "", text, count=1)
    return text if spec.max_chars is None else text[: spec.max_chars]


def _passes(row: dict, filters: dict[str, Any]) -> bool:
    return all(row.get(k) == v for k, v in filters.items())


def _iter_hf_streaming(spec: SourceSpec) -> Iterator[Doc]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = dict(split=spec.split, streaming=True, token=hf_token())
    if spec.config:
        kwargs["name"] = spec.config
    if spec.data_files:
        kwargs["data_files"] = list(spec.data_files)
    if spec.revision:
        kwargs["revision"] = spec.revision
    ds = load_dataset(spec.repo_id, **kwargs)
    for i, row in enumerate(ds):
        if not _passes(row, spec.filters):
            continue
        text = row[spec.text_field]
        if not isinstance(text, str) or len(text) < spec.min_chars:
            continue
        meta = {k: row[k] for k in row if k != spec.text_field and isinstance(row[k], (str, int, float))}
        meta["row"] = i
        yield {"text": _clip(text, spec), "meta": meta}


def _iter_hf_file_jsonl_xz(spec: SourceSpec) -> Iterator[Doc]:
    for fname in spec.data_files:
        local = hf_hub_download(spec.repo_id, fname, repo_type="dataset", revision=spec.revision, token=hf_token())
        with lzma.open(local, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                if not _passes(row, spec.filters):
                    continue
                text = row[spec.text_field]
                if not isinstance(text, str) or len(text) < spec.min_chars:
                    continue
                yield {"text": _clip(text, spec), "meta": {"file": fname, "row": i}}


def _iter_hf_file_parquet(spec: SourceSpec) -> Iterator[Doc]:
    import pyarrow.parquet as pq

    for fname in spec.data_files:
        local = hf_hub_download(spec.repo_id, fname, repo_type="dataset", revision=spec.revision, token=hf_token())
        pf = pq.ParquetFile(local)
        i = 0
        for batch in pf.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                idx, i = i, i + 1
                if not _passes(row, spec.filters):
                    continue
                text = row[spec.text_field]
                if not isinstance(text, str) or len(text) < spec.min_chars:
                    continue
                yield {"text": _clip(text, spec), "meta": {"file": fname, "row": idx}}


def _iter_hf_file_jsonl(spec: SourceSpec) -> Iterator[Doc]:
    for fname in spec.data_files:
        local = hf_hub_download(spec.repo_id, fname, repo_type="dataset", revision=spec.revision, token=hf_token())
        with open(local, encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                if not _passes(row, spec.filters):
                    continue
                text = row[spec.text_field]
                if not isinstance(text, str) or len(text) < spec.min_chars:
                    continue
                yield {"text": _clip(text, spec), "meta": {"file": fname, "row": i}}


_KINDS: dict[str, Callable[[SourceSpec], Iterator[Doc]]] = {
    "hf_streaming": _iter_hf_streaming,
    "hf_file_jsonl_xz": _iter_hf_file_jsonl_xz,
    "hf_file_jsonl": _iter_hf_file_jsonl,
    "hf_file_parquet": _iter_hf_file_parquet,
}


def iter_source(spec: SourceSpec) -> Iterator[Doc]:
    return _KINDS[spec.kind](spec)


def take(spec: SourceSpec, n: int, skip: int = 0) -> list[Doc]:
    out: list[Doc] = []
    for i, d in enumerate(iter_source(spec)):
        if i < skip:
            continue
        out.append(d)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"{spec.name}: wanted {n} docs after skip {skip}, got {len(out)}")
    return out


# --- registry -------------------------------------------------------------------------------
# Smoke-scale sources. Real-corpus specs (with pinned revisions) are declared in deliverable 3.

SMOKE_SOURCES: dict[str, SourceSpec] = {
    "general": SourceSpec(
        name="fineweb-sample-10BT",
        kind="hf_streaming",
        repo_id="HuggingFaceFW/fineweb",
        config="sample-10BT",
        text_field="text",
    ),
    # The Stack variants are gated behind a terms click; for the smoke fixture use codeparrot's
    # ungated validation shard (mixed languages). Real code corpus is chosen in deliverable 3.
    "code": SourceSpec(
        name="codeparrot-clean-valid",
        kind="hf_streaming",
        repo_id="codeparrot/codeparrot-clean-valid",
        text_field="content",
        max_chars=12000,
    ),
    "biomedical": SourceSpec(
        name="medical-abstracts",
        kind="hf_file_parquet",
        repo_id="TimSchopf/medical_abstracts",
        data_files=["data/train-00000-of-00001.parquet"],
        text_field="medical_abstract",
    ),
    "legal": SourceSpec(
        name="pile-of-law-small-subsets",
        kind="hf_file_jsonl_xz",
        repo_id="pile-of-law/pile-of-law",
        data_files=[
            "data/train.ftc_advisory_opinions.jsonl.xz",
            "data/train.examoutlines.jsonl.xz",
            "data/train.tos.jsonl.xz",
            "data/train.ed_policy_guidance.jsonl.xz",
            "data/train.medicaid_policy_guidance.jsonl.xz",
        ],
        text_field="text",
        max_chars=12000,
    ),
    "math": SourceSpec(
        name="open-web-math",
        kind="hf_streaming",
        repo_id="open-web-math/open-web-math",
        text_field="text",
        max_chars=12000,
    ),
}


# --- real corpora (deliverable 3) --------------------------------------------------------------
# Every source is a fixed list of files at a pinned repo revision, read in file order: rebuilding
# yields identical bytes. Shards are chosen to spread over each corpus (PubMed files are ordered by
# PMID, so every 28th file spans 1970s-2023; StarCoder/OWM/FineWeb shards are pre-shuffled).

_STARCODER_REV = "9fc30b578cedaec69e47302df72cf00feed7c8c4"
_OWM_REV = "fde8ef8de2300f5e778f56261843dab89f230815"
_POL_REV = "2e96169e7e4b43f8ea36230515ebb44b27423b94"
_FINEWEB_REV = "9bb295ddab0e05d785b879661af7260fed5140fc"
_MEDRAG_REV = "33da3593d5756bc04c8909f170003c0b14197957"

REAL_SOURCES: dict[str, SourceSpec] = {
    "general": SourceSpec(
        name="fineweb-sample-10BT",
        kind="hf_file_parquet",
        repo_id="HuggingFaceFW/fineweb",
        data_files=["sample/10BT/000_00000.parquet"],
        text_field="text",
        revision=_FINEWEB_REV,
        max_chars=20000,
    ),
    "code": SourceSpec(
        name="starcoderdata-python",
        kind="hf_file_parquet",
        repo_id="bigcode/starcoderdata",
        data_files=["python/train-00000-of-00059.parquet", "python/train-00001-of-00059.parquet"],
        text_field="content",
        revision=_STARCODER_REV,
        max_chars=20000,
        strip_header_re=r"^(?:<(?:reponame|filename|gh_stars)>[^\n]*\n)+",  # StarCoder metadata header lines
    ),
    "biomedical": SourceSpec(
        name="medrag-pubmed-abstracts",
        kind="hf_file_jsonl",
        repo_id="MedRAG/pubmed",
        data_files=[f"chunk/pubmed23n{i:04d}.jsonl" for i in range(1, 1001, 20)],  # 50 files, ~1.5 GB
        text_field="contents",  # title + abstract
        revision=_MEDRAG_REV,
        min_chars=300,
    ),
    "legal": SourceSpec(
        name="pile-of-law-courtlistener-opinions",
        kind="hf_file_jsonl_xz",
        repo_id="pile-of-law/pile-of-law",
        data_files=["data/train.courtlisteneropinions.0.jsonl.xz"],
        text_field="text",
        revision=_POL_REV,
        min_chars=500,
        max_chars=20000,
    ),
    "math": SourceSpec(
        name="open-web-math",
        kind="hf_file_parquet",
        repo_id="open-web-math/open-web-math",
        data_files=[
            "data/train-00000-of-00114-5a023365406cb9c4.parquet",
            "data/train-00001-of-00114-e32fc2813a15f61c.parquet",
            "data/train-00002-of-00114-1429d96b99aec578.parquet",  # 2 shards gave ~163M tokens; 3 cover the 203M budget
        ],
        text_field="text",
        revision=_OWM_REV,
        max_chars=20000,
    ),
}
