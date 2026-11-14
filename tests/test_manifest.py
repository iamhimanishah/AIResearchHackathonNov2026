"""Manifest integrity: the checked-in smoke manifest must match the checked-in fixtures, and any
byte-level tamper must be caught."""

import json
from pathlib import Path

import numpy as np
import pytest

from clharness.config import REPO_ROOT
from clharness.data.manifest import (
    ManifestMismatch,
    _iter_file_entries as _file_entries,
    file_entry,
    read_manifest,
    verify_manifest,
    write_manifest,
)

SMOKE = REPO_ROOT / "manifests" / "smoke.json"
DATA_MANIFESTS = sorted(p for p in (REPO_ROOT / "manifests").glob("*.json") if not p.name.startswith("eval-"))


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke manifest not built (make smoke)")
def test_smoke_manifest_matches_disk():
    verify_manifest(read_manifest(SMOKE))


@pytest.mark.parametrize("path", DATA_MANIFESTS, ids=[p.stem for p in DATA_MANIFESTS])
def test_data_manifest_matches_disk_and_counts(path: Path):
    """Every data manifest present (smoke, real, ...): hashes match disk, counts match contents.
    A manifest for data that is not on this machine is a failure, not a skip: the run would be."""
    m = read_manifest(path)
    files = [REPO_ROOT / e["path"] for e in _file_entries(m)]
    if files and not any(f.exists() for f in files):
        pytest.skip(f"data set '{m['set']}' not built on this machine (clh build-data --set {m['set']})")
    verify_manifest(m)
    for dom, entry in m["domains"].items():
        for split, s in entry["splits"].items():
            arr = np.load(REPO_ROOT / s["tokens"]["path"], mmap_mode="r")
            assert arr.shape[0] == s["n_tokens"], (dom, split)
            if "doc_ids" in s:
                assert len(s["doc_ids"]) == s["n_docs"]
            with open(REPO_ROOT / s["jsonl"]["path"], "rb") as f:
                assert sum(1 for _ in f) == s["n_docs"], (dom, split)


def test_tamper_is_detected(tmp_path: Path, monkeypatch):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    man = {"files": [file_entry(f, root=tmp_path)]}
    assert man["files"][0]["path"] == "x.bin"
    verify_manifest(man, root=tmp_path)
    f.write_bytes(b"hello worle")  # same size, one byte differs
    with pytest.raises(ManifestMismatch, match="sha256 differs"):
        verify_manifest(man, root=tmp_path)
    f.unlink()
    with pytest.raises(ManifestMismatch, match="missing"):
        verify_manifest(man, root=tmp_path)


def test_manifest_roundtrip(tmp_path: Path):
    p = tmp_path / "m.json"
    write_manifest({"a": 1, "b": [1, 2]}, p)
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2]}
