"""C5: the second task sequence on Tinker (analysis/PREREG.md section 10, exploratory).

LoRA on Qwen/Qwen3-8B, SST-2 -> MNLI -> AG News, 5k train / 1k eval per task, scored by
verbalizer log-likelihood. This package is self-contained: it emits its own `TinkerRow`
(the C5 tasks are not the protected evals, so the shared `ResultRow` does not apply), and
it never touches the frozen manifests or the bench loop.

The `tinker` SDK is an optional extra (`pip install clharness[tinker]`) and is imported
lazily inside runner.py; everything else here works without it and without network.
"""
