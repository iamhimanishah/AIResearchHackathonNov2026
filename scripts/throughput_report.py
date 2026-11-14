"""Print training tokens/s for a finished run: (gradient tokens) / (wall between first and last
checkpoint minus hook time)."""

import glob
import json
import sys

name, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(f"results/runs/{name}/metrics.jsonl")]
ck = [r for r in rows if "ckpt/index" in r]
metas = [json.load(open(p)) for p in sorted(glob.glob(f"results/runs/{name}/checkpoints/*/meta.json"))]
toks = metas[-1]["gradient_tokens"] - metas[0]["gradient_tokens"]
train_s = (metas[-1]["time"] - metas[0]["time"]) - sum(r["ckpt/hooks_wall_s"] for r in ck[:-1])
print(f"{name}: {toks / train_s / 1e3:.1f}k tok/s  ({toks / 1e6:.1f}M tokens in {train_s:.0f}s train; wall {t1 - t0}s)")
