# C5 — second task sequence (Tinker, exploratory)

generated 2026-11-15T08:54:23+00:00 · tasks: sst2 → mnli → ag_news · 1 smoke row(s) skipped

| arm | seeds | final avg acc | 95% CI | gap closed | 95% CI | forgetting | 95% CI | est $ |
|---|---|---|---|---|---|---|---|---|
| joint | 0,1,2 | 0.908 | [0.907, 0.909] | — | — | 0.000 | [0.000, 0.000] | 2.73 |
| naive | 0,1,2 | 0.896 | [0.894, 0.899] | — | — | 0.018 | [0.010, 0.029] | 2.95 |
| replay-200 | 0,1,2 | 0.882 | [0.864, 0.902] | -1.403 | [-5.477, 0.484] | 0.032 | [0.015, 0.054] | 2.89 |

**warning:** unstable denominator: joint - naive = 0.012 < 0.03; read gap closed with care (analysis/PREREG.md section 10)
