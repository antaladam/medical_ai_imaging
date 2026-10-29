# Memorisation check (W2) — test

Method: week7_design.md W2; `scripts/evaluation/memorisation.py`. Own = the study's own report (FINDINGS + IMPRESSION); pool = up to 20 other reports with the identical label vector, excluding every report shown to any system for that study.

Studies: 384 of 400 (left out: CXR152 (pool 4), CXR650 (pool 0), CXR1348 (pool 1), CXR1564 (pool 0), CXR1845 (pool 1), CXR1929 (pool 4), CXR2043 (pool 1), CXR2046 (pool 0), CXR2114 (pool 1), CXR2456 (pool 4), CXR2615 (pool 0), CXR2642 (pool 0), CXR2887 (pool 4), CXR3096 (pool 1), CXR3272 (pool 4), CXR3614 (pool 1)).

| System | 3-gram containment own / pool mean | Own − pool [95% CI] | Own above the pool's maximum (chance) | TF-IDF cosine own / pool mean | Own − pool [95% CI] |
|---|---|---|---|---|---|
| P1-adapted-ns | 0.053 / 0.048 | +0.004 [-0.001, +0.010] | 6.8% (5.0%) | 0.085 / 0.074 | +0.010 [+0.006, +0.015] |
| P2-adapted | 0.035 / 0.034 | +0.001 [-0.003, +0.005] | 4.9% (5.0%) | 0.084 / 0.075 | +0.009 [+0.005, +0.013] |
| P3-adapted | 0.033 / 0.031 | +0.002 [-0.002, +0.007] | 4.9% (5.0%) | 0.075 / 0.069 | +0.006 [+0.002, +0.010] |
| G | 0.059 / 0.053 | +0.006 [-0.000, +0.012] | 6.5% (5.0%) | 0.104 / 0.085 | +0.019 [+0.013, +0.025] |
| G−ind | 0.054 / 0.051 | +0.003 [-0.003, +0.009] | 5.7% (5.0%) | 0.096 / 0.085 | +0.011 [+0.006, +0.017] |
| G-img | 0.041 / 0.038 | +0.003 [-0.002, +0.010] | 6.8% (5.0%) | 0.094 / 0.079 | +0.015 [+0.010, +0.021] |
