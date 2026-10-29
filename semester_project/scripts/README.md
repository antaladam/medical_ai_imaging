# Scripts

Run every script from `semester_project/` with the project environment, e.g.
`.venv/Scripts/python -I scripts/g/g_system.py g val --pilot`. Each script's docstring gives its
usage. The project folder is computed only in `common.py`; every other script imports `PROJECT`
from it, so scripts can live in subfolders.

| Folder | What it holds |
|---|---|
| `common.py`, `metrics.py` | Shared libraries: API key, findings, questions, metered client (request and USD limits, call log), prices, cache; evaluation metrics (AUROC, threshold metrics, Youden, bootstrap) |
| `g/` | **G, the built system**: `g_system.py` (G, G−ind, G-img), `g_export.py` (answers → per-study table), `g_select.py` (pooled validation threshold + validation report) |
| `starters/` | The three starter projects as batch runners: `p1_adapted.py` (P1-adapted, P1-adapted-ns), `p2_adapted.py`, `p3_adapted.py` (P3-adapted, P3-orig with `--shipped-cap`) |
| `anchors/` | The non-LLM reference systems: S0 (`s0_weights.py`, `s0_score.py`, `s0_select.py`) and R0 (`r0_knn.py`) |
| `data_prep/` | The data pipeline (views, image quality, frontal selection, BiomedCLIP embeddings, CheXbert labels, ground truth, splits) and its week-1 checks |
| `evaluation/` | Scoring and analysis: `evaluate.py` (the Q3 evaluation, results §5.4), `memorisation.py` and `week7_samples.py` (week 7, results §5.6), `without_findings.py` (sensitivity check, results §5.5), `parse_checklist.py`, `pilot_summary.py`, `checklist_review.py`, `q2_retrieval_compare.py` |
| `tools/` | `spend.py` (API spend ledger) |
| `outdated/` | Kept, no longer used: `label_outputs.py` (CheXbert labelling of answers, replaced by the checklist question), `check_api_key.py` (week-1 model check; calls retired models and bypasses the metered client) |
| `tests/` | Offline tests (no API calls) |
