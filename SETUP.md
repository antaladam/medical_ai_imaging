# Setting up the project on a new computer

Commands are for Windows Git Bash, run from `semester_project/` unless stated. On macOS/Linux,
replace `.venv/Scripts/python` with `.venv/bin/python`.

## 1. Clone

```bash
git clone https://github.com/antaladam/msc_project.git
cd msc_project/semester_project
```

## 2. Python environment (Python 3.11)

```bash
python -m venv .venv
# PyTorch first, from the index that matches the machine:
.venv/Scripts/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU
# .venv/Scripts/python -m pip install torch==2.14.1 torchvision==0.29.1                                                   # CPU only
.venv/Scripts/python -m pip install -r requirements.txt
```

## 3. API key

Create `semester_project/.env` with one line (never commit it; `.gitignore` excludes it):

```
GOOGLE_API_KEY=your-key-here
```

Check it: `.venv/Scripts/python -I scripts/outdated/check_api_key.py`

## 4. Dataset (IU X-Ray, Open-i; CC BY-NC-ND 4.0, not in the repository)

**Source and licence.** Indiana University chest X-ray collection, distributed by the U.S.
National Library of Medicine through Open-i: Demner-Fushman D. et al., "Preparing a collection
of radiology examinations for distribution and retrieval", *JAMIA* 23(2), 2016. Licence:
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (attribution,
non-commercial, no derivatives).

**What is in this repository (decision 2026-10-18).** The raw images and reports, `data/processed/`
and all embedding files (`*.npy`, `*.npz`) are not committed. `semester_project/data/runs/` is
committed (model answers, evaluation outputs, API logs, week-7 review sheets and judgements),
so that every number in `results.md` can be traced. It quotes report text verbatim (neighbours'
reports inside cached prompts, indications, review sheets) and contains derived labels.
**This is allowed only because the repository is private.** Before the repository is ever made
public, remove `semester_project/data/runs/` from it and from its history; the licence's
no-derivatives condition forbids publishing adapted material.

```bash
mkdir -p data/raw/iu_xray && cd data/raw/iu_xray
curl -L -O https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz   # 1.1 MB
curl -L -O -C - --retry 5 https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz   # 1.36 GB, slow server
mkdir -p reports images
tar -xzf NLMCXR_reports.tgz -C reports
tar -xzf NLMCXR_png.tgz -C images
cd ../../..
```

## 5. Derived data (`data/processed/`)

**Option A (recommended): copy `semester_project/data/processed/` from the original machine**
(USB stick or private cloud; it is dataset-derived, so not on GitHub). This guarantees identical
frontal-image choices, labels and splits.

**Option B: regenerate**, in this order. Models (BiomedCLIP, CheXbert, bert-base-uncased) download
on first use; keep their cache inside the project with `export HF_HOME="$PWD/.cache/huggingface"`.

```bash
R=data/raw/iu_xray/reports/ecgen-radiology
I=data/raw/iu_xray/images
P=data/processed
mkdir -p $P
.venv/Scripts/python -I scripts/data_prep/classify_views.py   $R $I $P   # views.csv (+ check sheets), GPU recommended
.venv/Scripts/python -I scripts/data_prep/image_quality.py    $I $P      # image_quality.json, image_hf_ratio.json
.venv/Scripts/python -I scripts/data_prep/select_frontal.py   $P $I      # frontal.csv
.venv/Scripts/python -I scripts/data_prep/embed_and_dedup.py  $P $I      # BiomedCLIP embeddings (used later by R0 and G)
.venv/Scripts/python -I scripts/data_prep/pixel_duplicates.py $P $I      # duplicate check (prints only)
.venv/Scripts/python -I scripts/data_prep/chexbert_labels.py  $R $P/chexbert_labels.csv
.venv/Scripts/python -I scripts/data_prep/mesh_ground_truth.py $R $P     # ground_truth.csv
.venv/Scripts/python -I scripts/data_prep/make_splits.py      $P         # splits.csv
```

Optional checks (print only): `check_patient_ids.py $R`, `check_im_series.py $R`,
`check_image_mapping.py $R $I`, `label_stats.py $R $P`, `view_groups_sheet.py $P/views.csv $I $P`.

## 6. Verify the data is identical

```bash
sha256sum data/processed/frontal.csv data/processed/ground_truth.csv data/processed/splits.csv
```

Expected:

```
4d47269217f5b762976554ac290256d3406389fb785d01f3086c53793628c6a2  data/processed/frontal.csv
a962391f57e5a77e0cdeec3cb01a4c3e5b486174bd0be110fe82549a066940d2  data/processed/ground_truth.csv
df4afae451d3c79641463d96d151208c1d483699fa07aba1b8fbc202b0b99ef4  data/processed/splits.csv
```

If `frontal.csv` differs after regenerating, the view classifier (step 5, `classify_views.py`) gave
slightly different scores on this hardware. Copy `data/processed/` from the original machine
instead (option A), or the splits will not match the recorded results. Counts to compare against
`results.md`: 3,690 frontal studies; 3,604 ground-truth studies; splits 2,804 / 400 / 400.

## 6b. Anchor systems S0 and R0 (local, no API)

```bash
.venv/Scripts/python -I scripts/anchors/s0_score.py train,val   # S0 weights download to .cache/torchxrayvision/ on first use; ~3 min on GPU
.venv/Scripts/python -I scripts/anchors/s0_select.py            # selection + thresholds -> data/runs/s0/selection.json
.venv/Scripts/python -I scripts/anchors/r0_knn.py val           # leave-one-out k + thresholds on train, then val -> data/runs/r0/
```

The weights' SHA-256 values are in `semester_project/diary.md` (2026-10-17). They are loaded with
a restricted unpickler (`scripts/anchors/s0_weights.py`), not the library's unrestricted loader. Neither
script scores the test split without `--test` (plan: once, in week 6).

## 7. Run tests

```bash
.venv/Scripts/python -I scripts/tests/test_retry.py     # retry, metering, request and dollar limits (offline, fake API)
.venv/Scripts/python -I scripts/tests/test_metrics.py   # metrics vs. brute force
.venv/Scripts/python -I scripts/tests/test_parse_checklist.py   # checklist parser
```

## 8. Continue the work

Read `semester_project/diary.md`, the last "Next steps" entry first, and `results.md`. Those two
files hold the current state, every decision made so far, and the to-do list. The plan
(`semester_plan.md`) is frozen: record changes as deviations in `results.md` and the diary instead
of editing it.

API-run outputs (`data/runs/`) are not in the repository either; runners cache every response
there and skip studies already cached, so copy `data/runs/` too if you want to keep paid results.
