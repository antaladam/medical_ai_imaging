# Project Diary

A running log of every change to this semester project: what was done, why, and what was found.
Newest entries at the bottom. Plan: `semester_plan.md`.

---

## 2026-10-16 — Semester plan written and reviewed

- Created `semester_plan.md`: three questions (Q1 starter fitness, Q2 reusable strengths,
  Q3 built system G vs. starters, S0 and R0), 8-week timeline. The full thesis plan stays in
  `../thesis_plan.md` as the long-term goal.
- Review rounds changed the plan:
  - Added R0 (nearest-neighbour label vote, no LLM) to test whether Gemini adds anything beyond
    copying retrieved cases.
  - Added a train/validation/test split, a positive-enriched test set, frontal-image-only input,
    and study as the unit of analysis.
  - Contamination moved to a closing "Threats to validity" section (§8). S0 weights check and
    leakage guards moved into Systems and Data.
  - Week 1 gained a patient-ID check: the public IU X-Ray release may have no patient ID, in which
    case the split is by study and a repeat-patient screen + robustness check apply.
  - Removed McNemar tests (bootstrap CIs only); labeller check made per-system; "uncertain"
    CheXbert labels fixed as negative in advance; §8 bias wording corrected (memorisation also
    favours Gemini systems over S0).

## 2026-10-16 — Week 1 started: patient-ID check

- Downloaded the IU X-Ray reports archive from Open-i
  (`https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz`, 1.1 MB, archive dated
  2017-06-06) to `data/raw/iu_xray/`, extracted to `data/raw/iu_xray/reports/ecgen-radiology/`.
  Images (`NLMCXR_png.tgz`) not downloaded yet.
- Added `scripts/check_patient_ids.py` and `scripts/check_im_series.py`.
- **Result: the release has no patient identifier.** Across all 3,955 reports the only ID fields
  are study/report IDs (`uId` = CXR####, `pmcId`, `IUXRId`) and image IDs (`parentImage@id`).
- The `IM-####` part of image names looked like a possible link (813 values shared by >1 report),
  but it is a **sequential counter assigned over string-sorted report IDs** (CXR1, CXR10, CXR100,
  …): the number never decreases along that order except once (restart at CXR1015), shared values
  occur only between neighbours (2 exceptions), and a shared group mixes clearly different
  patients (e.g. IM-0013: a female with chest pain, a bone-marrow-transplant patient, a Burmese
  male after TB treatment). Not usable as a patient ID.
- **Consequence (plan §6, week-1 "if not" branch): the split is by study.**
- Evidence that repeat patients exist in general: COMPARISON field references a prior exam in
  1,909 / 3,955 reports; findings/impression wording suggests a prior in 1,105. This shows priors
  existed, not that they are in the collection. Still open: the dataset paper's exam-selection
  description, and the BiomedCLIP repeat-patient screen (needs the images).
- Other facts: 3,851 reports have images (104 have none); images per report: 1 → 446, 2 → 3,208,
  3 → 181, 4 → 15, 5 → 1; 7,470 images total.
- Side observation for G−ind: some indications state the finding outright
  (CXR1021: indication "Right pneumothorax.").

## 2026-10-16 — Plan frozen; results file created

- From now on `semester_plan.md` is **not edited**. Differences between plan and practice are
  recorded in `results.md` ("Deviations from the plan") and in this diary.
- Created `results.md`: the report to be handed in, structured by the plan's questions. Filled in
  so far: data source and counts (§2.1), the patient-ID finding and study-level split (§2.2),
  first threat to validity, two deviations, one observation.
- Deviations recorded:
  1. No patient IDs → split by study (the plan's anticipated "if not" branch).
  2. The plan asked for its own §2/§8 to be updated with this; instead it is recorded in
     `results.md`, because the plan is frozen.

## 2026-10-16 — Dataset paper read; image download started

- Started downloading the image archive
  (`https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz`) to `data/raw/iu_xray/`
  (resumable curl with retries; the first header request was reset by the server).
- Looked up the dataset paper: Demner-Fushman et al., JAMIA 2016;23(2):304–310,
  doi:10.1093/jamia/ocv080, full text PMC5009925 (the IU ScholarWorks copy was bot-blocked).
- **Key finding:** "In no case did we take more than one study per patient." Each of the
  original 4,000 reports came from a different patient (2,000 per institution). So the
  study-level split is effectively a patient-level split; repeat patients cannot cross splits.
- Also from the paper: outpatient studies were targeted; inclusion needed at least PA + lateral
  views; single-image public studies come from removing images showing teeth/jaw/jewellery/skull;
  public subset = 3,955 reports / 7,470 images, matching the downloaded counts exactly.
- **Deviation recorded in `results.md`:** the repeat-patient screen is reduced to a near-duplicate
  image check, and the week-7 repeat-patient robustness check is dropped (no longer needed).
- `results.md` §2.2, §2.3 and §6 updated accordingly; the earlier "study split is a threat" entry
  removed from §6.

## 2026-10-16 — Images downloaded and checked

- Image archive finished: 1,360,814,128 bytes, gzip integrity OK, archive dated 2016-09-06.
  Checked the archive's paths (no absolute or `..` paths), then extracted it to
  `data/raw/iu_xray/images/`: 7,470 PNGs + a stray `Thumbs.db`.
- Created a project virtual environment `.venv/` (inside `semester_project/`) and installed Pillow
  12.3.0. All project Python runs from here on use `.venv`.
- Added `scripts/check_image_mapping.py`. Result: all 7,470 PNGs map one-to-one to report image
  IDs (PNG names drop the middle `_<n>_`; CXR1 is the only exception, keeping it). All readable,
  all RGB; 3,332 portrait / 2,610 landscape / 1,528 square.
- **Finding: no frontal/lateral view label in the release.** Plan §2 assumes the frontal image
  can be picked, so a view classifier is needed. Recorded as a deviation in `results.md`.
- `results.md`: §2.1 image facts added; new §2.4 (frontal-image selection, pending); labels
  section renumbered to §2.4b; deviations table +1 row.

## 2026-10-16 — Libraries installed; S0 weights checked; frontal images selected

- Installed into `.venv`: torch 2.14.1+cu126, torchvision 0.29.1 (CUDA works on the RTX 4060,
  driver 610.88), torchxrayvision 1.5.5, open_clip_torch 3.3.0, transformers 5.20.0.
  Hugging Face cache kept inside the project (`HF_HOME=.cache/huggingface`).
- **S0 weights:** read `torchxrayvision/models.py`. `densenet121-res224-all` includes Open-i
  (IU X-Ray) → ruled out. Alternatives without Open-i listed in `results.md` §2.3b; choice
  deferred until findings are selected.
- **View classification:** `scripts/classify_views.py` (BiomedCLIP zero-shot, 3 prompts per
  view) → `data/processed/views.csv`. Checked by eye with contact sheets
  (`scripts/view_groups_sheet.py`): random 60/60 correct; the 0.5 threshold misses degraded
  frontal views.
- Found 4 unusable images (3 blank, 1 noise). A first noise metric (neighbour differences at
  128 px) failed: it flagged sharp normal images, so it was replaced with a full-resolution
  high-frequency/contrast ratio, where the noise image is a lone outlier.
- **Selection rule** (`scripts/select_frontal.py`): drop unusable, take the highest `p_frontal`
  per study, accept if ≥ 0.05. Checks: all 11 low-score selections frontal; 40/40 sampled
  exclusions lateral. Result: **3,690 studies** with one frontal image
  (`data/processed/frontal.csv`); 161 lateral-only studies excluded.
- `results.md`: §2.3b (S0 weights) and §2.4 (frontal selection) written; deviations table +2 rows.

## 2026-10-16 — Near-duplicate check

- `scripts/embed_and_dedup.py`: BiomedCLIP embeddings of the 3,690 selected frontal images,
  saved for later reuse by R0/G (`data/processed/biomedclip_frontal.npy` + `_ids.csv`).
  Nearest-neighbour similarity max 0.988, no outlier; top-30 pairs checked by eye, all different
  patients.
- `scripts/pixel_duplicates.py`: no identical files, no identical quantised thumbnails.
- Result: no duplicates. Noted in `results.md` §2.3 that BiomedCLIP similarities sit in a narrow
  band (median 0.970), which matters for retrieval later.

## 2026-10-16 — CheXbert labels; label quality check

- Read the `f1chexbert` 0.0.2 source (downloaded, not installed, to `.cache/pkg_inspect/`):
  it fetches `chexbert.pth` from HF `StanfordAIMI/RRG_scorers`; its default "rrg" mode counts
  uncertain as positive and it loads weights non-strictly. So a separate labeller was written
  (`scripts/chexbert_labels.py`, same model definition, credited) with uncertain → negative
  (plan §5), `weights_only=True` and a strict key check.
- Labelled all 3,955 reports (FINDINGS + IMPRESSION) → `data/processed/chexbert_labels.csv`.
- `scripts/label_stats.py`: positive counts per observation; comparison with the 161 excluded
  lateral-only studies (sicker); agreement with IU's manual MeSH codes (κ 0.76–0.90, Pneumonia 0.61).
- Error analysis: 55 of 62 reports with the template "The heart and lungs have XXXX XXXX in the
  interval… Heart and mediastinum normal" are labelled Cardiomegaly (13% of its positives).
- Tested removing the "XXXX" masks (`--drop-masks` option added) →
  `data/processed/chexbert_labels_dropmasks.csv`: no improvement. Kept as a recorded negative
  result; the main labels file is unchanged.
- **Stopped here for a decision:** which reference standard (CheXbert as planned, the manual
  MeSH codes, or a combination) and which findings. Recorded under "Open decisions" in
  `results.md`.

## 2026-10-16 — Decisions: MeSH ground truth, six findings

- Decisions: (1) ground truth from the manual MeSH codes, CheXbert kept for labelling system
  outputs; (2) six findings S0 covers: Lung Opacity, Cardiomegaly, Atelectasis, Lung Lesion,
  Pleural Effusion, Fracture.
- Inspected the real MeSH code variants and fixed a mapping (in `scripts/mesh_ground_truth.py`
  and `results.md` §2.4b): "borderline" cardiomegaly → negative; lung-located nodule/mass/
  cavitation → Lung Lesion (calcified granulomas excluded); opacity/density at lung,
  airspace disease, infiltrate, consolidation → Lung Opacity.
- Excluded 86 "No Indexing" studies → **3,604 ground-truth studies**
  (`data/processed/ground_truth.csv`). Positives: Lung Opacity 519, Atelectasis 296,
  Cardiomegaly 295, Pleural Effusion 138, Lung Lesion 104, Fracture 83.
- CheXbert vs ground truth κ 0.66–0.90. The Cardiomegaly template error falls mostly in the
  excluded uncoded studies (κ 0.79 → 0.87).
- `results.md`: decisions, mapping, counts; open-decisions section removed; deviations +2 rows.

## 2026-10-16 — Splits created; threshold-tuning issue found

- `scripts/make_splits.py` (seed 2026): test 400 (positive-enriched, ≥35 per finding),
  validation 400 (natural prevalence, pilot 50 inside it), train 2,804. Assertions pass
  (disjoint, complete). → `data/processed/splits.csv`.
- Issue: validation has Fracture 2, Lung Lesion 11, Pleural Effusion 12 positives, too few to
  tune per-finding thresholds as the plan intends. Recorded in
  `results.md` §2.5.
- Decision: keep the splits. S0 thresholds tuned on train (never saw IU); R0 thresholds
  and k by leave-one-out on train; G/G−ind one pooled threshold on validation; starters
  untuned. Recorded in `results.md` §2.5, deviations, and threats (thin index for Lung Lesion
  and Fracture).

## 2026-10-16 — API key check; starter models retired

- Created `.env` with the Gemini API key (the key itself was never printed or opened). Reminder: if the
  folder is zipped for hand-in, exclude `.env`.
- Decisions: keep all three starters; P2 becomes **P2-adapted** (Gemini Embedding 2
  instead of Cohere), with **its own embeddings**, not shared with P3, at Gemini Embedding 2's
  default settings plus P2's own image-shrinking step.
- Installed `google-genai` 2.31.0. Added `scripts/common.py` (key loading from `.env`, fixed
  findings question, response cache in `data/runs/<arm>/<split>/<study>.json`) and
  `scripts/check_api_key.py`.
- **Key check:** `gemini-2.5-pro` (P1) and `gemini-2.5-flash` (P2) return "no longer available
  to new users". `gemini-3-flash-preview` (P3) and `gemini-embedding-2` work. The key is on the
  free tier (Pro-preview quota error; no free tier for it per pricing page).
  `gemini-3.5-flash` requests are served by `gemini-3.6-flash` (alias), so `model_version` is
  recorded per call.
- Stopped to choose successor models for P1 and P2. Recorded in `results.md` §3.1.
- Decision: P1 and P2 both use `gemini-3-flash-preview`. P1 → P1-adapted (Pro → Flash).
  Recorded in `results.md` §3.1 and deviations (+2 rows: P2 embedder swap, model successors).

## 2026-10-16 — P2-adapted built and smoke-tested

- Checked IU image sizes: all frontal images 512 px wide, ≤ 0.34 MP, so P2's resize
  (2.46 MP limit) never triggers.
- Wrote `scripts/starters/p2_adapted.py` (index / run). Details and adaptations in
  `results.md` §3.2. Safeguards: leakage assertion; refuses a partial index unless `--smoke`,
  which writes to `data/runs/p2_adapted_smoke/` instead of the real cache.
- Smoke test: embedded 20 train images (`data/runs/p2_adapted/embeddings/`; these are real index
  vectors and will be reused by the full index), ran 2 pilot studies with `--smoke`. Vectors 3,072-d,
  norm 1.0. Tokens ~2,270 in / ~550 out / 820–2,180 thinking per study (≈ $0.007 paid). CXR8
  answered correctly (no findings).
- Next: full train index (2,804 embedding requests) and the 50-study pilot. Free-tier embedding
  quota unknown; runner retries with backoff on 429/503.

## 2026-10-16 — P1-adapted built

- Installed `agno` 3.1.4 (the starter requires agno>=2.2.10; major-version jump noted).
  agno with `markdown=True` adds "Use markdown to format your answers." to the system message,
  as in the shipped app.
- Wrote `scripts/starters/p1_adapted.py`: agno Agent + Gemini, verbatim 5-section prompt,
  width-500 resize as shipped, per-study temp file, DuckDuckGo off (`--web-search` for the
  sensitivity arm `p1_adapted_web`), findings question appended, `gemini-3-flash-preview`.
  Records the exact messages agno sent.
- **Bug found and fixed:** agno does not raise on API errors. On a 503 it returned a run whose
  *content* was the error JSON, with 0 tokens, and the runner cached it as an answer (CXR8,
  CXR86). Added `run_checked`: a run that is not COMPLETED or has no output tokens now raises,
  so it is retried and never cached. **Deleted the 2 invalid error records**
  (`data/runs/p1_adapted/val/CXR8.json`, `CXR86.json`).
- Rerunning the 2-study smoke test: the model is returning 503 "high demand"; the runner is
  retrying with backoff (in background).
- **Second bug found and fixed:** the rerun hung for 37 minutes with no output and ~0 CPU.
  Cause: agno's Gemini wrapper defaults to `timeout: None` and `retries: 0`, so a dropped
  connection during Google's overload waited forever. Stopped the hung process. Added a
  180 s per-request timeout to both runners (agno `Gemini(timeout=...)` for P1;
  `HttpOptions(timeout=...)` for P2), and moved the retry logic into `scripts/common.py`
  (`with_retry`), which now also retries timeouts. The 503s
  are Google-side overload ("high demand"), not the free-tier quota (that would be 429).
- Rerun with unbuffered output to `data/runs/logs/p1_smoke.log` (the earlier `| grep` pipe
  had buffered all output, hiding the retries). Retries now visible: agno requests kept getting
  503, while direct google-genai calls (text, then one image) succeeded in 1–2 s. Unresolved
  whether that was agno-specific or timing.
- Then a direct call returned **429: free-tier quota `GenerateRequestsPerDayPerProjectPerModel-
  FreeTier`, limit 20/day for gemini-3-flash**, retry after ~5.4 h. Failed 503 attempts seem to
  count. Recorded in `results.md` §3.1: the free tier cannot carry this study.
- No P1 results cached yet (`data/runs/p1_adapted/val/` empty).

## 2026-10-16 — Runners: quota-aware stopping and request budget

- `scripts/common.py` `with_retry` rewritten:
  - **daily quota** (429 whose details mention "PerDay") → stop the run immediately
    (`QuotaExhausted`); cached results are kept, and rerunning the same command continues;
  - other 429s (per-minute limits) → at most 2 retries, 65 s apart, then stop;
  - overload/timeouts (500/503/504, timeouts) → at most 5 retries (60, 120, 240, 300, 300 s);
  - any other error → raised immediately;
  - **request budget**: `--max-requests N` on both runners stops the run after N API attempts.
    Failed attempts count too, since they can count toward the provider's quota. Each run
    prints its total attempts at the end.
  - Fixed during writing: "FreeTier" alone is not a daily-quota marker (per-minute free-tier
    quota IDs contain it too), so only "PerDay" triggers the immediate stop.
- Known limitation: through agno (P1), a 429's error text may not include the quota ID, so a
  daily quota there costs 3 attempts (2 retries) before stopping instead of 1.
- Added `scripts/tests/test_retry.py` (offline, fake errors, no API calls): 9/9 pass.
- API quota (billing) is being arranged. No API calls made for this change.

## 2026-10-16 — Next steps (end of day)

Where things stand: week 1 of the plan is done (data, labels, splits, S0 weight check). P2-adapted
and P1-adapted runners are built; no real results yet. Free-tier quota (20 requests/day for
gemini-3-flash) is used up for today.

**Blocked**
1. Arrange API quota: enable billing (paid Tier 1) on the Gemini key, or obtain credits via the
   supervisor (Gemini for Research / education credits). Estimated total ≈ $30–35 standard,
   ≈ $15–18 with the Batch API.
2. Open decision: add the **G-img** arm (G's model + prompt, image only), proposed 2026-10-16 as
   a clean image-only Gemini reference (≈ $5). Not yet decided.
3. Supervisor questions still open (plan §9: scope, hosted API / budget, modality), now also:
   API credits, and acceptance of the recorded deviations (MeSH ground truth, P1/P2-adapted).

**Can be done now, no API needed**
4. **S0**: run the TorchXRayVision weights that exclude Open-i (`resnet50-res512-all`,
   `densenet121-res224-chex`, `-mimic_nb`, `-mimic_ch`) on the train split; pick by macro AUROC
   on train (S0 never saw IU); tune per-finding thresholds on train.
5. **R0**: kNN label vote on the existing BiomedCLIP embeddings
   (`data/processed/biomedclip_frontal.npy`); choose k and thresholds by leave-one-out on train.
6. **Output labelling pipeline**: run CheXbert (`scripts/chexbert_labels.py` model) on cached
   free-text answers → six binary findings, uncertain → negative; compare with ground truth.
7. **P3 runner** (code only): wrap `multimodal_agentic_rag`'s `rag_store` + ADK agent; index =
   train frontal images with their reports as annotation (blend weight 0.32 as shipped), own
   Gemini Embedding 2 vectors (not shared with P2); query by image embedding; top_k 6;
   `gemini-3-flash-preview`; same error/timeout/budget handling as P1/P2.
8. Install the DuckDuckGo search package for P1's `--web-search` sensitivity arm (check what
   agno 3.x's DuckDuckGoTools requires).

**Once quota is available (in this order, each with `--max-requests`)**
9. P1-adapted 2-study test (`p1_adapted.py val --pilot --limit 2 --max-requests 10`). Check the
   output and the messages agno sent. Watch whether agno requests still get 503 while direct
   calls succeed (unexplained on 2026-10-16).
10. P2-adapted full train index (`p2_adapted.py index --max-requests 3000`; 2,804 embeddings,
    ≈ $0.34; 20 already done and reused).
11. 50-study pilot for P1-adapted and P2-adapted (later P3), `--max-requests 70` each.
12. From the pilot: real token usage → projected cost of the full runs (report before
    continuing); Q1 qualitative notes (output format, refusals, hedging, invented content; how
    P1 handles the section-5 web-search instruction with no tool).
13. Then week 3 of the plan: Q1 write-up, Q2 synthesis → design of G.

**Housekeeping**
- When handing in or zipping `semester_project/`, **exclude `.env`** (API key) and `.venv/`,
  `.cache/`, `data/raw/` (large, re-downloadable).
- `data/runs/p2_adapted_smoke/` holds partial-index test output; it is not results.

## 2026-10-16 — Progress presentation

- Created `presentation/progress_review.html`: a self-contained 18-slide HTML presentation
  (brief → thesis plan → semester project → work done → deviations → next steps → supervisor
  questions). Content taken from the plans, `results.md` and this diary; no new numbers.
  Name and supervisor are placeholders (`[Name]`, `[Supervisor]`).
- Controls: arrow keys / click to move, N speaker notes, F fullscreen, Ctrl+P saves a PDF
  (one slide per page). Works offline (system fonts only).
- Checked every slide by rendering with headless Edge; fixed one CSS bug (slide 9's big
  numbers rendered at body size).
- Created the Hungarian version, `presentation/progress_review_hu.html`: same 18 slides, speaker
  notes and controls. Hungarian number format (3 955; 0,73), „…” quotes; the official English
  thesis title and quoted English source sentences kept in English; finding names in Hungarian
  (English equivalents in slide 11's notes). Rendered all slides with headless Edge; fixed three
  line-break issues (dollar signs and a quote split from their numbers/text).
- Rewrote the Hungarian text ("too clunky"): less literal phrasing
  throughout. Terminology fixes: *elváltozás* for the finding categories (*lelet* now only means
  the radiology report, as in Hungarian radiology usage); *referenciacímkék* instead of the
  literal *alapigazság* for ground truth; *oldalirányú* (lateral) view; *Csonttörés*;
  *előteszt* for the pilot; *rögzített terv* for the frozen plan. Re-rendered all 18 slides
  (isolated Edge profile; stopped one leftover headless Edge process from a stalled render).

## 2026-10-16 — Pushed to GitHub

- Initialised a git repository at `Project 4/` and pushed to
  https://github.com/antaladam/msc_project (branch `main`, commit 7d13d43): 51 files, ~1.2 MB —
  the brief, both thesis plans, the starter analysis and starter projects, and the semester
  project (plan, results, diary, scripts, presentations).
- Added `.gitignore` (excludes `.env`, `.venv/`, `.cache/`, and all of `semester_project/data/`,
  i.e. the CC BY-NC-ND dataset and everything derived from it) and a top-level `README.md`
  (contents, what is not in the repo, third-party credits). Checked before committing: no
  `.env`, environment or data files staged; no API-key-like strings in any committed file.
- This diary entry was pushed as a follow-up commit.

## 2026-10-16 — Setup for a second computer

- Added `semester_project/requirements.txt` (pinned versions from this machine; PyTorch installed
  first from the CUDA or CPU index).
- **Closed a reproducibility gap:** `image_quality.json` and `image_hf_ratio.json` (used by
  `select_frontal.py`) had been produced by one-off inline code. Added
  `scripts/image_quality.py`. Verified in a scratch folder: it reproduces both files exactly
  (max difference 0.0), and re-running `select_frontal.py`, `mesh_ground_truth.py` and
  `make_splits.py` from it gives byte-identical `frontal.csv`, `ground_truth.csv` and
  `splits.csv` (SHA-256 recorded in `SETUP.md`). The real files were not overwritten.
- Added `SETUP.md` (clone, environment, `.env`, dataset download, pipeline order, checksums,
  where to continue) and linked it from `README.md`.

## 2026-10-17 — Semester-only presentation; wording and README changes

Made by hand (commits 1440fa9 to d4e4e98).
- Both presentations: the name and supervisor placeholders filled in (Antal Ádám; supervisor
  Dr. Alwahab Dhulfiqar Zoltán). "Owner decision" removed from a speaker note and a table cell
  in the English version.
- "Owner" wording removed from this diary (entry titles and bullets now say "Decision") and from
  the deviations table in `results.md`.
- `.gitignore`: `.DS_Store` added.
- New `presentation/progress_review.html`: a 13-slide version about the semester project only.
  Compared with the 18-slide version it drops the two thesis-plan slides (the comparative study
  and the ladder of systems), "Waiting on decisions", "Roadmap" and the supervisor questions.
  Section labels shortened (e.g. "Done so far · week 1" → "Done so far"). Slide-number comments
  in the source fixed after the removals.
- The 18-slide English version and the Hungarian version moved to `presentation/old/`.
- Top-level `README.md` replaced with the project brief, headed with the author's name (same
  text as `main.txt`). The previous README content became `CONTENTS.md`.

## 2026-10-17 — Progress review rebuilt in the faculty template

- Created `presentation/progress_review_using_template.html`: the 13-slide
  `presentation/progress_review.html` rebuilt in the Óbuda University NIK presentation template
  (`NIK_prezentacio_sablon_2025`, downloaded as `…pptx.ppt`; the file is actually a `.pptx`).
  The original is unchanged.
- The template: 3 sample slides (empty title-slide variants) and 112 slide layouts in three
  families: white with navy header and footer bands and the white logo; full navy background;
  white with the colour ÓE/NIK logo. 4:3 (10 × 7.5 in), Open Sans, navy `#1D294D`, with yellow
  `#FCAF17` and light blue `#9AD1F0` as panel fills. Layout types: title slide, content + text
  (*tartalomrész*), two-column comparison (*összehasonlítás*, right panel plain, navy, yellow or
  light blue, optionally with column headers), picture with caption, references, closing slide.
- Mapping (slide → template layout):

  | Slides | Family | Layouts |
  |---|---|---|
  | 1, 5, 11 | Navy | title slide; content above text; comparison (extended to three panels) |
  | 3, 7, 9, 13 | White, navy bands | content above text; text left / content right; single content area |
  | 2, 4, 6, 8, 10, 12 | White, colour logo | comparison with light-blue or navy right panel; content above text; single content area |

  Cards use the template's navy, yellow and light-blue panel fills; tables have a navy header
  row. The section labels ("Done so far" etc.) sit top right, where the template puts its
  header text. Template positions are kept for the title (0.92 in from the top), the content
  areas and the footer (text bottom left, slide number bottom right).
- Kept from the original: all slide text, speaker notes, footer text and the controls (arrows /
  click, N notes, F fullscreen, Ctrl+P PDF, now 4:3 pages). The logos are embedded in the file;
  only Open Sans loads from Google Fonts (offline it falls back to Calibri / Arial).
- Not added: the template's closing slide ("KÖSZÖNÖM A FIGYELMET!"), since it would add text.
- Checks: a script compared the visible text and speaker notes of every slide with the
  original: identical on all 13. All slides were rendered with WebKit and checked by eye.
  Fixed: cards stacked vertically instead of side by side on slides 3, 10 and 11 (CSS bug);
  large empty gaps on slides 4, 7 and 9; text too small for the space on slides 8, 9 and 13; the
  navigation hint covering the slide number (it now fades after 4 s).
- Known: on lighter slides (3, 7, 10, 11, 13) the content sits in the upper part with empty
  space below, because the template's layouts anchor content at the top.

## 2026-10-17 — Remaining "owner" wording removed

- Completes the wording change from the morning (see above): six places still said "owner".
  - `results.md` §2.4b, §2.5 and §3.1: "(project owner, 2026-10-16)" → "(2026-10-16)" in the
    decision labels.
  - This diary, 2026-10-16 entries: "Owner is arranging API quota (billing)" → "API quota
    (billing) is being arranged"; the "Blocked on the owner" heading in "Next steps" →
    "Blocked".
  - `scripts/starters/p2_adapted.py`, top comment: "(owner decision, no Cohere account)" →
    "(decision: no Cohere account)". Comment only; the code is unchanged.
- No "owner" wording is left in the repository except in the entries above that describe
  the change.

## 2026-10-17 — S0 and R0 started: design decisions before any code

Started from the end-of-day "Next steps" items 4 (S0) and 5 (R0). Local copy fast-forwarded to
`origin/main` (74f1045) first; `frontal.csv`, `ground_truth.csv` and `splits.csv` re-checked
against the SHA-256 values in `SETUP.md` (all match). Decisions, each with its reason:

**S0 — which weights are candidates.** Read `torchxrayvision/datasets.py` to see which training
datasets define each of the six labels:
- `densenet121-res224-chex`, `-mimic_nb`, `-mimic_ch`: all six findings are trained outputs.
- `resnet50-res512-all` (pc-nih-rsna-siim-vin): **its "Lung Lesion" output is untrained.** None
  of its five datasets defines Lung Lesion (NIH, PadChest and VinDr have Nodule / Mass instead),
  which is why the library gives it a `nan` operating threshold. It could only score Lung Lesion
  through a proxy such as max(Nodule, Mass).
- `-nih`, `-pc`, `-rsna`: each lacks at least two of the six → not candidates.
- Plan §3 names S0 as "TorchXRayVision **DenseNet**". **Decision: the selection set is the three
  DenseNets** (plan-conformant, all six findings native, no proxy). The ResNet-50 is scored too,
  as a reported-only reference with Lung Lesion = max(Nodule, Mass); if it were clearly better,
  that would be raised as an open question, not switched to silently.

**S0 — preprocessing.** The library's documented pipeline is: scale to [−1024, 1024], one
channel, centre-crop to a square, resize to 224. Measured on the 3,690 frontal images: 2,109
landscape, 759 square, 822 portrait; the crop keeps a median 82% (min 71%) of the long side. On
landscape films that removes ~9% from each lateral edge (lateral chest wall, ribs,
costophrenic angles), i.e. where effusions and rib fractures show. **Decision: crop vs.
pad-to-square (black, = air) is a second selection factor.** Candidates = 3 DenseNets × 2
preprocessings, selected together by one pre-declared rule (below). All 6 results are reported,
not only the winner.

**S0 — selection and thresholds on train.** Train is unseen data for S0 (no IU in any candidate)
and has the positives that validation lacks (results §2.5).
- Selection rule, fixed before scoring: **highest macro AUROC over the six findings on train.**
  A paired bootstrap (studies resampled) of winner − runner-up shows whether the choice is
  meaningful or within noise; either way the rule's choice stands.
- Per-finding thresholds on train by **Youden's J** (max sensitivity + specificity − 1). Reason:
  the test set is positive-enriched (plan §2), and sensitivity / specificity do not depend on
  prevalence, while an F1-optimal threshold tuned at train prevalence would be too conservative
  for the enriched test set. Tie-break: the higher threshold (higher specificity). **The same
  criterion will be used for R0 and for G's pooled threshold**, so all tuned systems are tuned
  the same way.
- Probabilities are stored as plain sigmoid outputs (the library's op_norm rescaling is
  monotone per finding, so it changes neither AUROC nor the Youden choice; it is not applied).

**S0 / R0 — which splits are scored now.** Train (selection, tuning) and validation (reported).
**Test is not scored now:** plan §6 runs every system on test once, in week 6. The scripts take
the split as an argument, so the week-6 run uses the same code with the fixed selection.

**Weights loading (security).** The library loads all weight files with
`torch.load(weights_only=False)`, which can execute code from the file. The DenseNet files are
whole pickled modules; the ResNet file is a plain state dict. Consistent with the CheXbert
labeller (`weights_only=True`, strict keys), weights are downloaded with the library's own
downloader into the project cache (`.cache/torchxrayvision/`), their SHA-256 recorded, and
loaded with `weights_only=True` (an allow-list of the needed classes for the DenseNet pickles),
then copied into a freshly built library model with a strict key check.

**R0 — design.** kNN over the existing BiomedCLIP frontal embeddings (L2-normalised, cosine).
- Score per finding = the unweighted share of the k nearest training studies that have it
  (plan §3: "vote"). Not similarity-weighted: BiomedCLIP similarities sit in a narrow band
  (median NN 0.970, results §2.3), so weights would be nearly uniform anyway.
- One k for all findings, chosen by leave-one-out on train (each train study scored from the
  other 2,803) over k ∈ {1, 3, 5, 10, 20, 30, 50, 75, 100}, by the same rule as S0: highest
  macro AUROC. Then per-finding Youden thresholds on the LOO scores at that k.
- A plain 0.5 majority vote is not used as R0's decision rule: with train prevalences of
  1.5–14%, the neighbours' majority is almost never positive, so sensitivity would be near zero.
- Leakage guards in code: LOO excludes the query itself (asserted); validation queries are
  asserted absent from the train index.

**Metrics code.** scikit-learn is not installed; scipy is. A small shared `scripts/metrics.py`
(rank-based AUROC with tie handling, threshold metrics, Youden threshold, paired bootstrap) with
an offline test against brute-force computation, rather than a new dependency. G will reuse it.

## 2026-10-17 — S0: weights, safe loading, scoring, selection

**Weights download.** The library's downloader crashed on the Windows console (cp1252 cannot
print its progress-bar character U+2588) after 1 MB and **left a truncated weights file** that
the library would then treat as already downloaded. Running with `PYTHONIOENCODING=utf-8` did
not help, because `python -I` ignores `PYTHON*` variables. Deleted the partial file. Added
`scripts/s0_weights.py`: downloads to `<name>.part`, checks the size against Content-Length,
then renames. Files and SHA-256 (in `.cache/torchxrayvision/`):

| Weights | File | Bytes | SHA-256 |
|---|---|---|---|
| densenet121-res224-chex | chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt | 28,381,998 | 1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4 |
| densenet121-res224-mimic_nb | mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt | 28,381,970 | a15dc8e9d0880d7b8184b450f03c5c0204897fa0d9ac3a3a5c638038028a65e5 |
| densenet121-res224-mimic_ch | mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt | 28,382,012 | 23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189 |
| resnet50-res512-all | pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt | 94,471,272 | 7724248ef6972be85a3e0d2fabe0a561634e11eb776d51b3999682c19fc596d9 |

**Safe loading.** `weights_only=True` cannot read the DenseNet files at all: they are legacy
(pre-zip) whole-module pickles that store `('module', class, source)` records, which the
weights-only unpickler refuses. Instead:
- Scanned every pickle opcode of the three files with `pickletools` (executes nothing): 16
  `GLOBAL` opcodes, no `STACK_GLOBAL` / `INST` / `OBJ`. The globals are only torch `nn` layers,
  the library's 4 DenseNet classes, `OrderedDict`, torch storages and tensor-rebuild helpers.
- `s0_weights.load` passes torch's legacy loader a restricted unpickler that allows exactly those
  globals (a non-allowed global, e.g. `builtins.print`, is refused, tested). The ResNet file is a
  plain state dict and loads with `weights_only=True`. Weights are copied into a freshly built
  model with a strict key check.
- Equivalence: logits from the safe path and from the library's own loader are **identical**
  (max difference 0.0) for all four weights, on random inputs.

**Preprocessing check by eye** (contact sheet, not kept): landscape, square and portrait films,
crop vs. pad. Correct orientation and intensity range ([−1024, 1024]); the crop visibly cuts the
lateral chest walls (landscape) and the apices / bases (portrait). RGB channels are
byte-identical (300 images checked), so the channel mean loses nothing.

**Scoring** (`scripts/s0_score.py train,val`): 4 weights × {crop, pad} × {train, val} →
16 files `data/runs/s0/<weights>__<prep>__<split>.csv`, sigmoid probabilities of every trained
output. Test is refused without `--test` (checked).
- **Reproducibility issue found and fixed:** re-scoring 64 validation studies differed from the
  saved file by up to 1.1e-4, and an image's score depended on which batch it was in. Cause:
  cuDNN uses TF32 for convolutions by default on this GPU. Disabled TF32 and set cuDNN to
  deterministic in `s0_score.py`; afterwards batch 64 vs. 32 differ by ≤ 1.2e-7 and GPU vs. CPU by
  ≤ 3.6e-7 (below the 6 decimals stored). Re-scored everything; results changed only in the
  third decimal of one AUROC (Fracture 0.677 → 0.676).

**Selection** (`scripts/s0_select.py`, report saved as `data/runs/s0/selection_report.md`,
choice in `data/runs/s0/selection.json`). Train macro AUROC over the six findings:

| Config | crop | pad |
|---|---|---|
| DenseNet chex | 0.691 | 0.683 |
| DenseNet mimic_nb | 0.713 | **0.728** |
| DenseNet mimic_ch | 0.711 | 0.712 |
| ResNet-50 res512 "all" (reference only; Lung Lesion = max(Nodule, Mass)) | 0.795 | 0.808 |

- **The rule selects DenseNet mimic_nb + pad** (0.728). Paired bootstrap vs. the runner-up
  candidates: −0.014 [−0.026, −0.002] (mimic_nb crop), −0.016 [−0.034, +0.003] (mimic_ch pad).
- **Padding helps where the crop cuts anatomy, as expected:** for mimic_nb, Pleural Effusion
  0.842 → 0.868 and Fracture 0.616 → 0.676; the ResNet shows the same pattern (Fracture 0.715 →
  0.746). For chex, pad is slightly worse overall.
- **The ResNet-50 reference beats every DenseNet clearly:** +0.080 [+0.052, +0.106] macro AUROC
  vs. the selected DenseNet, with large gains on Cardiomegaly (0.913 vs. 0.798), Atelectasis
  (0.819 vs. 0.734), Lung Lesion (0.697 via the proxy vs. 0.547) and Pleural Effusion (0.943 vs.
  0.868). Not a pipeline bug: both go through the same image loading and squaring code.
  **Not switched to**: as decided beforehand, this is raised as an open decision (plan §3 says
  "DenseNet"; the ResNet needs a Lung Lesion proxy). Recorded under "Open decisions" in
  `results.md`.
- The DenseNets' Lung Lesion is near chance (0.49–0.56). Their "Lung Lesion" (CheXpert / MIMIC
  report label) evidently does not transfer to IU's MeSH nodule / mass codes.
- Youden thresholds on train for the selected config; on near-chance scores Youden picks
  degenerate thresholds (Lung Lesion 0.0072: sensitivity 0.71, specificity 0.38). Validation
  results (400 studies; Fracture has only 2 positives, so its numbers mean little): macro AUROC
  0.757 [0.695, 0.802].

## 2026-10-17 — R0: leave-one-out k choice, thresholds, validation

- Checked the existing BiomedCLIP embeddings: 3,690 × 512, unit norm (so dot product = cosine).
  Checked BiomedCLIP's own preprocessing (open_clip): resize the short side to 224, then
  **centre-crop** to 224 × 224. So the retrieval index, like S0's "crop" variant, does not see the
  lateral edges of landscape films. Not changed here: plan §3 ties R0 to "the same BiomedCLIP
  index as G", and the index is part of G's design (Q2). Noted for that design.
- Wrote `scripts/r0_knn.py` (rules in its docstring, as decided in the morning entry). Leakage
  assertions pass (no study retrieves itself in leave-one-out; no validation study in the index).
  The test split is refused without `--test` (checked).
- **Grid correction.** First run: the best k was 100, the largest value in the grid, with macro
  AUROC still rising (75 → 100: 0.732 → 0.741). A maximum at the edge means the grid missed the
  rule's target, so the grid was extended to k ≤ 1000 (**train only**; no validation or test
  data involved in choosing k) and the rule re-applied.
- Leave-one-out macro AUROC by k: 1: 0.572, 3: 0.626, 5: 0.651, 10: 0.671, 20: 0.704, 30: 0.712,
  50: 0.712, 75: 0.732, 100: 0.7405, **150: 0.7413**, 200: 0.736, 300: 0.740, 500: 0.736,
  750: 0.739, 1000: 0.739. **Selected k = 150** by the rule; anything from 100 to 1,000 is
  equivalent (plateau, differences ≤ 0.006).
- **Interpretation:** R0's signal is the prevalence of a finding in a broad neighbourhood, not
  a few near-identical cases: the single nearest case gives 0.572. For G, whose prompt can hold
  only a few retrieved cases, retrieval at small k is weak evidence (k = 5: 0.651).
- Per finding (leave-one-out, train): Cardiomegaly 0.891, Pleural Effusion 0.898, Lung Opacity
  0.792, Atelectasis 0.747, **Lung Lesion 0.555 and Fracture 0.565 (near chance at every k)**:
  a whole-image embedding does not capture small focal findings.
- Thresholds (Youden, leave-one-out train): vote shares 0.013–0.100, i.e. a finding is called
  when roughly 2–15 of the 150 neighbours have it.
- Validation (400 studies): macro AUROC 0.700 [0.624, 0.777]; Cardiomegaly 0.959, Pleural
  Effusion 0.896; Lung Lesion 0.435 (11 positives) and Fracture 0.496 (2 positives) are noise.
- Outputs: `data/runs/r0/train.csv` (leave-one-out scores), `val.csv`, both with the 150
  neighbour IDs per query (for retrieval-quality and copying-rate analyses in week 7),
  `selection.json`, and the printed report `selection_report.md`.
- Side note: on train, retrieval alone (0.741, leave-one-out) already matches or beats the
  selected zero-shot DenseNet (0.728), mainly on Cardiomegaly (0.891 vs. 0.798). R0 uses IU's own
  labels through its neighbours; S0 transfers from other hospitals' data. The ResNet reference
  (0.808) beats both.

## 2026-10-17 — Write-up; next steps

- `results.md`: §2.3b (label coverage per weights file), new §5.1 (S0), §5.2 (R0), §5.3 (test
  pending), the shared threshold rule at the top of §5, two threats (§6), a new
  "Open decisions" section (S0 DenseNet vs. ResNet), two observations for G's design. No new
  deviations from the plan: R0's leave-one-out k and S0's tuning on train were already recorded
  (2026-10-16); the Youden rule and the crop / pad choice fill in details the plan left open.
- `SETUP.md`: new §6b with the S0 / R0 commands; `test_metrics.py` added to the tests.
- New files: `scripts/metrics.py`, `scripts/s0_weights.py`, `scripts/s0_score.py`,
  `scripts/s0_select.py`, `scripts/r0_knn.py`, `scripts/tests/test_metrics.py`. Outputs (not in
  git): `data/runs/s0/`, `data/runs/r0/`, weights in `.cache/torchxrayvision/`.

**Next steps (replaces items 4 and 5 of the 2026-10-16 list; the rest of that list stands)**
1. **Decide S0** (results "Open decisions" 1): DenseNet mimic_nb + pad (selected by the rule) or
   the ResNet-50 + pad with Lung Lesion = max(Nodule, Mass). If the ResNet: add it as the
   selected config in `s0_select.py`, re-run, record a deviation (plan §3 says DenseNet).
2. When G's index is designed (Q2): decide crop vs. padded BiomedCLIP embeddings; if padded,
   re-embed and re-run `r0_knn.py` so R0 keeps sharing G's index.
3. Week 6: `s0_score.py test --test`, then evaluate S0 and R0 on test with the fixed choices and
   thresholds (an evaluation step for test still has to be written; the selection scripts only
   report train and validation).

## 2026-10-17 — Decision: S0 stays the DenseNet

- Decision: **no further deviation from the plan**, so S0 stays **DenseNet mimic_nb, padded**,
  as selected by the rule. `data/runs/s0/selection.json` already holds this choice; nothing
  re-run. The ResNet-50 remains a reported-only reference in results §5.1 and is not run on test.
- `results.md`: "Open decisions" → none open; the question moved to a new "Resolved decisions"
  section with the decision. New threat in §6: S0 is not the strongest available image model
  (ResNet-50 0.808 vs. 0.728 on train), so beating S0 does not mean beating the best single image
  model; the Q3 write-up must say so next to every S0 comparison.
- Next-steps item 1 of the previous entry is done; items 2 and 3 stand.

## 2026-10-17 — Paid API available; spend metering; P1 malformed function calls

- The Gemini key now has billing, with a **monthly cap of USD 45**. Spending is to be kept to
  a minimum: every run gets a `--max-requests` budget, and a cost projection is reported after
  the pilot before any larger run (2026-10-16 next-steps item 12).
- Prices re-checked on ai.google.dev/gemini-api/docs/pricing: `gemini-3-flash-preview` $0.50 /
  $3.00 per 1M input / output tokens (thinking billed as output), Batch half price;
  `gemini-embedding-2` $0.00012 per image ($0.00006 batch). Unchanged from 2026-10-16.
- **P1 2-study test** (`--limit 2 --max-requests 10`): the first study (CXR8) failed with
  `MALFORMED_FUNCTION_CALL`. P1's verbatim prompt says "IMPORTANT: Use the DuckDuckGo search
  tool…", but with web search off (plan §3) the agent has no tool, and the model tries to call
  one anyway.
- **Hidden calls found:** agno retries a malformed function call once internally ("retry with
  guidance": it re-sends the request with an added "the previous function call was malformed"
  message; `retry_with_guidance_limit = 1`). So one runner attempt was **2 billed calls**, the
  request budget counted 1, and neither call was cached (no record of its cost).
- **Fix: metered client** (`common.metered_client`). Both runners now use one google-genai
  client whose `generate_content` / `embed_content` count every real HTTP call against the
  budget (`count_request`) and append it to `data/runs/logs/api_calls.jsonl` with arm, split,
  study, finish reason and token usage, including failed calls and agno's internal ones. agno
  gets it via `Gemini(client=...)`; the client also carries the 180 s timeout. `with_retry` no
  longer counts (the budget now means billed calls, not loop attempts). `tests/test_retry.py`
  updated and extended (fake API: logging, library-internal retries counted, failed calls
  logged without usage): 12/12 pass.
- **`scripts/spend.py`**: the ledger, from the call log (exact token counts × prices; failed HTTP
  calls carry no usage and are not billed). The 2 calls made before the log existed are entered
  at the cost of their logged identical repeat ($0.0015).
- **Probe, 5 pilot studies** (`--limit 5 --max-requests 12`, 8 calls): **3 of 5 studies end in
  `MALFORMED_FUNCTION_CALL`** after agno's guidance retry (CXR8, CXR205, CXR271); 2 answer
  normally (CXR86, CXR164). A malformed call bills only its input (~1,500 tokens, ≈ $0.0007);
  an answer costs ≈ $0.007 (1,452 input; 1,000 output + 500–1,100 thinking). Spend so far:
  **$0.018**.
- **Decision (follows the plan, no deviation):** web search stays off (plan §3). A malformed
  function call is P1's own behaviour, so it is **cached as the study's outcome**
  (`"outcome": "malformed_function_call"`) and the run continues, like a refusal (plan §3:
  "refusals are recorded, not prompted away"). Real API errors (429, 5xx, timeouts) still retry
  or stop as before. **Scoring rule, fixed now before any scoring:** a malformed outcome counts
  as "no findings reported" (all six negative), not as a missing study; excluding those studies
  would overstate P1. Its rate is reported as a Q1 result.
- Q1 observation: when P1 answers, section 5 still lists "references" with URLs (Radiopaedia,
  StatPearls, ACR) although no search tool exists. They come from the model's memory and are
  unverified. To be checked in the Q1 write-up (invented content, plan §5).
- **P1-adapted pilot done** (50 validation studies, `--pilot --max-requests 100`). The 5 probe
  studies were already cached and skipped; this run made 73 billed calls for the other 45. All
  50 studies together: 81 logged calls + 2 made before the log existed = 83 (corrected
  2026-10-17; an earlier version of this line wrongly said "73 runner attempts = 81 billed
  calls"). **31 / 50 studies (62%) ended in `MALFORMED_FUNCTION_CALL`**, 19
  answered. No 429 / 503 / retries (the 2026-10-16 agno 503 problem did not recur). Answers
  average 1,457 input / 1,090 output / 987 thinking tokens (≈ $0.0067). Spend so far: $0.18.
- Started the P2-adapted train index (`p2_adapted.py index --max-requests 2900`; 2,782 images
  still to embed, ≈ $0.33).

## 2026-10-17 — Diary audit: items missing from today's entries

Checked today's entries against everything done and every file written today. Added here:
- **Commits pushed:** `8745a69` (S0, R0, metrics, write-up) and `b891556` (decision: S0 stays
  the DenseNet). The spend-metering and P1 work is not yet committed (P2 index still running).
- **`results.md` §3.3 added:** P1-adapted pilot (malformed-call rate, handling and scoring
  rule, cost, invented references) and a note on cost tracking.
- **`scripts/spend.py` had a first version** that totalled token usage from the cached result
  files (`data/runs/<arm>/<split>/*.json`) and counted embedding files by date. It was replaced
  the same hour by the call-log version, because the cache misses billed calls that ended in an
  error and agno's internal retries.
- **`scripts/starters/p2_adapted.py`:** besides switching to the metered client and tagging
  calls (`set_call_context`), the now-unused `from google import genai` import was removed.
  No change to what P2 sends or retrieves.
- **`scripts/tests/test_metrics.py`:** one check compared floats exactly and failed on rounding
  ((1 − 0.95)/2 = 0.025000000000000022); changed to an approximate comparison. Test code only.
- **Outputs written today, not listed before:**
  - `data/runs/logs/api_calls.jsonl`: the metered call log (one line per API call);
  - `data/runs/logs/p1_test2.log`, `p1_probe5.log`, `p1_pilot.log`: console output of the P1
    2-study test, 5-study probe and pilot;
  - `data/runs/logs/p2_index.log`: console output of the P2 index build (running);
  - `data/runs/p1_adapted/val/*.json`: 50 cached P1 outcomes (19 answers, 31 malformed);
  - `data/runs/p2_adapted/embeddings/*.npy`: train-image embeddings (being added by the index run);
  - S0 / R0 outputs were listed in their entries. Their first versions (S0 with TF32 on; R0
    with the k ≤ 100 grid) were overwritten by the re-runs described there.
- **Deleted today:** only the truncated weights file (described in the S0 entry). No data,
  labels, splits or cached results were deleted or edited by hand.
- **Not project data:** a preprocessing contact sheet in the session's temporary folder (not
  kept; described in the S0 entry).

## 2026-10-17 — P2-adapted train index built

- `p2_adapted.py index --max-requests 2900`: 2,784 embedding calls (= 2,804 train studies − 20
  embedded in the 2026-10-16 smoke test), no failures, no retries, ~28 min. Cost $0.33.
- Checks: 2,804 / 2,804 train vectors present; all 3,072-dimensional, norm 1.00000, no NaN, no
  exact duplicates. The folder also holds CXR8 and CXR86, validation *query* vectors from the
  smoke test; `load_index` reads train IDs only, so they cannot enter the index.
- Gemini Embedding 2 also packs chest X-rays into a narrow similarity band (median nearest-
  neighbour similarity within train 0.979, max 0.992), like BiomedCLIP (results §2.3).
- Spend to date: $0.51.

## 2026-10-17 — P2-adapted pilot

- `p2_adapted.py run val --pilot --max-requests 110`: 98 calls (48 query embeddings, 2 reused
  from the smoke test; 50 generations), no failures, no retries.
- **50 / 50 studies answered**; no empty answers; no refusal-like phrasing (regex check for
  "cannot provide / diagnose", "consult a doctor"; to be confirmed by reading in Q1). All answered
  by `gemini-3-flash-preview`.
- Tokens per study (mean): 2,268 input / 495 output / 2,163 thinking → **≈ $0.0091 per study**
  (the 2-study smoke test suggested $0.007; thinking is the larger share).
- Top-1 retrieval similarity (Gemini Embedding 2): min 0.956, median 0.981, max 0.992.
- Pilot cost $0.46; **spend to date $0.98** (2.2% of the monthly cap).
- The earlier smoke-test answers (`data/runs/p2_adapted_smoke/val/`, partial index) remain
  separate and are not results.

## 2026-10-17 — P3 source review; Google ADK installed

- Read `rag_tutorials/multimodal_agentic_rag/backend/` (`rag_store.py`, `agent.py`, `server.py`,
  `app_state.py`). What the runner must reproduce:
  - **Index:** per source, `add_file_source` embeds the image (`gemini-embedding-2`,
    `output_dimensionality=768`, no task prefix) and an annotation text `"<title>. <notes>"`
    (prefix `"task: retrieval document"`), blends them 0.68 / 0.32 and renormalises. 2
    embedding calls per train study.
  - **Answering (`/ask`):** the server first runs `RAG_STORE.search(question, top_k=6)` (request
    default 6), then builds the agent with its retrieval tool replaced by one that returns that
    precomputed packet, whatever query text the model passes. The agent receives text only:
    `"Question: {question}\nUse the retrieval tool result for this exact question."` It also has
    `inspect_embedding_space` (counts of sources, dimensions). Model `gemini-3-flash-preview`,
    temperature 0.25, max 900 output tokens.
  - The depth was first misread as 5 (the agent module's own tool default); the server path
    overrides it, so **top-k = 6 as the plan says**.
- **Naming: stays P3-orig.** Model, embedder, dimensions, blend weight, top-k, agent prompt and
  tools are all kept. Adaptations (minimal, as listed in plan §3): retrieval by the patient's
  image embedding instead of a text query; store and agent called directly, no server / UI;
  index = train split with each study's report as annotation.
- **Installed `google-adk` 2.12.0** (starter requires `>=1.0.0`; major-version jump, as with
  agno). Adds 18 packages (fastapi, starlette, uvicorn, opentelemetry, jsonschema, …) and
  **downgrades `websockets` 16.1.1 → 15.0.1**. Checked beforehand: `websockets` is used only by
  google-genai's Live API (not used here; google-genai accepts 13–16.x) and by agno extras that
  are not installed; no project script uses it. google-genai, torch, agno, numpy and scipy are
  unchanged, so no existing result is affected.
- After the install: `pip check` reports no broken requirements; `test_metrics.py` and
  `test_retry.py` pass; P1, P2, S0 and metrics modules import; google-genai 2.31.0 and agno 3.1.4
  unchanged. `requirements.txt`: added `google-adk==2.12.0` and `websockets==15.0.1`.

## 2026-10-17 — P3-orig runner written; metering extended to async calls

- **Metering extended** (`common.py`): Google ADK calls the *async* API
  (`client.aio.models.generate_content`), which the meter did not cover. Added `_meter_async`;
  the sync and async paths share the logging. Paths that are not metered (streaming
  generation, async embedding) now **refuse to run** instead of bypassing the budget. ADK's
  model gets the pre-configured metered client (`Gemini(model=..., client=...)`, ADK's documented
  hook), which also keeps ADK's own HTTP retry options out (they would be invisible to the
  budget).
- **Embedding log entries now say what was embedded:** images vs. text characters. Text
  embeddings are billed per token ($0.20 / 1M) and the response has no token count, so
  `spend.py` estimates tokens as characters / 4 (fractions of a cent). Older entries (P2, images
  only) are priced as before; the ledger total was unchanged ($0.9751) after the change.
- `tests/test_retry.py`: +4 checks (async calls metered and counted; unmetered paths refuse;
  embed item split). 16/16 pass.
- **`scripts/starters/p3_orig.py`** (design and adaptations in its docstring): imports and
  calls the starter's own `rag_store`, `app_state.RAG_STORE` and `build_agent`. Index entries
  are built as `add_file_source` builds them (image + annotation embeddings, blended 0.68 /
  0.32); title = the PNG file name, notes = the study's FINDINGS + IMPRESSION (the text CheXbert
  labelled, with section names); vectors cached in `data/runs/p3_orig/index/<study>.npz`.
  Answering follows server.py's `/ask`: top-6 retrieval first (by the patient's image
  embedding, cached in `data/runs/p3_orig/query_embeddings/`), then the agent with the
  substituted retrieval tool and the shipped user message. The UI's 3-D PCA projection is
  skipped (it does not affect the matches). The generator never sees the patient's image.
- Offline checks (0 API calls): report notes correct; agent as shipped (gemini-3-flash-preview,
  temperature 0.25, max 900 output tokens, both tools); ADK uses the metered client (identity
  check); `search_by_vector` reproduces a brute-force ranking with the shipped scoring on
  synthetic vectors.
- **5-study index check** (`index --limit 5 --max-requests 12`): 10 calls (1 image + 1 text per
  study, annotation 216–379 characters), vectors 768-d with norm 1.0, inline path; $0.0007.
  Spend to date $0.976.

## 2026-10-17 — P3-orig index and 2-study agent test

- **Full index** (`p3_orig.py index --max-requests 5700`): 5,598 calls (2 per study for the
  2,799 not yet indexed), no failures or retries, ~60 min, $0.37. Checks: 2,804 / 2,804 train
  entries, no non-train entries; image and annotation vectors 768-d, norm 1.0, no NaN; all
  images inline; no exact duplicates (median nearest-neighbour similarity 0.980 for images,
  0.949 for annotations; every annotation starts with its own file name).
- **2-study agent test** (`run val --pilot --limit 2`):
  - CXR8 completed: 1 query embedding + 3 generation calls (inspect tool → retrieval tool →
    answer; inputs 372 → 582 → 1,301 tokens), $0.0023. Answer correct (no findings; ground
    truth none) and phrased about the retrieved cases ("The retrieved reports … indicate that
    none of the requested findings are present"), as expected for a generator that never sees
    the patient's image.
  - **Bug found and fixed:** CXR86 failed with `RuntimeError: Event loop is closed` before any
    billed generation. The runner starts one `asyncio.run` loop per study, but the metered
    client's async connections belonged to the first study's loop (ADK's documentation warns
    about pinning one client to a model across loops). Fix: `run_agent` creates its own metered
    client per study (no network call; same budget counter and log). The failed call is logged
    as not ok and not billed; the run stopped instead of retrying (unknown error), as intended.
  - Rerun: CXR86 completed with the same 3-call pattern, $0.0040 (584 thinking tokens on the
    answer). Cached query embedding reused.
  - Log notice from ADK 2.12: "[EXPERIMENTAL] feature JSON_SCHEMA_FOR_FUNC_DECL is enabled"
    (ADK's default way of declaring tools to the model; not set in the project code).
- Spend to date $1.37. Started the P3 pilot for the other 48 studies (`--max-requests 220`,
  ≈ 192 calls, ≈ $0.17).

## 2026-10-17 — P3-orig pilot

- `p3_orig.py run val --pilot --max-requests 220`: 191 calls (48 query embeddings, 143
  generations) for the 48 remaining studies; no failures, no retries.
- **50 / 50 answered**; no empty answers. Every study: `inspect_embedding_space` then
  `retrieve_relevant_context` (49 in 3 calls; CXR3345 called both tools in one turn, 2 calls).
  Cost **≈ $0.0030 per study** (max $0.0047); median answer 851 characters.
- **4 / 50 answers (8%) are truncated mid-sentence** (finish reason `MAX_TOKENS`; CXR513,
  CXR2019, CXR2039, CXR2843). The shipped config sets `max_output_tokens=900`, and on
  gemini-3-flash-preview thinking counts toward it: those answers used 733–863 thinking tokens,
  leaving 33–163 for the text. Kept as shipped (plan §3: minimal adaptation); the truncated
  text is scored as written and the truncation rate reported as a Q1 result.
- Spend to date **$1.51** (3.4% of the cap).

## 2026-10-17 — Cost projection from the three pilots (next-steps item 12)

Measured cost per study (generation + query embedding): P1-adapted **$0.0036** (38% answers at
$0.0067, 62% malformed at ~$0.0015 for 2 input-only calls), P2-adapted **$0.0092**, P3-orig
**$0.0031**. One-off indexes already paid (P2 $0.33, P3 $0.38).

| Remaining run | Studies | Estimate |
|---|---|---|
| Starters on test (P1, P2, P3) | 3 × 400 | $6.4 ($1.4 + $3.7 + $1.2) |
| P1 web-search sensitivity arm (plan §3) | 50 | ~$0.5–1.5 (tool loop + search results; to measure on 2 studies first) |
| G: prompt development on the pilot | ~4 × 50 | ~$2 |
| G and G−ind on validation (threshold) and test | 4 × 400 | ~$16 (G not built; assumed ≈ $0.010 per study, P2-like) |
| **Core total** | | **≈ $25** → with $1.51 spent, **≈ $27 (60% of the cap)** |
| Optional: G+S0 (plan §3, optional) | 2 × 400 | ~$8 |
| Optional: G-img (open decision, 2026-10-16) | 2 × 400 | ~$8 |

- Core plan fits the USD 45 cap with ~$18 margin. Both optional arms together would bring the
  total to ≈ $43, too close to the cap.
- The Batch API (half price) could save ~$6–8 on the large test / validation runs, but needs
  new batch code; not needed for the core plan.
- G's per-study cost is the main uncertainty; re-project after G's first pilot run.

## 2026-10-17 — Decision: P1-adapted-ns replaces P1-adapted

- Decision: P1-adapted (verbatim prompt, search off) fails on 62% of pilot studies, so it is not
  run further; its pilot results stay as evidence (`data/runs/p1_adapted/`, results §3.3).
  **New arm P1-adapted-ns:** the whole of section 5 ("Research Context") is removed from the
  prompt; sections 1–4, the closing format line, the appended findings question, agno, the
  width-500 resize and `gemini-3-flash-preview` are unchanged. Purpose: one system that is
  image + fixed prompt → answer, without the tool conflict.
- Why the whole section rather than only the "IMPORTANT: Use the DuckDuckGo search tool" line:
  its remaining bullets (literature, links, 2–3 references) only make sense with search and
  would invite invented references.
- Discussed: the remaining sections 1–4 shape the answer (role, structure, severity, primary
  diagnosis with confidence, differential diagnoses, patient explanation). Differentials may be
  read as positives by CheXbert (plan §5 risk), to be measured in the per-system hand check. A
  truly minimal prompt (image + findings question only) was discussed as a different,
  non-starter arm; not chosen.
- Code: `p1_adapted.py --no-search-section` (arm folder `p1_adapted_ns`). `SECTION_5` holds the
  section exactly as in the verbatim prompt; an assertion checks it occurs once; the new prompt
  equals the verbatim prompt with that block removed (8 lines; checked: no DuckDuckGo or
  "Research Context" left). Cached records gain `prompt_variant`. `--web-search` and
  `--no-search-section` cannot be combined.
- `results.md`: §3.3 explains the new arm; deviations table +1 row; "Open decisions": whether to
  keep the plan's 50-study web-search arm (P1 as shipped).

## 2026-10-17 — P1-adapted-ns pilot

- `p1_adapted.py val --pilot --no-search-section --max-requests 60`: **50 calls, 50 / 50
  answered, 0 malformed function calls**, all finish reason STOP, no retries.
- Mean 1,390 input / 720 output / 1,110 thinking tokens → $0.0062 per study; pilot $0.31.
- All answers have sections 1–4 (median 2,939 characters); none contains web links (the
  verbatim arm's answers had references invented from memory).
- Cost projection updated: P1 on test ≈ 400 × $0.0062 ≈ $2.5 (was $1.4 for P1-adapted); core
  total ≈ $28.5, still ≈ 63% of the cap.
- Spend to date **$1.82** (4.1% of the cap). `results.md` §3.3: comparison table added.

## 2026-10-17 — Output labelling built; CheXbert fails on the answers' format

- `chexbert_labels.py`: model loading moved into `load_model()` (reused). Verified: re-labelling
  all 3,955 reports into a scratch file gives a **byte-identical** `chexbert_labels.csv`.
- **Labelling rules fixed before seeing labels** (`scripts/label_outputs.py`): malformed outcome
  → empty text → all negative; remove the markdown characters `# * _ ` |`; split answers into
  sentence chunks of ≤ 512 tokens and combine (positive > uncertain > negative > blank), because
  CheXbert reads at most 512 tokens and plain truncation would cut **all 50 P1-ns answers and
  30 / 50 P2 answers** (P3: none); positive only; same heads as the ground-truth check. Checked on
  all 200 pilot answers: longest chunk exactly 512 tokens, chunks reproduce the text exactly.
- **Bug avoided:** malformed P1-adapted records store agno's *error message* as
  `response_text`; labelling it would have been wrong. `answer_text()` maps them to "".
- Labelled the pilots of p1_adapted, p1_adapted_ns, p2_adapted, p3_orig →
  `data/runs/<arm>/val_labels.csv`. `scripts/pilot_summary.py` → `data/runs/pilot_summary.md`.
- **Result looked implausible** (P2 "calls" Atelectasis in 45 / 50 studies, P3 Cardiomegaly in
  41 / 50, against 1 and 2 true positives). Direct test of CheXbert:
  "Cardiomegaly: Not present." → **positive**; "Atelectasis: Not present." → positive;
  "Lung Opacity: Absent." → positive; "No cardiomegaly." → negative (correct).
  CheXbert (trained on prose reports) does not understand a "Finding: status" format, nor
  lists such as P2's "none of the specific findings listed — Lung Opacity, Cardiomegaly, … —
  are present". The appended question invites both formats: "Finding: not present"-style lines
  in 42 / 50 P1-ns, 9 / 50 P2 and 8 / 50 P3 answers (regex; undercounts the list form).
- **Conclusion: the CheXbert labels of the starters' answers are invalid; `pilot_summary.md`
  measures mainly labeller errors.** Files kept as evidence. This is what the plan's per-system
  hand check of the labeller exists to catch (plan §5, §7). Stopped for a decision on the
  labelling method before any further scoring.

## 2026-10-17 — Decision: option B (report-style rewrite before CheXbert)

**Options considered** for labelling the starters' answers after CheXbert failed on their format:
A. LLM extractor (Gemini reads each answer, returns present / absent / uncertain per finding);
B. deterministic rewrite rules, then CheXbert as planned; C. hand-label everything (not feasible
for ~1,600 test answers); D. reword the appended findings question so answers come out
report-style. D rejected: it changes the systems to suit the measuring tool, does not guarantee
compliance (P1-ns's prompt still demands differentials; P3's agent prompt demands bullets),
needs reruns (~$1) and another deviation. **Decision: B**, with A as fallback if B's checks fail.

**What B is:** a deterministic pre-processing step that rewrites a *copy* of each answer into
the sentence forms CheXbert was trained on, then CheXbert labels it exactly as before (same
heads, positive only, chunking, markdown clean-up).
- **The raw answers are never modified.** Cache files (`data/runs/<arm>/<split>/<study>.json`)
  stay exactly as the API returned them; plan §8 needs the raw text.
- **Traceability:** for every answer the rewritten text and the rules that fired are saved next
  to the labels, so each label can be traced raw answer → rewrite → CheXbert label.

**Rule design (form only, never meaning):**
1. Status lines "<finding> <separator> <verdict>" (separators `:`, `–`, `-`, bullets) → report
   form: negative verdict ("not present", "absent", "none", "no", "not seen / identified /
   detected", "normal") → "No <finding>."; positive verdict ("present", "yes", severity words)
   → "<Finding> is present."; uncertain verdict ("possible", "borderline", "cannot be excluded")
   → "Possible <finding>." Text after the verdict (the model's explanation) is kept.
2. Negated lists ("none of the findings listed — A, B, … — are present", "no evidence of A, B or
   C") → "No A. No B. No C."
3. Everything else unchanged.
4. **Only unambiguous verdicts are rewritten.** Mixed or hedged verdicts are left as they are and
   flagged for manual review, never guessed.

**Checks:**
- Rules written on the **pilot (validation) answers only**, then **frozen before any test-set
  answer is labelled**.
- **Residual check:** after rewriting, flag any finding name still in a non-report pattern (e.g.
  still followed by a colon or inside a list); each flag is a missing rule or goes to review.
- **Reports unchanged:** the rewrite must not change the labels of the IU reports, so the
  CheXbert-vs-ground-truth agreement (results §2.4b) still applies.
- **Hand check (plan §5):** ~20 answers per system, labels compared with the *raw* answer.
- **Known limit:** differential-diagnosis lists (P1-ns section 3) name conditions as possible;
  CheXbert may read them as positive. Measured in the hand check; any rule for it is decided
  separately (it is close to system-specific).
- Fallback to A if the residual or hand check shows too many misses.
- To be recorded in `results.md` as a deviation once adopted (an added pre-processing step).

## 2026-10-17 — Option B checked before implementation: not sufficient

Before writing rules, sampled the answer lines that name a finding (P1-ns 426, P2 302, P3 267
lines) and tested one real sentence of each type through CheXbert. Results (correct label in
brackets):
- Status lines: "Cardiomegaly: Not present" POS [neg]; "Fracture: Not present." neg [neg];
  "Cardiomegaly: Equivocal (…)" POS [uncertain]; "Atelectasis: There are no signs of atelectasis,
  which is …" POS [neg] — inconsistent.
- **Definitions / general knowledge** (P2): "Cardiomegaly refers to an enlarged heart …" POS;
  "Lung Opacity refers to any area …" POS; "Atelectasis refers to … linear opacities" →
  Atelectasis POS and Lung Opacity POS; "cardiomegaly is a sign of an underlying condition" POS
  [all: not about the patient].
- P3: "Pleural Effusion was specifically ruled out …" POS [neg]; "… mentioned as possibilities …"
  POS [uncertain]; "Atelectasis was not explicitly mentioned …" uncertain [neg / not stated].
- Negated list "None of the findings listed, Lung Opacity, Cardiomegaly, … are present":
  Cardiomegaly POS, others neg [all neg].
- Handled correctly: ordinary prose ("no signs of collapsed lungs (atelectasis)", "(no
  cardiomegaly)", "A subtle, ill-defined Lung Opacity is visible …", positive summaries).
- **Prevalence of definition / explanation sentences naming a finding** (keyword cues such as
  "refers to", "is defined as", "can be caused by"): **P2 38 / 50 answers**, P1-ns 3 / 50, P3
  0 / 50. Cause: P2's shipped prompt asks to "Be as elaborate as possible giving extra relevant
  information".
- **Conclusion:** B as decided ("form only, never meaning; only unambiguous rewrites") cannot
  handle these: deciding whether a sentence is about the patient or a textbook definition, or
  that "ruled out" negates, is a judgement of meaning. Rules for it would be keyword heuristics,
  brittle and hard to defend. No code written for B. Stopped for a decision; per the morning
  decision the fallback is A (LLM extractor), which costs money and is a different labeller,
  so it is not switched to without asking.

## 2026-10-17 — Decision: systems state their own verdicts (question v2); CheXbert dropped for answers

- Discussion: why prompt changes were avoided, yet a second Gemini call (option A)
  would read Gemini's answers. Conclusion: the starters' **own prompts** stay verbatim (Q1:
  as shipped); the findings question was always a project addition. A *strict machine-readable*
  verdict request differs from asking for report-style prose (option D): non-compliance is
  visible to a parser and can be counted, instead of being misread silently by a labeller.
- **Decision:** the appended findings question gets a format instruction (**question v2**); each
  system's answer starts with its own verdict per finding, parsed directly. No CheXbert and no
  extractor for the starters' answers. Supersedes options A and B.
- **Question v2** (`common.QUESTION_V2`; v1 kept as `QUESTION`):
  "Which of these findings are present: Lung Opacity, Cardiomegaly, Atelectasis, Lung Lesion,
  Pleural Effusion, Fracture? Begin your answer with exactly six lines, one per finding, in the
  form "<finding>: present", "<finding>: absent" or "<finding>: uncertain". Then continue as
  instructed." The lines come first so P3's end-truncation cannot cut them. It goes where the
  question went before: appended after P1's prompt (P1 has no question input), into P2's
  `Question: {question}` slot, into P3's user message `Question: {question}`.
- **Parsing rules, fixed before any v2 answer is seen** (`scripts/parse_verdicts.py`):
  markdown characters `# * _ ` |` and bullet markers ignored; a verdict line is
  "<finding> <: or dash> <present|absent|uncertain>" (case-insensitive). Per finding: exactly one
  verdict → that verdict; none → `missing`; several that disagree → `conflict`. Binary: present →
  1; absent, uncertain (plan §5), missing, conflict → 0. Missing and conflict rates are reported
  per system. A malformed-call outcome → all `missing`. Whether the six lines come first is
  recorded (compliance).
- **Runs:** new arm folders with suffix `__q2` (`p1_adapted_ns__q2`, `p2_adapted__q2`,
  `p3_orig__q2`); v2 is the runners' default, `--question v1` reproduces the old runs. The v1
  pilot answers, their CheXbert labels and `pilot_summary.md` stay as evidence. P1-adapted
  (verbatim) is not re-run. Indexes and cached query embeddings are reused (no new embedding
  cost). Each arm: 2-study check first, then the rest of the pilot; ≈ $0.9 in total.
- **Checks:** parser unit test (offline); for ~20 answers per system, verdict lines compared
  with the rest of the answer (consistency); missing / conflict rates.
- **Effect on G:** G will state its findings in its JSON; the plan's separate CheXbert path for
  G's `impression` is no longer needed for the starter comparison (all systems scored on their
  own stated verdicts). CheXbert stays validated on the reports (results §2.4b).

## 2026-10-18 — Question v2 implemented and checked

- `common.py`: `QUESTION_V2` (text as decided), `question(version)`, `arm_name(base, version)`
  (v2 answers in `<arm>__q2`). All three runners: `--question v1|v2` (default v2); cached
  records gain `question_version`. P1: the prompt is built as (prompt [minus section 5]) +
  question; for v1 an assertion checks it equals the original prompt. P2 / P3: indexes and query
  embeddings stay in the base arm folder (reused, no new embedding cost); only answers go to
  `__q2`.
- `scripts/parse_verdicts.py` (rules as recorded) and `tests/test_parse_verdicts.py`: 9/9 pass.
- v1 paths intact: re-running all three v1 pilots with `--max-requests 0` found every study
  cached (0 calls).
- **2-study v2 checks** (CXR8, CXR86; 10 calls, $0.035): all 6 answers begin with the six verdict
  lines; parsed with 0 missing, 0 conflicts; the rest of each answer follows the starter's own
  prompt (P1 sections, P2 prose, P3 short answer). P3 called CXR86 "Lung Opacity: present"
  because one of its six retrieved cases had interstitial opacities (its design: it answers
  from the neighbours' reports).
- Started the remaining 48 pilot studies for the three arms (caps 55 / 55 / 160 calls, ≈ $0.9).

## 2026-10-18 — Rename "question v2" → "checklist question"; checklist pilots

- **Rename:** "q2" collided with the plan's research questions Q1–Q3, and
  "verdict" was unclear. Now: **checklist question** (`common.QUESTION_CHECKLIST`), the six lines
  are **checklist lines**, folders `<arm>__checklist`, runner option
  `--question original|checklist` (default checklist), `scripts/parse_checklist.py` →
  `<split>_checklist.csv` (columns `checklist_<finding>`), `tests/test_parse_checklist.py`,
  `data/runs/pilot_summary_checklist.md`.
- Folders `p1_adapted_ns__q2`, `p2_adapted__q2`, `p3_orig__q2` **moved** to `…__checklist`; the
  cached answers inside were not edited. The derived `val_verdicts.csv` files were deleted and
  regenerated as `val_checklist.csv`. **Not renamed (raw records):** the cached answers still say
  `"arm": "…__q2"` and `"question_version": "v2"`, and the API call log keeps the tags
  `…__q2` (so `spend.py` lists those calls under `__q2`). Mapping: q2 / v2 = checklist.
- Checks after the rename: parser test 9/9, retry test 16/16; all six runner × question
  combinations find every pilot study cached (0 calls).
- **Checklist pilots done** (all 50 studies per arm; this run 48 + 48 + 136 calls, no retries;
  checklist pilots $1.02 in total incl. the 2-study checks). Spend to date **$2.84** (6.3%).
- Parsing: **P1-ns and P2: checklist at the top in 50 / 50, 0 missing, 0 conflicts.**
  **P3: at the top in 9 / 50; 44 of 300 entries missing (40 answers incomplete).**
- **Cause (P3):** the answering call ended with `MAX_TOKENS` in **45 / 50** studies (original
  question: 4 / 50). With the checklist question the model thinks much more on the answering
  call (median 861 thinking tokens, before 0) and P3's shipped `max_output_tokens=900` includes
  thinking, leaving ~35 tokens: the checklist itself is cut off (e.g. CXR205 ends after
  "Pleural Effusion: absent"). P1 and P2 have no output cap.
- **P3's checklist results are not usable as they are:** missing entries count as absent, which
  would penalise P3 for an artefact of the checklist question plus its cap. Stopped for a decision.
- P1-ns / P2 pilot behaviour (50 studies, 10 positive findings; behaviour only): P1-ns called
  31 findings, 5 correct, 34 / 42 normal studies called normal; P2 called 11, 3 correct, 40 / 42
  normal. (Table: `data/runs/pilot_summary_checklist.md`.)

## 2026-10-18 — Decision: P3-adapted (output cap lifted)

- Options discussed for P3's truncated checklists: A lift the 900-token output cap; B keep the
  cap and score missing entries as absent (unfair: an artefact of the checklist question + the cap);
  C reduce thinking (changes how it reasons); D keep P3 on the original question (back to the
  labeller problem). **Decision: A.**
- **Change:** the agent's `max_output_tokens` (900 in `agent.py`) is removed, i.e. the model's
  default limit, as for P1 and P2, which set none. Everything else as shipped (temperature 0.25,
  instruction, tools, retrieval, index).
- **Renamed P3-orig → P3-adapted**: the plan renames only for core components and a
  token cap is not one, but the change affects the output (it also removes the 8% truncation
  seen with the original question), so the name says it is not the shipped configuration.
- Code: the runner becomes `scripts/starters/p3_adapted.py` (git mv); default = cap lifted, arm
  folders `p3_adapted…`; `--shipped-cap` reproduces P3-orig (arm folders `p3_orig…`). The index
  and query embeddings stay in `data/runs/p3_orig/` (unchanged components, reused; no new
  embedding cost).
- Kept as evidence: `p3_orig/val` (original question, 4 / 50 truncated) and
  `p3_orig__checklist/val` (checklist, 45 / 50 truncated).
- Next: 2-study check, then P3-adapted checklist pilot (≈ $0.3).
- Code done: `p3_orig.py` → `p3_adapted.py` (git mv); `--shipped-cap` = P3-orig; an assertion
  checks the shipped config has 900 before it is lifted; records gain `max_output_tokens`.
  Both P3-orig variants still found cached (0 calls).
- **2-study check** (CXR8, CXR86; 6 calls): checklists complete and at the top, all calls end
  with STOP. Without the cap the answering call thinks much more (1,845 and 4,604 tokens), so
  P3-adapted costs ≈ $0.007–0.015 per study (P3-orig ≈ $0.003). Projection: rest of the pilot
  ≈ $0.5 (estimated $0.3 before); P3 on test ≈ $4 (was $1.2). Still within the margin.
- Started the remaining 48 studies (cap 160 calls). `pilot_summary.py` now uses
  `p3_adapted__checklist`.

## 2026-10-18 — P3-adapted pilot; cost tail from runaway thinking

- P3-adapted checklist pilot complete (this run 141 calls; 3 Google-side
  `504 DEADLINE_EXCEEDED` on answering calls, each retried after 60 s, not billed).
- **All three starters now: checklist at the top in 50 / 50, 0 missing, 0 conflicts**
  (`data/runs/<arm>/val_checklist.csv`; summary regenerated in
  `data/runs/pilot_summary_checklist.md`).
- P3-adapted answering-call thinking: median 1,959, p75 3,375, p90 7,493 tokens; **3 / 50
  studies (CXR1995, CXR3693, CXR2843) hit ~62,912 thinking tokens** — the model's own thinking
  maximum (a runaway), each ≈ $0.19; their answers are complete and normal. Cost per study:
  median $0.0087, **mean $0.021**; pilot $1.05. (P1-ns and P2 with the checklist question show no
  such tail: see the maxima printed with this entry's check.)
- This confirms that any fixed output cap near typical thinking would cut many answers
  (a 1,500 cap would have cut most of them).
- **Cost projection updated:** P3-adapted on test ≈ 400 × $0.021 ≈ $8.4 (runaways ≈ half of
  it). Spent $3.89. Remaining core: P1-ns test $2.5, P2 test $3.7, P3-adapted test $8.4, G
  (development, G / G−ind on validation and test) ~$18 → **≈ $36.5 in total (≈ 81% of the
  cap)**. G's cost is the main uncertainty; its thinking can be set by design.

## 2026-10-18 — Runaway risk; decision: per-run dollar limit, P3 thinking unchanged

- Distribution of P3-adapted answering-call thinking (pilot, sorted): 46 / 50 ≤ 7,493 tokens
  (most 500–5,000, mean ≈ 2,500 without outliers); then 16,631; then 3 × ~62,912 (ceiling).
- Google's Gemini 3 docs (ai.google.dev/gemini-api/docs/thinking, checked 2026-10-18) document
  only `thinking_level` (minimal / low / medium / high, default high), no token budget, and
  confirm that `max_output_tokens` (which includes thinking) truncates the output. So a token
  bound on thinking is not available; an output cap would bring back truncated checklists.
- Runaway rate: 3 / 50 = 6%, 95% CI 1.3–16.5% → on 400 test studies ≈ 24 (5–66). Not tied to
  case type in the pilot (2 of 3 runaways were normal studies; their retrieved cases looked like
  those of ordinary studies); likely partly chance (temperature 0.25). P3 test cost ≈ $5 / $8.4 /
  $16 (low / expected / high); in the high case the project total reaches ≈ $44.
- **Decisions:** (1) P3-adapted keeps the default thinking level (no medium-vs-high
  comparison, no further change to P3). (2) **A per-run dollar limit of USD 10**: every metered
  run stops cleanly once its own logged cost reaches the limit (cached results kept; re-running
  continues), default 10, overridable with `--max-usd`. It can overshoot by at most one call,
  since a call's cost is known only afterwards. (A project-wide guard was offered, not chosen.)
- **Implemented** (`common.py`): prices moved here from `spend.py` (one table: `GEN_PRICES`,
  `EMBED_PER_IMAGE`, `EMBED_TEXT_PER_M`) with one cost formula `call_cost()`, used by both the
  ledger and the limit (ledger total unchanged after the move: $3.8887). Every logged call adds
  its cost to the run's total; `count_request` refuses the next call once the total reaches the
  limit (`SpendLimitReached`, a clean stop) and refuses a call to a model without a price
  before it is made. `DEFAULT_MAX_USD = 10`; all three runners take `--max-usd` and print the
  run's cost at the end.
- Tests: `test_retry.py` now 22 checks (cost formula for generation / embeddings / failed
  calls; limit stops before the next call, overshoot at most one call; unpriced model refused
  without being counted or logged; default 10). The fake model names "m" / "e" in the older
  tests were replaced by the real names (now required to have a price). All three test suites
  pass; all three runners re-checked on their cached pilots (0 calls, cost $0.0000).
- `SETUP.md`: tests list updated.

## 2026-10-18 — Checklist hand check: protocol (fixed before reading)

- Purpose (plan §5, adapted to the checklist scoring): does each answer's checklist agree with
  what the rest of the same answer says?
- **Sample:** 15 pilot answers per starter (P1-adapted-ns, P2-adapted, P3-adapted; checklist
  question), drawn with seed 2026 by `scripts/checklist_review.py`: up to 8 answers with at least
  one "present" in the checklist, the rest from answers with none (all-absent answers are the
  majority, but positive calls are where disagreement matters most).
- **Judgement per finding** (6 per answer, 90 per system), reading the whole answer:
  `agree` (the text supports the checklist entry, including "not mentioned" for an absent entry),
  `disagree` (the text asserts the opposite: e.g. checklist absent but the text describes the
  finding as present, or checklist present but the text rules it out), `unclear` (the text is
  hedged or ambiguous; e.g. checklist uncertain).
- **Who:** first pass by Claude (the assistant), recorded with a short reason for every
  `disagree` / `unclear`; the review sheets are saved so any of them can be checked.
  Reported as such in results.
- Output: `data/runs/review/checklist_review_<arm>.md` (the answers) and
  `data/runs/review/checklist_review.csv` (judgements).

## 2026-10-18 — Checklist hand check done

- `scripts/checklist_review.py` drew the sample (seed 2026): 15 answers per starter; P1-ns 8 with a
  "present" (13 available), P2 5 (all 5 available), P3 8 (34 available). Sheets:
  `data/runs/review/checklist_review_<arm>.md`; sample `checklist_review_sample.csv`.
- Read all 45 answers in full (first pass by Claude). Judgements in
  `data/runs/review/checklist_review.csv`:
  **P1-ns 89 agree / 1 unclear; P2 90 agree; P3 89 agree / 1 unclear; 0 disagree (268 / 270
  agree).** Unclear: P1-ns CXR1012 Atelectasis (checklist present; text gives it only as a
  differential); P3 CXR3345 Atelectasis (checklist absent; text mentions "possible
  subsegmental atelectasis" in one report, overall negative).
- **Conclusion: the checklists represent what the answers say; scoring on them is sound.**
- Observations (not inconsistencies), for Q1 and G's design:
  - **Lung Opacity counted alongside Lung Lesion:** when the only abnormality is a nodule or
    mass, P1-ns and P2 mark both present (P1-ns CXR1592, CXR2039, CXR3419; P2 CXR2039). The
    ground truth codes nodules / masses as Lung Lesion only (results §2.4b), so this will appear
    as Lung Opacity false positives — a definition mismatch. G's prompt can define the findings.
  - **P3 aggregates over its neighbours:** a finding is "present" if any of the six retrieved
    cases has it ("in some cases", "across different cases"; CXR494, CXR3009, CXR3419). Its
    checklist is faithful to its text; the text is about the retrieved cases, not the patient.
  - "uncertain" is rare (P1-ns CXR3419 Atelectasis; P3 CXR3525 Cardiomegaly), used for hedged
    statements as intended.

## 2026-10-18 — Q1 write-up (results §3)

- `results.md` §3 brought up to date and completed:
  - §3.2 P2-adapted: train index and pilot results (original and checklist question) added.
  - §3.3 P1: the CheXbert scoring-risk note marked as superseded; checklist pilot added.
  - §3.4 renamed "P3-orig and P3-adapted": runner name, the cap change and the P3-adapted pilot
    (truncation table, runaway thinking, cost) added.
  - **New §3.5** "Scoring the starters": the CheXbert failure (table of real sentences), why
    rewrite rules and a second reader were not used, the checklist question, compliance,
    the hand check, the two scoring patterns (Lung Opacity + Lung Lesion; P3 aggregating).
  - **New §3.6** "Q1 summary": what reaches each model, the changes each needed, pilot
    behaviour, failure modes, cost, with the caveat that the pilot (10 positive findings)
    shows behaviour, not accuracy.
- Figures checked against the ledger and summary files (one cost corrected before saving:
  P3-orig checklist ≈ $0.0045 per study).

## 2026-10-18 — Q2 started: plan and retrieval comparison design (fixed before running)

- Decision: Q2 (which starter components to carry into G, which gaps G must fill) is
  answered before G is designed, because it drives G's design decisions.
- **Evidence for Q2:** (1) `../codebase_analysis.md` (the three starters' code); (2) the pilots
  and their failure modes (results §3); (3) a free retrieval experiment on train (below).
- **Retrieval comparison** (`scripts/q2_retrieval_compare.py`; no API calls; train split only,
  so validation and test stay untouched): R0's leave-one-out kNN vote (each train study scored
  from the other 2,803; unweighted share of the k neighbours with each finding; macro AUROC over
  the six findings), for k ∈ {1, 5, 10, 50, 150}, with five embeddings already available or
  computable locally:
  (a) BiomedCLIP, centre-crop (R0's index, `biomedclip_frontal.npy`);
  (b) BiomedCLIP, padded to square (black) before its preprocessing — new local embedding,
      saved as `data/processed/biomedclip_frontal_pad.npy`, the existing file untouched;
  (c) Gemini Embedding 2, 3,072-d, image (P2's index);
  (d) Gemini Embedding 2, 768-d, image (P3's index, image part);
  (e) P3's blended index (image + report, 0.68 / 0.32) queried by the 768-d image (as P3 does).
  Reported: macro and per-finding AUROC per k; paired bootstrap CI (2,000 resamples of studies)
  of each embedding's macro AUROC minus (a)'s, at k = 5 (the scale of retrieval a prompt can
  hold) and k = 150 (R0's k). Nothing is selected by this; it informs Q2 and G's design.

## 2026-10-18 — Q2 retrieval comparison: results

- Ran `scripts/q2_retrieval_compare.py` (no API calls; train only). Computed the padded
  BiomedCLIP embeddings locally (3,690 frontal images, TF32 off) →
  `data/processed/biomedclip_frontal_pad.npy`; the existing `biomedclip_frontal.npy` (dated
  2026-10-16) is untouched. Report: `data/runs/q2/retrieval_compare.md`.
- Macro AUROC (k = 1 / 5 / 10 / 50 / 150): BiomedCLIP crop 0.572 / 0.651 / 0.671 / 0.712 /
  0.741; BiomedCLIP pad 0.584 / 0.655 / 0.680 / 0.732 / 0.731; Gemini Embedding 2 3,072-d 0.529 /
  0.574 / 0.600 / 0.667 / 0.720; 768-d image 0.531 / 0.576 / 0.601 / 0.664 / 0.712; P3 blend 0.550 /
  0.595 / 0.635 / 0.703 / 0.737.
- vs. BiomedCLIP crop, k = 5: pad +0.004 [−0.012, +0.020]; Gemini 3,072 −0.077 [−0.100, −0.054];
  Gemini 768 −0.075 [−0.098, −0.052]; P3 blend −0.056 [−0.079, −0.034]. k = 150: pad −0.010
  [−0.031, +0.013]; Gemini 3,072 −0.021 [−0.049, +0.008]; 768 −0.029 [−0.058, −0.001]; blend −0.004
  [−0.033, +0.024].
- **Conclusions:** (1) at prompt-sized k, BiomedCLIP retrieves clearly more informative
  neighbours than the starters' general-purpose embedder; (2) crop vs. pad makes no measurable
  difference for retrieval (unlike S0) → no reason to change R0's / G's index; (3) report text in
  the index helps the Gemini embedder but does not close the gap at small k.
- `results.md` §4 (Q2) written: §4.1 component review (keep / adapt / upgrade / drop / gap, each
  with its evidence), §4.2 retrieval comparison, §4.3 gaps none of the starters fills, §4.4
  design of G pending (what the evidence already settles is listed there).

## 2026-10-18 — G design proposal written (for decision)

- Facts checked first (free): the INDICATION field is present in 98–99% of studies (train 2,736 /
  2,804; validation 394 / 400; test 396 / 400), median ≈ 30 characters, mostly symptoms or purpose;
  it names one of the six findings in 55 train, 3 validation and 6 test studies (correct in 14, 2
  and 5). Train reports (FINDINGS + IMPRESSION): median 242 characters (p90 442), so 10 neighbours
  ≈ 600 tokens.
- Google's Gemini 3 guide (ai.google.dev/gemini-api/docs/gemini-3, checked 2026-10-18): "we
  strongly recommend keeping the temperature parameter at its default value of 1.0 … setting it
  below 1.0 may lead to unexpected behavior, such as looping"; `thinking_level` defaults to high
  on Gemini 3 Flash; legacy `thinking_budget` cannot be combined with it.
- **Hypothesis (not tested):** P3's runaway thinking may be related to its shipped temperature
  of 0.25 (Google warns of looping below 1.0); P1 and P2 run at the default and showed none.
- `g_design.md` written: settled choices (A1–A8) and decisions needed (D1 k, D2 finding
  definitions, D3 output format and score, D4 thinking level, D5 prompt outline, D6 prompt
  development and freezing, D7 optional arms G-img / G+S0), with recommendations, evidence and
  a cost estimate (G family ≈ $19; project ≈ $37.5).

## 2026-10-18 — G design decided

- **Decision: all recommendations adopted** in `g_design.md` (status changed to decided):
  D1 k = 10, fixed now (deviation: plan said choice of k on validation; tuning would cost ≈ $7 per
  k value); D2 the six findings defined in G's prompt with the ground truth's definitions; D3
  JSON enforced by the API's response schema (status, probability, cited case IDs, rationale per
  finding; impression), status → binary as for the starters, probability → AUROC and the pooled
  Youden threshold on validation; D4 default thinking level (high); D5 the prompt outline; D6 at
  most 3 prompt versions on the 50 pilot studies, then frozen before validation; D7 **G-img added
  (test only)**, **G+S0 dropped**.
- `results.md`: §4.4 filled in; deviations table +3 rows (k fixed a priori; G-img added; G+S0
  dropped); threat added (G gets finding definitions the starters do not).

## 2026-10-18 — G runner written and checked offline

- `scripts/g_system.py` (arms g, g_ind, g_img; design as decided): R0's BiomedCLIP index;
  top-10 train neighbours; one direct call with the patient's PNG (sent as is), the prompt
  (definitions, steps, indication, the 10 neighbours' FINDINGS + IMPRESSION with ID and
  similarity); JSON enforced with a pydantic response schema (`Output` / `FindingOut`); default
  temperature and thinking; responses cached with the exact prompt in
  `data/runs/<arm>__prompt<N>/<split>/`; each response validated (exactly the six findings,
  probability in 0–1, cited IDs among the given neighbours) and problems recorded; test split
  refused without `--test`.
- **Precision on "G−ind = G minus the indication":** G−ind also omits the one step that refers
  to the indication ("Take the clinical indication into account."), since that step would
  otherwise point at missing text. Nothing else differs (tested). A missing indication in G is
  shown as "Clinical indication: not provided".
- `tests/test_g_system.py`, 10 checks, all pass, including: **G's 10 neighbours equal R0's first
  10 for all 400 validation studies**; leakage guard refuses a train study as query; the
  validator catches a missing finding, an out-of-range probability and an uncited case ID. (One
  test first failed on step renumbering in G−ind; the test now compares steps without their
  numbers.)
- A real prompt (CXR8) is ≈ 4,300 characters.

## 2026-10-18 — G 2-study check; prompt-1 pilot started; exporter

- **2-study check** (`g val --pilot --limit 2 --max-requests 4`): 2 calls, both valid JSON, all
  six findings, no problems; both studies correctly called normal (ground truth: none of the
  six). ≈ $0.0088 per study (input 2,122–2,254, output ≈ 530, thinking 1,472–2,595 tokens). For
  "absent" calls G cites neighbours that also lack the finding (support for its call, as
  instructed). Probabilities for clear negatives: 0.05–0.10.
- Started the prompt-1 pilot for the other 48 studies (cap 55 calls, ≈ $0.43).
- `scripts/g_export.py` (rules in its docstring, fixed before any pilot result): G answers →
  `<split>_checklist.csv` with the starters' columns plus `prob_<finding>`; status used if the
  finding occurs exactly once, else "missing"; invalid JSON → all missing; missing probability →
  0.0; all counted. `pilot_summary.py` now includes `g__prompt1`.

## 2026-10-18 — G prompt-1 pilot results

- 50 / 50 studies (this run 48 calls, $0.48; no retries). **50 / 50 valid JSON, all six findings
  in every answer, no missing statuses or probabilities.** Thinking median 2,015, p90 4,140, max
  5,525 tokens (no runaways); cost ≈ $0.010 per study (max $0.019). Spend to date $4.39.
- Behaviour (50 studies, 10 positive findings): G called 8 findings, 4 correct, 4 false alarms;
  40 / 42 normal studies called normal; uncertain used once. (P2: 11 / 3 / 8 / 40; P1-ns 31 / 5;
  P3 75 / 6; S0 93 / 6; R0 53 / 5.) Table: `data/runs/pilot_summary_checklist.md`.
- **Problem 1 — citation format (3 / 50 answers):** CXR205 and CXR3179 cited neighbours by list
  position ("1", "3", …) instead of case ID; CXR2573 wrote "CXR1", "CXR5", "CXR9", i.e. positions
  with a prefix. Cause: the prompt numbers the cases ("[1] CXR316 …"). Statuses and
  probabilities are unaffected; these citations cannot be checked (plan §5 citation agreement).
- **Problem 2 — coarse probabilities:** of 300 values, 198 are 0.05 and 44 are 0.10; 13 distinct
  values in total. Most negatives tie, which limits ranking (AUROC, the anchored comparison).

## 2026-10-18 — Decision: G prompt version 2 (both changes)

- Decision: both proposed changes (D6: second of at most three versions):
  1. Neighbours listed as "Case CXR316 (similarity 0.965): …" without position numbers, plus
     "Cite cases by their ID exactly as written, e.g. CXR316."
  2. Probability wording: "Give the probability with two decimals as your honest estimate; for
     subtle, borderline or partly obscured appearances, use intermediate values rather than a
     fixed low value."
- Known limits stated beforehand: (2) may create artificial spread; the pilot (10 positive
  findings) cannot show improved ranking, only fewer ties. Checks after the re-run: all
  citations valid; number of distinct probability values and ties; G's statuses about the same
  as with prompt 1 (only the probability wording changed; a large shift would be a warning).
- Prompt-1 answers stay in `data/runs/g__prompt1/` as evidence.
- Code: `PROMPTS[2]` = prompt 1 with only `step_output` (two sentences added) and `case_line`
  ("Case CXR… (similarity …): …") changed; the case line is now part of each prompt version.
  Tests (14, all pass): v2 lists cases by ID without numbers; v2 has both new sentences; v2
  differs from v1 only in the case lines and the output step; **prompt 1 rebuilt byte-identical
  for all 50 cached pilot answers** (the refactor did not change version 1).
- Started the prompt-2 pilot (`g val --pilot --prompt 2 --max-requests 55`).

## 2026-10-18 — G prompt-2 pilot: checks

- 50 / 50 studies, 50 calls, $0.53 (≈ $0.0105 per study), no retries; thinking median 2,165, max
  6,943 (no runaways). Spend to date **$4.91** (10.9% of the cap).
- **Check 1, citations:** 0 answers with problems (prompt 1: 3). Fixed.
- **Check 2, probability resolution:** distinct values 13 → 19; the most common value (0.05) 66% →
  57% of the 300 probabilities. Better, modestly; still many ties.
- **Check 3, statuses:** binary agreement with prompt 1 **296 / 300**; 7 status changes (CXR272
  LO absent→uncertain, AT absent→present; CXR1592 FR absent→present; CXR2019 LL absent→uncertain;
  CXR2568 CM present→absent; CXR3539 LO uncertain→present, AT absent→uncertain). Calls 8 → 10,
  correct 4 → 4, normal studies called normal 40 → 39 of 42.
- **Caveat:** G runs at temperature 1.0, so repeated runs differ by chance; changes of this size
  cannot be attributed to the wording. Single-run differences of a few calls between systems are
  within this noise (hence the test set and bootstrap CIs).

## 2026-10-18 — Prompt 2 frozen; next steps

- **Decision: prompt 2 is frozen** (D6; version 3 not used). `g_system.py`:
  `FROZEN_PROMPT = 2`, default `--prompt 2`, and any non-pilot run with another version is
  refused (checked: `g val --prompt 1` refused; `g val --pilot` found all 50 cached, 0 calls;
  tests pass). `g_design.md` §D and `results.md` §4.4 record the prompt development; §4.4 also
  states precisely that G−ind omits the indication line and the one step referring to it.
- Spend to date **$4.91** (10.9% of the USD 45 cap).

The next-steps list from this entry is at the end of the diary.

## 2026-10-18 — Wording: no "user" / "owner" attributions

- Decision: the records say "Decision: …", not who decided. Replaced in this diary (11 places:
  "Raised with the user", "Discussed with the user", "user's request", "user agreed",
  "discussed with the user", "Decisions (user)", "so the owner can check", "User's decision",
  "User agreed to all recommendations", "User chose", "Decision (user)"), in `g_design.md`
  (status line) and in two code comments (`common.py`, `tests/test_retry.py`). `results.md` had
  none. A stale bytecode file of the renamed `p3_orig.py` was deleted (`__pycache__`).
- Left as they are: "user message" (ADK's term for the message sent to the agent) and the
  2026-10-17 entries that describe the earlier removal of "owner" wording (they quote the old
  word to say what was replaced).
- Decision: first-person phrasing made neutral, the hand check stays attributed to Claude.
  - Neutral now (diary): "So I wrote my own labeller" → "So a separate labeller was written";
    "never printed or opened by me" → "the key itself was never printed or opened"; "I first
    misread" → "The depth was first misread"; "our downloaded counts / our six labels / our
    appended question / our addition / our question / not set by us" → neutral wording.
    (results): "our adaptation 4" → "adaptation 4"; "chosen by us" → "chosen in this project";
    "Plan §6 told us" → "Plan §6 said".
  - Hand check: `results.md` §3.5 "first pass by the assistant" → "first pass by Claude, an AI
    assistant" (matches the diary).
  - Unchanged: quotations (the dataset paper's "we …", Google's "we strongly recommend …").

## 2026-10-18 — Scripts folder reorganisation: analysis and plan (before moving)

- Request: separate G (the system), the required accessory scripts and the outdated scripts;
  nothing deleted.
- **What moving would affect (checked in the code):**
  1. **Project paths:** 12 scripts compute the project folder as `Path(__file__).parent.parent`.
     One folder deeper this points at `scripts/`, so outputs would silently go to the wrong place
     (e.g. `r0_knn.py` would create `scripts/data/runs/`; `s0_weights.py` would miss its weights in
     `.cache/` and re-download them). Fix: the project path is computed only in `common.py`
     (which stays in `scripts/`); every other script imports `PROJECT` from it.
  2. **Imports:** scripts add their own folder to `sys.path` to import `common` / `metrics`; from a
     subfolder they must add `scripts/` instead (`parents[1]`). Cross-folder imports need explicit
     paths: `label_outputs` → `chexbert_labels`, tests → `g_system`, `parse_checklist`;
     `q2_retrieval_compare`'s unused `starters` path is removed. Same-folder imports
     (`s0_score` → `s0_weights`) keep working.
  3. **Commands and documentation:** `SETUP.md`, `results.md`, `g_design.md` and the scripts'
     "Usage:" lines are updated; old diary entries keep the paths they had (history) and this
     entry gives the mapping.
  4. Not affected: cached results and logs (no script paths stored), the P3 runner's import of
     the starter code (uses `common.PROJECT`), the API call log.
- **Layout:** `scripts/` top level = shared libraries only (`common.py`, `metrics.py`);
  `g/` (G: `g_system.py`, `g_export.py`); `starters/` (unchanged); `anchors/` (`s0_weights`,
  `s0_score`, `s0_select`, `r0_knn`); `data_prep/` (pipeline and its checks: `classify_views`,
  `image_quality`, `select_frontal`, `embed_and_dedup`, `pixel_duplicates`, `chexbert_labels`,
  `mesh_ground_truth`, `make_splits`, `check_patient_ids`, `check_im_series`,
  `check_image_mapping`, `label_stats`, `view_groups_sheet`); `evaluation/` (`parse_checklist`,
  `pilot_summary`, `checklist_review`, `q2_retrieval_compare`); `tools/` (`spend.py`);
  `outdated/` (`label_outputs.py`: CheXbert labelling of answers, replaced by the checklist;
  `check_api_key.py`: week-1 check of the then-current and retired models, and it bypasses the
  metered client); `tests/` (unchanged).
- **Verification plan:** SHA-256 of every output that can be regenerated without the API, before
  the move; after the move, regenerate with the moved scripts and require byte-identical files;
  all test suites; the runners find their cached pilots with 0 calls; no stray `data/` folder.
  Moves with `git mv` (history kept); stale `__pycache__` folders removed.

## 2026-10-18 — Scripts folder reorganised

- Moved with `git mv` (history kept), as planned: `g/` (g_system, g_export); `anchors/` (s0_weights,
  s0_score, s0_select, r0_knn); `data_prep/` (13 pipeline and check scripts); `evaluation/`
  (parse_checklist, pilot_summary, checklist_review, q2_retrieval_compare); `tools/` (spend);
  `outdated/` (label_outputs, check_api_key). Unchanged: `common.py`, `metrics.py` (top level),
  `starters/`, `tests/`. Stale `__pycache__` folders removed. New `scripts/README.md` describes
  the layout.
- Code: in 10 moved scripts `PROJECT = Path(__file__).resolve().parent.parent` replaced by
  `from common import PROJECT` (also added in `s0_weights.py`); `sys.path` now adds `scripts/`
  (`parents[1]`); explicit paths for `label_outputs` → `data_prep/chexbert_labels`, `s0_score` →
  `anchors/s0_weights`, tests → `g/`, `evaluation/`; the unused `starters` path in
  `q2_retrieval_compare` removed. 62 path references updated in the scripts' usage lines,
  `results.md`, `g_design.md`, `SETUP.md` and `CONTENTS.md` (old diary entries keep their paths).
- **Verification:** all four test suites pass; all 31 scripts load and resolve `PROJECT` to
  `semester_project/`; S0's weights are found in place (no download). **12 outputs regenerated
  with the moved scripts are byte-identical to their SHA-256 before the move** (R0 train / val /
  selection, S0 selection and report, the four checklist tables incl. G prompt 1 and 2, the pilot
  summary, the Q2 retrieval report). Ledger total unchanged ($4.9120). No stray `data/`, `runs/`
  or `.cache` folder inside `scripts/`. All four API runners find their cached pilots (0 calls).
- Old → new path mapping for earlier diary entries: `scripts/<name>.py` → `scripts/g/…` (g_system,
  g_export), `scripts/anchors/…` (s0_*, r0_knn), `scripts/data_prep/…` (pipeline and checks),
  `scripts/evaluation/…` (parse_checklist, pilot_summary, checklist_review, q2_retrieval_compare),
  `scripts/tools/spend.py`, `scripts/outdated/…` (label_outputs, check_api_key).

## 2026-10-18 — G and G−ind validation runs started; decision: pooled-threshold script

- Started `g_system.py g val` (350 new calls; the 50 prompt-2 pilot answers are reused) and
  `g_system.py g_ind val` (400 calls), frozen prompt 2, default USD 10 per-run limit each.
  Progress logs: `data/runs/logs/g_val_run.txt`, `g_ind_val_run.txt`.
- New script `scripts/g/g_select.py` for to-do step 1 (results §2.5: one threshold per arm,
  pooled across the six findings, tuned on validation; results §5: Youden's J, ties → higher).
  Written before the full validation answers were seen.
- Decision: details the earlier records left open:
  1. "Pooled" = one Youden threshold over all (study, finding) pairs of the stated
     probabilities (400 × 6 = 2,400 pairs); positive = probability ≥ threshold.
  2. Also reported, not used for tuning: per-finding sensitivity / specificity / F1 of the
     stated status ("present"), next to the same at the threshold; the number of pairs where
     the two binary readings disagree; thinking tokens per answer (median, mean, max, > 20,000).
  3. Per-finding AUROC with bootstrap 95% CI (2,000 study resamples, seed 2026), as R0.
  Reason: (1) is the direct reading of the 2026-10-16 decision; (2) and (3) cover the to-do
  item's checks (schema compliance, thinking) and show whether the threshold reading and the
  stated status tell the same story before both are used on test.
- Output: `data/runs/<arm>/selection.json` (threshold + validation results); report to stdout.

## 2026-10-18 — G and G−ind validation results (to-do step 1 done)

- Runs complete: G 350 new calls ($3.44), G−ind 400 calls ($4.12); all OK, no retries logged.
  Exported (`g_export.py val g__prompt2`, `… g_ind__prompt2`): 0 invalid JSON, 0 validation
  problems, 0 missing statuses or probabilities, in both arms.
- `g_select.py` → `selection.json` + `selection_report.md` in each arm's folder. Pooled threshold
  **0.08 for both arms**. Macro AUROC G 0.770 [0.701, 0.837], G−ind 0.724 [0.678, 0.782]
  (S0 val 0.757, R0 val 0.700). Thinking: max 7,903 (G), 8,263 (G−ind); no runaways.
- Stated status is conservative (macro sensitivity 0.30 / 0.27 at specificity 0.97); at the
  threshold 0.63 / 0.82 (G). Both readings go to test, as planned.
- Checked by hand before writing results: S0 and R0 validation Lung Lesion AUROC (0.482, 0.435)
  from their reports; corrected a draft sentence (G above G−ind on 5 of 6 findings, not 4).
- `results.md`: new §5.3 (G and G−ind on validation); test-set results renumbered §5.4.
- Spend now $12.47 (27.7% of the cap). Re-projection: + G-img pilot ≈ $0.3, test: starters
  ≈ $14.6, G + G−ind ≈ $8.0 (measured $0.010 per study), G-img ≈ $2.5 → total ≈ $37.9 of $45.

## 2026-10-18 — G-img pilot (to-do step 2 done)

- `g_system.py g_img val --pilot --max-requests 55` (frozen prompt 2): 50 calls, $0.32, all OK.
  Exported (`g_export.py val g_img__prompt2`): 0 invalid JSON, 0 problems, 0 missing; no case
  IDs cited. Thinking median 1,180, max 7,179.
- Compared with G on the same 50 studies (descriptive; 10 positive pairs): G-img 16 "present"
  (3 correct) vs. G 10 (4 correct); G-img's extra calls are mostly Atelectasis (7 vs. 3);
  stated statuses agree on 286 / 300 pairs. Fit for the test run. results §5.3.
- Spend now $12.79 (28.4% of the cap); projection unchanged (≈ $37.9).

## 2026-10-18 — Evaluation design proposal written (for decision)

- `eval_design.md` (to-do step 3): settled items A1–A7 from earlier decisions; open choices
  E1 (pre-declared comparisons, primary = G − S0 macro AUROC), E2 (precision@k per finding with
  prevalence baseline; own k and k = 1), E3 (copying rate: kNN decision at k = 10 with G-img as
  control), E4 (citation agreement with an all-neighbours baseline), E5 (one script + offline
  tests). Facts checked for it: P2 retrieves 1 case, P3 6 (all 50 pilot answers), G 10, R0 150;
  R0's thresholds are vote shares 0.013–0.100.
- No code yet; waiting for the decision.

## 2026-10-18 — Decision: evaluation method for the test set (E1–E5 all adopted)

Decision: all five recommendations of `eval_design.md` adopted, before any test-set result
exists. Written up in full in results §5.4 (new); the test-set results section is now §5.5.
Plan §5 lists *what* is measured; these five points fix *how*, where the plan is silent or its
wording does not work on these data. No API calls involved.

**E1 — Which differences get confidence intervals.**
- Question: plan §5 wants paired bootstrap CIs "for the differences" but names none. Eight
  systems × two binary readings for G / G−ind × several metrics = dozens of possible pairs.
- Options: (a) fixed list with one primary comparison; (b) all pairs; (c) per-system CIs only.
- Chosen: (a). Primary: **G − S0, macro AUROC**. Secondary: G − R0 (AUROC); G − G−ind and
  G − G-img (AUROC, macro F1 stated); G-img − S0 (AUROC); G − each starter (macro F1,
  sensitivity, specificity of stated verdicts).
- Reasons: G − S0 is the thesis question itself (does image + retrieval + text beat the single
  image model); AUROC compares ranking without depending on either threshold. Naming it before
  the test run means the answer cannot be picked afterwards from whichever pair looked best.
  The secondary comparisons take G apart step by step (LLM over retrieval; the indication;
  retrieval + indication within one design; general LLM vs. specialist on the image alone) and
  answer Q3 for the starters. Starters give only verdicts, so their comparison uses G's stated
  verdicts, scored by the same rule. Option (b): at 95%, some of dozens of intervals would
  exclude zero by chance. Option (c): overlapping per-system CIs do not answer a paired
  question. No multiple-comparison correction; only #1 is called the main result.

**E2 — Retrieval quality (precision@k).**
- Question: plan's "share of retrieved training cases that share the query's labels" is
  dominated by normal–normal matches with six rare findings, so it would hardly show whether a
  retriever finds cases with the same disease.
- Options: (a) per finding, on queries that have it, with train prevalence as baseline (lift);
  (b) exact six-label match; (c) Jaccard of label sets.
- Chosen: (a), plus a separate row for normal queries (share of normal neighbours). Depth: each
  system's own k (P2 1, P3 6, G 10, R0 150) and all at k = 1 (the only depth every system has).
  R0's first 10 neighbours = G's 10 (same index), so reported once.
- Reasons: the LLM needs cases with the same finding as evidence; (a) measures exactly that.
  Prevalence makes it readable (0.15 is excellent for Fracture, ≈ 1.5% of train, poor for a
  common finding). (b) mostly counts normals retrieving normals; (c) is undefined for two empty
  label sets. Deviation from the plan's wording: recorded in results "Deviations".

**E3 — Copying rate.**
- Question: does G read the image, or repeat its 10 cases? Plan: agreement with "the
  neighbours' majority vote", especially when the vote is wrong. Problems: a strict majority
  (≥ 6 / 10) is almost never positive here (R0's thresholds are vote shares 0.013–0.100), so
  agreement would just be G's specificity; and agreement alone cannot separate copying from
  both reading the same image.
- Options: (a) neighbours' decision = R0's rule at k = 10 (vote share ≥ per-finding Youden
  threshold, leave-one-out on train), with G-img as control; (b) strict majority; (c) any
  positive neighbour.
- Chosen: (a). Reported for G and for G-img: agreement with the neighbours' decision, overall
  and where that decision is wrong; G minus G-img = the neighbours' pull. Also: "present" rate
  by number of positive neighbours (0, 1, 2, 3+), G and G-img side by side.
- Reasons: (b) is nearly constant ("absent"); (c) fires for most studies on common findings
  (Lung Opacity ≈ 14% of train, so most sets of 10 contain one). (a) reuses R0's existing
  method, tuned on train only. G-img sees the same image and prompt but no cases, so its
  agreement is what shared image reading alone gives; the difference is influence. Vote at
  G's k = 10, not R0's 150 (results "Observations"). Deviation from the plan's wording: recorded.

**E4 — Citation label agreement.**
- Question: when G says "present" and cites cases, do those cases have the finding? (Are G's
  explanations verifiable?)
- Options: (a) the plan's share plus a baseline; (b) the share alone.
- Chosen: (a). Per finding, pooled over G's (and G−ind's) "present" verdicts: share of cited
  cases with the finding vs. share of all 10 neighbours with it. Counted: "present" without
  citation; citations under "absent" / "uncertain" (counted, not scored).
- Reasons: without the baseline, 40% could mean selective citation (neighbours 10%) or none
  (neighbours 40%). A "present" claim without evidence is itself an explanation failure.
  Citations under "absent" cannot be checked meaningfully (a normal case "shows no effusion").

**E5 — Implementation.**
- Chosen: one script `scripts/evaluation/evaluate.py <split> [--test]` (no API), output
  `data/runs/eval/<split>/report.md` + `results.json`; offline tests
  `scripts/tests/test_evaluate.py` with hand-computed answers for precision@k, copying,
  citation agreement and study alignment.
- Reasons: one command reproduces Q3; aligning studies across eight systems (likeliest silent
  error) is written once; new metrics tested before real data, as for `metrics.py`. Run on
  validation first (S0, R0, G, G−ind: 400; starters, G-img: 50 pilot; compared on common
  studies) to check end to end, not as results. On test: assert all 400 for every system;
  refuse test without `--test`.

**Records:** results §5.4 (full write-up), "Resolved decisions" #2, three rows in "Deviations"
(E1 as a specification, E2, E3); `eval_design.md` marked decided.

## 2026-10-18 — Evaluation script written and checked on validation (to-do step 3 done)

- `scripts/evaluation/evaluate.py <split> [--test]`: implements E1–E5 as decided; no API.
  Output `data/runs/eval/<split>/report.md` + `results.json`. Listed in `scripts/README.md`.
- `scripts/tests/test_evaluate.py`: 18 tests on synthetic data with hand-computed answers
  (precision@k incl. lift and the normal row; copying overall / when wrong / pooled; present
  rate by positive neighbours; citation share, baseline, no-citation and absent-citation
  counts; study alignment; per-finding metrics; exactly one primary comparison). All pass; the
  other four test files still pass.
- Two fixes during the first validation run:
  1. Crash: a finding with no positives in the 50 pilot studies gives an all-nan bootstrap, and
     `metrics.ci` fails on an empty array. Fixed with a local wrapper in `evaluate.py` returning
     an undefined CI ("–"); the shared `metrics.py` left unchanged.
  2. **Rounding bug found by cross-checking:** R0's sensitivity / specificity did not match its
     own report (val Lung Lesion 0.27 / 0.70 vs. 0.55 / 0.44). Cause: `r0/<split>.csv` stores vote
     shares with 4 decimals (2/150 → 0.0133) while the thresholds are exact (0.01333…), so
     studies exactly at a threshold flipped to negative. Fix: R0 scores recomputed exactly from
     the stored 150 neighbour IDs, with an assertion that they match the table to within
     rounding. AUROC was unaffected (ranks unchanged). R0's own report used exact values.
- Cross-checks after the fixes, all identical: S0 and R0 per-finding rows vs. their
  `selection_report.md`; G and G−ind vs. `g_select.py`; starters vs. `pilot_summary_checklist.md`
  (e.g. P1 Lung Opacity 3 / 5 found, 12 calls → specificity 36 / 45 = 0.80). G's neighbours =
  R0's first 10 for 400 / 400 studies.
- Validation numbers (not results; starters and G-img only 50 studies), for orientation:
  G − S0 macro AUROC +0.013 [−0.045, +0.079]; G − R0 +0.070 [+0.012, +0.115]. Copying:
  neighbours' kNN@10 decision wrong on 383 of 2,400 pairs; G (stated) agrees with it there
  0.277, G-img 0.286 (50 studies only). Citations: G's cited cases have the finding far more
  often than its neighbours overall (e.g. Atelectasis 0.80 vs. 0.23). G also cites cases under
  "absent" / "uncertain" verdicts 2,129 times (counted, not scored, E4).

## 2026-10-18 — Decision: no P1 web-search arm

- The open decision (results "Open decisions" #1, to-do step 6): run P1 with its DuckDuckGo tool
  on, i.e. as shipped, on the 50 pilot studies (~$0.5–1.5)? Never run so far: P1-adapted's 62%
  malformed calls were with the tool *off* while the prompt still ordered it; the arm would be
  prompt *and* tool together.
- Source corrected: the arm comes from `thesis_plan.md` (systems table: "a 50-case sensitivity
  run with it on"), not the semester plan; earlier entries (2026-10-16/17) and results cited it
  as "plan §3". The semester plan §3 only says search is off (reproducibility); that citation
  stays correct.
- **Decision: not run.** Reasons:
  1. Reproducibility: live web results change; the arm could not be repeated (the reason already
     recorded for search off).
  2. Search does not make the diagnosis: section 5 asks only for literature and references; the
     findings come from the image. The arm would measure reference-gathering, not what the
     thesis compares.
  3. Possible route to the data: IU X-Ray is publicly searchable (Open-i); a findings search could
     return IU reports, possibly the study's own. Cannot be ruled out; an uncontrolled input no
     other system gets (on top of plan §8's memorisation risk).
  4. Incomparable and costlier: the only system with web access, with search results in its
     context; its results could not be read cleanly against the other arms.
  Considered against: it is the only measurement of P1 exactly as shipped; without it Q1 says
  only that P1 fails when its tool is removed. It would not have entered the test comparison.
- Records: results "Resolved decisions" #3 (with reasons and the argument for running it);
  "Open decisions" now empty; new "Deviations" row (thesis plan's arm not run); §3.3's two
  forward references updated. `p1_adapted.py --web-search` stays unused.

## 2026-10-18 — Supervisor one-pager; step 7 marked completed for now

- `presentation/supervisor_questions.md`: the plan's three questions (§9: scope, API / budget,
  modality) plus a fourth (keep the DenseNet S0 although the ResNet-50 is stronger, results
  "Resolved decisions" #1); the 13 method deviations with plan / done / reason; the five
  informational ones. Notes that items 1, 2, 7 and 13 set how the test run is scored.
- Decision: the supervisor's answers do not block the study. To-do step 7 is marked completed
  for the time being; answers are folded in when they come (a rejected deviation would be
  handled then, and recorded).

## 2026-10-18 — Week 6: test-set runs started (to-do step 4)

- **S0 scorer restricted on test.** `s0_score.py` scored all 8 configurations of any split it was
  given; on test that would have produced test scores for the ResNet-50 reference and the
  unselected DenseNets, against "Resolved decisions" #1 (only the selected S0 on test, so the
  choice cannot be revisited after seeing test results). Changed before running: on test it
  scores only `selection.json`'s configuration and refuses mixing test with other splits.
  `s0_score.py test --test` → `densenet121-res224-mimic_nb__pad__test.csv` (400) only.
- **R0:** `r0_knn.py test --test` rewrote `selection.json` with train and test results only
  (validation results dropped from the file). Re-run as `r0_knn.py val,test --test`: k = 150 and
  thresholds unchanged (deterministic, from train); the validation section of the report is
  byte-identical to `selection_report.md`. Report with test: `selection_report_with_test.md`
  (the original `selection_report.md` kept). Test numbers not looked at; they are read through
  `evaluate.py`.
- API runs started in parallel, each with the default USD 10 per-run limit; logs
  `data/runs/logs/test_*.txt`: `p1_adapted.py test --no-search-section` (checklist),
  `p2_adapted.py run test`, `p3_adapted.py run test`, `g_system.py g|g_ind|g_img test --test`.
  All six answered their first studies normally.
- Noted: the starter runners have no `--test` guard (unlike S0, R0 and G); not needed now, as
  this is the one planned test run.
- Progress check (≈ 20% done): all running; projected cost of the six runs ≈ $20 (estimate
  $25.5), P3 without runaways so far. **G-img, CXR206: invented citations** (CXR102, CXR205,
  CXR331) although no cases were given; caused by G's shared prompt wording and the schema's
  `cited_case_ids` slot; flagged by `validate()`, statuses kept by the export rule; 1 of 151
  G-img answers. Not changed (prompt frozen, test run underway, D7's identical-prompt design).
  Recorded in results "Observations"; for the week-7 failure review.
- Finished, all 400 / 400: G-img $2.75 (2 answers with invented citations: CXR206, CXR3080), P1-
  adapted-ns $2.65 (400 checklists at the top, 0 missing), G $4.05 (0 problems), G−ind $4.22
  (0 problems), P2-adapted $4.18 (400 checklists at the top, 0 missing). Exported / parsed to
  `test_checklist.csv`. P3 still running (runaways ≈ 3%, server 504 timeouts retried).
- **Correction of wording (this entry and the one above said "test results / numbers not looked
  at"):** precisely, **test performance has not been computed or seen**: no test label read, no
  hit / miss count, no metric. What was inspected: the run logs (study ID, format check OK /
  PROBLEMS, tokens; the two invented citations were seen there), run health (completeness,
  cost, thinking, timeouts, exporter counts), and one full test answer (G-img, CXR206: statuses,
  probabilities, citations, rationales) while explaining the invented citation. R0's test
  section of `selection_report_with_test.md` was written to a file and not opened. Nothing was
  changed after any of this (prompts frozen, thresholds and evaluation fixed beforehand).

## 2026-10-18 — Test set: P3 finished; evaluation run once; results §5.5 (to-do step 4 done)

- P3-adapted: 400 / 400, $5.37 (1,634 calls: agent loop + retries); 11 runaways (≈ 63k thinking),
  37 server 504 timeouts (retried, not billed); checklist parsed: 400 parsable, 396 at the top,
  0 missing. Test-run total $23.21; project total **$36.00** (80.0% of the USD 45 cap), as
  projected.
- `evaluate.py test --test`, run once: all 8 systems 400 / 400; 24,000 checks in
  `per_study.csv`; all assertions held (R0 recomputed scores = table; per-study outcomes =
  reported metrics; G's neighbours = R0's first 10, 400 / 400).
- **Primary (E1 #1): G − S0 macro AUROC +0.049 [+0.013, +0.087]** (G 0.754, S0 0.705). Gain in
  Lung Lesion (+0.19) and Cardiomegaly (+0.07). Decomposition: G-img − S0 +0.010 [−0.029,
  +0.050]; G − G-img +0.039 [+0.008, +0.071]; G − G−ind +0.020 [−0.008, +0.046]; G − R0 +0.028
  [−0.004, +0.060]. Starters (stated): G lower sensitivity, higher specificity than all three;
  F1 level with P1-ns and P2, better than P3 (+0.097). Copying: G agrees with wrong neighbours
  0.426 vs. G-img 0.399. Citations: 0.81–0.96 of cited cases have the finding vs. 0.24–0.43 of
  all neighbours (Lung Lesion 0.43 vs. 0.07).
- Framing (decision): lead with the pre-declared primary result; directly after it the
  decomposition (Gemini alone ≈ S0; the gain comes from retrieval and context); S0 not being
  the strongest model and possible Gemini memorisation stated as limits on the claim, next to
  it; starters: "G is more cautious, not more accurate in its verdicts".
- Write-up checked figure by figure against `report.md` before committing; three corrections in
  the draft: test-run cost ($23.21, not the old estimate $25.53); the reason test < validation
  (many more positives of hard, rare findings, equal weight in the macro average, not "rarer");
  k = 1 retrieval (BiomedCLIP better than P3's embedding for Atelectasis, Lung Lesion, Pleural
  Effusion, not "comparable").
- results: §5.5 written; G-img invented-citation observation updated (2 of 450: CXR206,
  CXR3080).

## 2026-10-18 — Week 7 started: design proposal (for decision)

- `week7_design.md`: settled A1–A3 (test set, cached outputs, no API; first pass by Claude,
  recorded as such, seeded samples; judged from text, not images). Open: W1 failure review
  (stratified seeded sample of 23 errors, fixed categories), W2 memorisation check (word
  3-gram containment of FINDINGS + IMPRESSION, without INDICATION, vs. 20 same-label reports;
  all six Gemini systems), W3 hand check of G (15 answers, status vs. rationale + basis of each
  "present" call).
- Facts checked for it: IU report sections are COMPARISON, INDICATION, FINDINGS, IMPRESSION;
  reports contain "XXXX" de-identification tokens (to be removed before comparing text).
- Noticed while checking the sections (not yet a finding): CXR206's report describes a lingula
  mass and an ill-defined right-lung opacity, so G-img's "present" calls for Lung Lesion and
  Lung Opacity there were correct although their citations were invented.
- No case judged yet; waiting for the decision.

## 2026-10-18 — Week 7 done: W1–W3 (to-do step 5)

- Decision: W1–W3 adopted as recommended; `week7_design.md` marked decided.
- **W2 code:** `scripts/evaluation/memorisation.py` + `scripts/tests/test_memorisation.py`
  (10 tests). One test was wrong on first run (my expected value: "a b c a b c" vs. "a b c" has
  3 distinct 3-grams, 1 found → 1/3, not 2/3); corrected, all pass. A shell edit had also broken
  the XXXX-token regex (backslashes lost); fixed with an exact edit and covered by a test.
  Pool exclusion detail fixed with the code, before running: every report shown to any system
  for the study (G's 10, P3's 6, P2's 1) is excluded from that study's baseline pool, so copying
  from a shown case cannot raise the baseline. Result: 384 / 400 studies; no 3-gram excess in
  any system (all CIs include 0; own > pool max 4.9–6.8% vs. 5% chance); small TF-IDF excess in
  all (+0.006 to +0.019), read as case-specific content, not recall.
- **W1/W3 samples:** `scripts/evaluation/week7_samples.py` (seed 2026). The first sheet printed
  G's answer twice for G cases; fixed and regenerated, sample identical (diffed).
- **W1 judgements** (23, `data/runs/review/failure_review.csv`): missed finding 6, copied
  neighbour 6, invented finding 4, definition mismatch 3, other 4 (threshold artefact 2,
  misclassified real abnormality 1, invented citation with correct verdicts 1). Patterns: old /
  non-rib fractures missed (G's rationales check "acute"); copying visible in wording; Cardiomegaly
  false alarms on low-volume / AP films; Lung Opacity definition mismatches; pooled threshold
  0.08 turns G's "absent, 0.10" into positives.
- **W3 judgements** (90, `data/runs/review/g_rationale_review.csv`): agree 88, unclear 2,
  disagree 0. Basis of 17 "present" calls: image 10, image + cases 4, image + indication 3,
  cases / indication alone 0.
- results §5.6 written; checked against the judgement files before committing: one correction
  (the copied-neighbour cases where G-img was right are the 3 sampled from that stratum, not
  "two of these").
- All judgements are a first pass by Claude from text only, recorded as such.

## 2026-10-18 — Decision: commit `data/runs/` to the private repository

- Question: commit `data/runs/` (until now excluded with all of `data/`) so that the week-7
  judgements and every result can be traced from the repository?
- Content checked first (151 MB): no images or image bytes, no API key; **report text quoted
  verbatim** (2,171 distinct training reports inside cached G prompts, test indications, P3's
  retrieved reports, review sheets with test reports); ground-truth labels of the test studies;
  ≈ 100 MB of image embeddings (`*.npy`, `*.npz`, P2 / P3 indexes).
- Licence: IU X-Ray / Open-i is CC BY-NC-ND 4.0. Its conditions apply to sharing with the public;
  keeping copies for own non-commercial research, including in a private repository, is within
  it. ND forbids publishing adapted material (embeddings, derived labels, processed text).
  Not a legal opinion.
- Checked: the repository is not publicly visible (anonymous GitHub API and web requests: 404).
- **Decision: commit `data/runs/` without `*.npy` / `*.npz`** (3,801 files, 43.5 MB, largest
  3.4 MB). Reasons: traceability of all results; the embeddings are the most clearly adapted
  material, most of the size, unreadable, and regenerable (≈ $0.70 of API calls).
- `.gitignore`: `data/*` ignored except `data/runs/`, embeddings still ignored. `SETUP.md` §4:
  source citation (Demner-Fushman et al., JAMIA 2016), licence link, and the rule that
  `data/runs/` must be removed from the repository and its history before it is ever made
  public.

## 2026-10-18 — Week 8: report finalised; sensitivity check changes the primary reading

- `results.md` finalised: header (final version, author); new **Summary** answering Q1–Q3 with
  the limits of the main claim; §5 intro no longer says S0 / R0 are not run on test; §6 Threats
  completed (memorisation result, one collection / one sample, enrichment and F1, hand
  judgements by an AI from text only, rationales as self-reports; "pending" removed).
- **Sensitivity check (promised in §6 on 2026-10-17, before any test run: macro averages without
  Lung Lesion and Fracture "if it changes a conclusion").** Computed with the same paired
  bootstrap: G − S0 +0.049 [+0.013, +0.087] (six findings) → **+0.020 [−0.020, +0.063] without
  Lung Lesion** → +0.008 [−0.028, +0.044] without Lung Lesion and Fracture. G − G-img: +0.039 →
  +0.040 [+0.005, +0.077] → +0.030 [−0.003, +0.065]. **It changes the reading:** the primary
  result rests largely on Lung Lesion, where S0's label does not match IU's codes; what retrieval
  and context add within G's design holds without it. Reported as limit 5 next to the primary
  result in §5.5, in the §5.5 summary, in the report Summary and in §6. The pre-declared result
  is kept as declared; this is reported as a sensitivity check, not a replacement.
- First computed inline, then saved as `scripts/evaluation/without_findings.py` (output
  `data/runs/eval/test/without_findings.md`), which reproduces the numbers exactly.

## 2026-10-18 — Week 8: final presentation

- `presentation/final_presentation.pptx` (35 slides, speaker notes on every slide), built by
  `presentation/build_final_deck.py` from the faculty template `NIK_prezentacio_sablon_2025.pptx`
  (its layouts and masters: navy title / section / closing slides, white content slides with the
  navy bands). The progress review decks are unchanged. Supervisor questions left out (request).
- Content, in order: premise; the three questions and eight systems; **data** (what IU X-Ray is,
  with the example study CXR2130's images and report; the cleaning funnel 3,955 → 3,604 with
  every check; MeSH ground truth and why not CheXbert; splits); **S0** (pipeline, example scores,
  how the DenseNet was chosen); **R0** (pipeline, k curve, example); **starters** (overview and
  P-numbering; the checklist question; for P1, P2, P3: as shipped, changes, and a real
  prompt → result for CXR2130); Q1 summary; **G** (pipeline, what it took from each starter,
  prompt → result for CXR2130, ablations and validation); the test protocol; results (primary,
  sensitivity check without Lung Lesion, decomposition, starters, mechanisms); week 7; limitations;
  conclusions.
- Example study CXR2130 chosen for having three findings and differing results (P1 5/6, P2 3/6,
  P3 3/6, G 6/6, S0 3/6, R0 4/6); the prompts and answers on the slides are the cached test
  records, abridged. Footnote on its slides: the report also mentions two nodules that are not
  MeSH-coded, so Lung Lesion counts as absent.
- Fonts: the template asks for Open Sans, which is not installed on this machine; the slide text
  is set in Calibri (installed and shipped with Office) so the rendered check matches what is
  shown here. The template's own elements keep their fonts.
- Checks: every slide rendered through PowerPoint and inspected; fixed in two rounds: template
  slide-number boxes carried a hard-coded total ("/39", "/34") → set to /35; seven titles wrapped
  behind the header band → one title size (32 pt) and shorter wording; chart axis labels and
  sizes; a table colliding with a footnote; empty panel space. Text dump: no leftover template
  text; package validator: passed. Built with python-pptx in a separate scratch environment (not
  added to the project venv).

## 2026-10-18 — Presentation redone as HTML; PowerPoint version removed

- The faculty template was given as a design reference only; the intended format is a single
  HTML deck like `progress_review_using_template.html`. A PowerPoint deck and a Python builder
  were not wanted. `final_presentation.pptx` and `build_final_deck.py` removed from the
  repository (they stay in the git history, commit 654d3d3).
- `presentation/final_presentation.html`: the progress review's page (styles, embedded ÓE NIK
  logos, navigation, notes, footer) with its slides replaced by the 35 final slides, written
  directly; same content as described in the entry above. Charts are inline SVG / CSS, drawn by
  hand; the example X-rays are linked from `../data/raw/iu_xray/images/` (shown only where the
  dataset is present, as `data/raw/` is not in the repository). Keys as before: ← → navigate,
  N notes, F fullscreen, Ctrl+P to PDF.
- Checked by screenshotting every slide in headless Edge at 1440 × 1080; fixed: monospace prompt
  text too small (Courier New fallback), a cramped six-step flow, overlapping axis labels, a
  wrapped code word.

## 2026-10-18 — Presentation edits; how G's cut-off 0.08 was chosen

- Edits on request: "Week 6 / Week 7" labels removed; P1's five prompt sections explained
  (slide 14, from `ai_medical_imaging.py`); dollar amounts and API costs removed; G's prompt
  step 4 shown in full on the G example (where the probability is asked for).
- The 0.08 cut-off appeared without explanation. New slide 25 "G's cut-off: how 0.08 was
  chosen" (deck now 36 slides): G's probability → yes / no needs a cut-off; every value tried on
  the 400 validation studies; scored by Youden's J; one cut-off pooled over the six findings;
  highest J at 0.08. Its trade-off table (macro sensitivity / specificity at cut-offs 0.05–0.50
  and for G's own "present") was computed on validation only and added to results §5.3. S0 / R0
  / G slides now describe the cut-off in the same words; slide 30's G rows renamed "G · its own
  'present'" and "G · probability ≥ 0.08 (slide 25)".
- Quick check made while answering a question, not used anywhere (recorded here only): on
  validation G's stated probabilities rank well but are overconfident (stated ≈ 0.84 → present
  35%; mean stated 0.095 vs. base rate 0.047). Calibration was out of scope this semester
  (thesis plan H2). Correction to an earlier remark in conversation: for calibrated
  probabilities the Youden-optimal cut-off sits near the base rate (≈ 0.05 here), not 0.5, so
  0.08 by itself is not evidence of miscalibration.
- `presentation/qa_prep.md`: 100 question-answer pairs for the presentation (terms, data, S0 /
  R0, starters, G, results, limits), numbers from the deck and this report.
- The two example X-rays (CXR2130) were linked from `data/raw/`, which is not in the repository,
  so they were missing on another machine. Now embedded in the HTML as base64 (the original PNG
  bytes, unmodified), like the template's logos: the deck is one self-contained file (504 KB).
  Licence: CC BY-NC-ND allows verbatim copies with attribution; the slide credits Open-i and the
  licence.
- `presentation/summary_presentation.html`: a 17-slide summary of the final deck, same template
  and self-contained. Kept: the dataset (IU X-Ray, cleaning + ground truth on one slide, splits),
  S0, R0, the starters (overview + one slide each for P1, P2, P3 with the CXR2130 example), G
  (how it works with its example; what it took from each starter), then two results slides and
  the conclusions. Left out: the cut-off derivation, the test-protocol slide, the week-7
  analyses and limitations (one line in the speaker notes). No new numbers; all taken from the
  full deck. Then, on request, the S0 weight-choice slide ("S0: how the model was chosen") added
  back after the S0 slide (18 slides). Its right panel reworded on request: instead of "the
  frozen plan names a DenseNet", it explains the eligibility rule (no weights trained on
  Open-i → the combined DenseNet excluded; of the single-dataset DenseNets only CheXpert,
  MIMIC-nb, MIMIC-ch cover all six findings; highest macro AUROC on train) and that the
  ResNet-50, also IU-free, was kept as reference because its Lung Lesion output is untrained
  (results §2.3b).
- Summary deck, new slide 15 "Right and wrong calls, all systems" (19 slides), on request: per
  system, the 2,400 test calls split into correct / missed / false alarm, as a table with
  stacked bars. Counted from `data/runs/eval/test/per_study.csv` (LLM systems on their stated
  verdicts, S0 / R0 on their thresholds, G also at p ≥ 0.08). Added a trivial "always absent"
  row (2,076 correct) to show that raw counts reward caution, so they are not a comparison
  metric; this is descriptive only and adds no new comparison.
- Summary deck reordered on request: S0, S0 weights and R0 moved after G (order: data →
  starters → G → reference systems → results). Section numbers in the slide headers renumbered,
  the premise table reordered to match; two forward references to things not yet explained
  reworded (G's retrieval box no longer says "R0's index"; G's cut-off line no longer names
  Youden's J, which is now explained later on the S0 slide).
- Summary deck: strict alternation of the two white template looks restored (slides 14 and 15
  swapped looks); navy stays for title, G, conclusions and closing. Looks carry no meaning.
  Then, on request, the white-with-colour-logo look dropped entirely: every content slide uses
  the navy-bar look; navy background for title, G, conclusions and closing.
- **Per-study table added to `evaluate.py`** (before the test evaluation): `per_study.csv`, one
  row per study × finding × system × reading with truth, verdict, outcome (hit / miss / false
  alarm / correct absent) and score. Needed for the week-7 failure review (which studies each
  system got wrong); the metrics are unchanged. The script asserts that counting the table
  reproduces every reported sensitivity and specificity. 3 new tests (21 in total, all pass);
  validation run: 15,600 checks written, assertion holds.

## 2026-10-24 — Further research possibilities (results §7)

- Three questions raised after the report was finalised: (1) Gemini Embedding 2 vs. BiomedCLIP:
  results, impact, cost; (2) improvements to G suggested by the results; (3) whether a system
  like G carries over to other imaging tasks (skin photographs, CT, …) more easily than a
  specialist model.
- Decision: answered from the existing results only (no runs, no API calls), recorded in
  `results.md` as a new **§7 "Further research (possibilities)"** after §6; header notes the
  addition. §1–§6 and the Summary are unchanged.
- Sources: §4.2 retrieval comparison, §5.5 (E2–E4, sensitivity check), §5.6 (W1), §6; costs from
  `scripts/tools/spend.py` (Gemini Embedding: P2 train index $0.33, P3 index $0.38, ≈ $0.83 in
  all, ≈ 2.3% of $36.00; BiomedCLIP local, $0, run time not logged); the calibration quick check
  from the 2026-10-18 entry "Presentation edits".
- Main points: BiomedCLIP retrieves better at small k (k = 5 −0.077 for Gemini), cost difference
  negligible, effect on a full system not measured (possible check: G with Gemini retrieval on
  validation, ≈ $4). Seven possible G changes, three computable from cached outputs without API
  (calibration / per-finding thresholds, G + S0 average, k = 150 vote line). Adaptability: easier
  to re-target (no training; definitions in the prompt, shown by the Lung Lesion mismatch), better
  accuracy in other modalities not shown; possible test on dermoscopy (ISIC / HAM10000) in the
  thesis.
- Any new test-set run would be after the fact (test declared "run once", §5.4); stated in §7.
- To-do list: item 9 added (open, not scheduled).

## 2026-10-18 — Next steps (current to-do list)

Moved here unchanged from the entry "Prompt 2 frozen; next steps", so the current to-do list
is always the last section of the diary. Script paths now follow the reorganised folders
(`scripts/README.md`).

**Next steps (replaces the earlier lists; in order)**
1. ~~**G and G−ind on validation**~~ — done 2026-10-18 (entry above; results §5.3; thresholds
   in `data/runs/g__prompt2/selection.json`, `data/runs/g_ind__prompt2/selection.json`).
2. ~~**G-img pilot**~~ — done 2026-10-18 (entry above). Was: (50 studies, ≈ $0.3): schema compliance and behaviour of the image-only arm
   before its test run (D7).
3. **Evaluation script for the test set** (no API): per finding and macro sensitivity,
   specificity, F1 for every system (binary: stated status / checklist; S0 and R0 at their
   thresholds; G family also at the pooled threshold); AUROC for S0, R0 and the G family; paired
   bootstrap CIs (studies resampled) for the planned differences (plan §5); retrieval quality
   (precision@k) for R0, P2, P3, G; copying rate (G vs. its 10 neighbours' majority vote);
   citation label agreement (G's cited cases for its "present" calls); G vs. G−ind; G vs. G-img.
   Write and test it on validation data before the test runs.
   **Method decided 2026-10-18 (E1–E5, results §5.4, `eval_design.md`); this replaces the
   wording above where they differ (copying rate: kNN decision at k = 10 with G-img control,
   not a majority vote). Next: write `evaluate.py` + tests.**
   **Done 2026-10-18** (entry "Evaluation script written and checked on validation"). On test:
   `evaluate.py test --test` after all step-4 runs.
4. **Week 6, test set, once:** S0 (`s0_score.py test --test`), R0 (`r0_knn.py test --test`),
   P1-adapted-ns, P2-adapted, P3-adapted (checklist question), G, G−ind, G-img (`--test`).
   Estimated API cost ≈ $14.6 (starters) + $8.4 (G, G−ind) + $2.5 (G-img); each run under the
   USD 10 limit (P3-adapted ≈ $8.4 expected, may stop at $10 if runaways are frequent).
   **Done 2026-10-18:** all 400 / 400, $23.21; evaluated once; results §5.5.
5. **Week 7, analysis:** (metrics and CIs done by `evaluate.py`; remaining items below; the
   memorisation check needs its method decided first, like E1–E5) metrics and CIs; failure-case review (~20 errors across systems); the
   after-the-fact memorisation check (plan §8: overlap of free text with the study's own report
   vs. same-label reports); hand check of G's status vs. its rationale on a sample (as for the
   starters' checklists).
   **Done 2026-10-18** (W1–W3, `week7_design.md`; results §5.6).
6. ~~**Open decision**: the 50-study P1 web-search arm~~ — decided 2026-10-18: not run (entry above).
7. ~~**Supervisor questions**~~ — completed for the time being 2026-10-18: one-pager
   `presentation/supervisor_questions.md` (entry above); not blocking; answers folded in later.
   **Removed 2026-10-19:** the one-pager is no longer needed; deleted from the repository (still
   in git history).
8. **Week 8:** write-up and presentation. **Done 2026-10-18:** `results.md` final (Summary,
   threats, sensitivity check); `presentation/final_presentation.html` (entries above).
9. **Further research (added 2026-10-24, not scheduled):** possibilities in `results.md` §7.
   Free first steps (cached outputs, validation, no API): calibration / per-finding thresholds,
   G + S0 average, k = 150 vote line. Paid: G with Gemini Embedding retrieval (≈ $4, validation).
