# Starter Agents vs. a Grounded Image + Text System for Chest X-ray Decision Support — Results

Semester project report, Antal Ádám (final version, 2026-10-18; §7 "Further research" added
2026-10-24, possibilities only; the explanation of S0's Lung Lesion weakness corrected
2026-10-31, see "Corrections"). The plan it follows is
`semester_plan.md`, frozen as of 2026-10-16. Where the work differs from the plan, the difference
is recorded here, under "Deviations from the plan", and in `diary.md`. The plan itself is not
edited.

---

## Summary

**Task.** Detect six findings (Lung Opacity, Cardiomegaly, Atelectasis, Lung Lesion, Pleural
Effusion, Fracture) on frontal chest X-rays of the public IU X-Ray collection, as decision
support. Ground truth: the collection's manual MeSH codes. Eight systems were each run **once**
on a positive-enriched test set of 400 studies, with every choice, prompt and threshold fixed
beforehand on train and validation data.

**Q1 — How fit are the three starter apps?** None ran on chest X-rays as shipped: each needed at
least one change beyond the planned minimum (retired models; a prompt that ordered a removed
search tool, 62% failed calls; an output cap that cut the answers). Adapted, all three answer
reliably, but their verdicts are coarse: on test, macro sensitivity / specificity is 0.52 / 0.86
(P1-adapted-ns), 0.42 / 0.90 (P2-adapted) and 0.43 / 0.75 (P3-adapted). P3's generator never sees
the patient's image and answers about its retrieved neighbours (§3, §5.5).

**Q2 — What is worth reusing?** The structured radiology prompt (P1), retrieval of similar cases
(P2, P3) and citation objects (P3). Missing everywhere: clinical text as input, the patient's
image *and* textual evidence together, a per-finding score, definitions aligned with the ground
truth, and machine-checkable citations. The built system G fills these: the patient's image +
the indication + the reports of the 10 most similar training cases (BiomedCLIP) → one Gemini call
→ per finding a status, a probability, cited case IDs and a rationale (§4).

**Q3 — How do the results differ?**
- **Primary, pre-declared comparison: G ranks findings better than the specialist image model S0**
  (zero-shot TorchXRayVision DenseNet): macro AUROC **0.754 vs. 0.705, +0.049 [+0.013, +0.087]**.
  **The gain rests largely on Lung Lesion,** a finding on which S0 is near chance in every split
  (cause not established; not a difference in definition, §5.1): without
  Lung Lesion the difference is +0.020 [−0.020, +0.063], not distinguishable from zero.
- **Where it comes from:** Gemini reading the image alone (G-img) is level with S0 (+0.010
  [−0.029, +0.050]); adding the retrieved cases and the indication gives a gain (G − G-img
  +0.039 [+0.008, +0.071]) that holds without Lung Lesion (+0.040 [+0.005, +0.077]). Neither the
  indication alone nor the LLM over plain retrieval (R0) is shown to add a gain on its own.
- **Against the starters,** G's stated verdicts are more cautious, not more accurate (F1 level
  with P1-adapted-ns and P2-adapted, better than P3-adapted). Its advantages are a usable score
  and citations that can be checked: 81–96% of the cases it cites for a present finding have that
  finding (Lung Lesion 43%), against 24–43% of all cases it was shown.
- **Mechanisms (week 7):** the neighbours pull G's verdicts by about 3 points where they are
  wrong, and when G copies, its rationale says so; G's statuses agree with its own rationales
  (88 / 90), and its "present" calls are stated in terms of the image. No system shows verbatim
  recall of the IU reports.

**Limits of the main claim.** S0 is the plan's DenseNet, not the strongest specialist available
(a ResNet-50 scored 0.808 vs. 0.728 on train and was not run on test); Gemini has probably seen
IU X-Ray during training (no recalled wording was found, but recalled labels cannot be excluded);
G receives finding definitions that the other systems do not; the test set is a single 400-study
sample from one hospital collection. The result is "a grounded image + text system beats the
planned zero-shot specialist on this collection, mainly on one finding where the specialist
performs at chance level", not "multimodal retrieval beats image models". The firmer finding is the
within-design one: giving Gemini the retrieved cases and the indication improves its ranking
over the image alone.

**Cost.** API spend USD 36.00 of a USD 45 monthly cap (test runs USD 23.21).

---

## 1. Questions

- **Q1 (fitness):** How do the three starter projects work, and how well do they perform on
  chest-X-ray finding detection when run with minimal adaptation?
- **Q2 (strengths):** Which of their components are worth carrying into a grounded image + text
  system, and which gaps must be filled?
- **Q3 (comparison):** How much do the built system's results differ from the starters' results,
  from a single specialist image model (S0), and from retrieval alone (R0)?

All work is decision support only and uses public, de-identified data.

---

## 2. Data

### 2.1 Source

Indiana University Chest X-ray Collection (IU X-Ray), distributed by Open-i (NLM).
Reports archive `NLMCXR_reports.tgz` (archive dated 2017-06-06), downloaded 2026-10-16.
Licence stated in the files: CC BY-NC-ND 4.0.

| Item | Count |
|---|---|
| Reports (studies) | 3,955 |
| Reports with at least one image | 3,851 |
| Reports without images | 104 |
| Images | 7,470 |
| Images per report: 1 / 2 / 3 / 4 / 5 | 446 / 3,208 / 181 / 15 / 1 |

Images archive `NLMCXR_png.tgz` (1.36 GB, archive dated 2016-09-06), downloaded 2026-10-16:
7,470 PNG files (plus a stray Windows `Thumbs.db`). Checked with `scripts/data_prep/check_image_mapping.py`:
- Every PNG matches exactly one image listed in the reports, and the reverse. PNG names drop the
  middle `_<n>_` of the report's image ID (`CXR1000_1_IM-0003-1001` → `CXR1000_IM-0003-1001.png`).
  The one exception is CXR1, whose two PNGs keep it.
- All 7,470 images open without error. All are stored as RGB, although the content is greyscale.
- Orientation: 3,332 portrait, 2,610 landscape, 1,528 square.
- **The release has no view label (frontal vs. lateral).** Neither the reports nor the file
  names say which image is which. The report caption gives only the exam type (e.g. "Xray Chest
  PA and Lateral"). Selecting the frontal image (plan §2) therefore needs a view classifier,
  checked by eye on a sample (§2.4, pending).

### 2.2 Patient identifiers and split level

**Finding: the public release contains no patient identifier.** Every element and ID attribute
in all 3,955 report files was listed (`scripts/data_prep/check_patient_ids.py`). The only identifiers are
study/report IDs (`uId` = CXR####, `pmcId`, `IUXRId`) and image IDs.

The `IM-####` part of image names (e.g. `CXR1_1_IM-0001-3001`) was tested as a possible hidden
patient link, since 813 of its values occur in more than one report
(`scripts/data_prep/check_im_series.py`). It is not one. It is a sequential counter assigned over the
report IDs sorted as text (CXR1, CXR10, CXR100, …):
- along that order the number decreases only once (a restart at CXR1015);
- reports sharing a value are neighbours in that order (2 exceptions);
- a shared group mixes clearly different patients. For example, IM-0013 covers a female with
  chest pain, a bone-marrow-transplant patient and a Burmese male after TB treatment.

**Consequence: the train/validation/test split is by study, not by patient.** A patient with
several studies may appear in more than one split. If that happens, retrieval could return the
same patient's other exam, which would favour the retrieval systems (R0, G, P2, P3) over P1 and
S0.

**Resolution: the collection holds one study per patient, by design.** The dataset paper
(Demner-Fushman et al., *J Am Med Inform Assoc* 2016;23(2):304–310, doi:10.1093/jamia/ocv080,
full text PMC5009925) states:
- "We limited the retrieval to 4000 such reports each from a different patient and took 2000 from
  each institution."
- "In no case did we take more than one study per patient."

So **the study-level split is equivalent to a patient-level split**, and repeat patients cannot
cross splits. The concern above does not arise.

The prior-exam references in the reports are consistent with this. The COMPARISON section refers
to a prior exam in 1,909 of 3,955 reports, and the findings or impression suggest one in 1,105.
Those earlier exams existed but were not included in the collection.

Other selection facts from the paper, relevant later:
- **Outpatient studies** were targeted, so the collection is likely dominated by normal or
  mild findings (to be confirmed by the label statistics, §2.4).
- Inclusion required labelled report sections and **at least a PA and a lateral view**. The 446
  public studies with a single image are explained by the public-release step: images showing
  teeth, jaw, jewellery or partial skull were removed. 41 studies were removed entirely, and
  2 for identifying information.
- The paper's public-subset counts (**3,955 reports, 7,470 images**) match the downloaded files
  exactly (§2.1).

### 2.3 Repeat-patient screen

**Not needed for repeat patients** (§2.2: one study per patient). A near-duplicate image check
was run on the 3,690 selected frontal images (§2.4) as a general leakage guard.

- **Embedding similarity** (BiomedCLIP, `scripts/data_prep/embed_and_dedup.py`). Nearest-neighbour cosine
  similarity percentiles: 50th 0.970, 90th 0.981, 99th 0.985, 99.9th 0.988, max 0.988. No pair
  reaches 0.99, and the top of the distribution has no outlier. The 30 most similar pairs were
  checked by eye: all are different patients with similar-looking normal films, with a few "hub"
  images recurring (e.g. CXR3827). None is a duplicate.
- **Pixel level** (`scripts/data_prep/pixel_duplicates.py`): no identical files (MD5) and no identical
  16×16 quantised thumbnails (which would catch resized or re-encoded copies).

**Result: no duplicates; nothing removed.**

Side observation for G and R0: BiomedCLIP packs chest X-rays into a narrow similarity band
(median nearest neighbour 0.970). Retrieval ranking will depend on small differences, and
similarity scores will not be very informative as confidence signals. To be kept in mind when
choosing k and when reading the retrieval-quality results.

### 2.3b S0 weights (TorchXRayVision)

Checked in the installed package source (`torchxrayvision` 1.5.5, `models.py`):
- The combined **`densenet121-res224-all`** weights were trained on
  "nih-pc-chex-mimic_ch-google-openi-rsna". **Open-i (= IU X-Ray) is included, so these weights
  cannot be used for S0.**
- Options that exclude Open-i:
  - single-dataset DenseNets: `-nih`, `-pc` (PadChest), `-chex` (CheXpert), `-rsna`,
    `-mimic_nb`, `-mimic_ch`;
  - **`resnet50-res512-all`**, trained on "pc-nih-rsna-siim-vin" at 512×512, with the full
    18-label output.
- Coverage of the six findings (checked in `datasets.py`, which training datasets define each
  label): `-chex`, `-mimic_nb` and `-mimic_ch` have all six as trained outputs. `-nih`, `-pc` and
  `-rsna` each lack at least two. **`resnet50-res512-all` lists Lung Lesion, but none of its five
  datasets defines it, so that output is untrained** (the library gives it no operating
  threshold); it can score Lung Lesion only through a proxy, max(Nodule, Mass).
- Choice of weights and preprocessing: §5.1.

### 2.4 Frontal-image selection

No view label exists (§2.1), so views were classified zero-shot with BiomedCLIP
(`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`). Each image is compared with the
averaged text embeddings of three "frontal" prompts and three "lateral" prompts
(`scripts/data_prep/classify_views.py`), giving a frontal probability `p_frontal` per image
(`data/processed/views.csv`).

**First pass, threshold 0.5.**

| Check (by eye) | Sample | Result |
|---|---|---|
| Random images | 60 | 60/60 correct |
| Studies with 2+ images called frontal | 40 of 240 images | 39 real frontal views (studies genuinely have repeat frontals); 1 noise image |
| Studies with no image called frontal | 40 of 195 images | 37 lateral; 3 frontal views missed (scores 0.12–0.38) |
| Uncertain scores (0.2–0.8) | 40 of 45 | Mostly frontal views (portable or degraded), many wrongly called lateral; 4 blank/noise images |

So a fixed 0.5 threshold misses degraded frontal views, and a few images are unusable.

**Unusable images:** 4 of 7,470.
- 3 near-blank (grey-level standard deviation < 15 on a 128 px copy; next-lowest image 20.8).
- 1 pure noise (high-frequency to contrast ratio 0.387 at full resolution; next-highest 0.08,
  median 0.032).

**Final rule** (`scripts/data_prep/select_frontal.py`):
1. Drop the 4 unusable images.
2. Per study, take the image with the highest `p_frontal`.
3. Accept it if `p_frontal` ≥ 0.05.

No study's best image falls between 0.02 and 0.05, so the cut-off is not sensitive.

| Check (by eye) | Sample | Result |
|---|---|---|
| Selected images with `p_frontal` < 0.5 | all 11 | 11/11 frontal |
| Excluded studies' best image | 40 of 161 | 40/40 lateral |

**Result:** 3,690 studies have one frontal image (`data/processed/frontal.csv`).
- 3,955 reports − 104 without images − 161 lateral-only studies = 3,690.
- For the 117 studies with more than one image scoring ≥ 0.5, the highest-scoring one is used.
- The 161 lateral-only studies are excluded, as the plan specifies. They are mostly
  single-image studies; whether they differ in findings from the rest will be checked with the
  labels (§2.4b).

### 2.4b Labels and findings

**Labeller.** CheXbert (Smit et al. 2020), 14 CheXpert observations, applied to FINDINGS +
IMPRESSION (`scripts/data_prep/chexbert_labels.py`).
- Weights: `chexbert.pth` from Hugging Face `StanfordAIMI/RRG_scorers`, as used by the
  `f1chexbert` package. The package source was read before use; the weights are loaded with
  `weights_only=True` and a strict key check.
- The `f1chexbert` package's default "rrg" mode counts *uncertain* as positive. This project
  uses the original CheXbert classes and counts uncertain as **negative**, as fixed in plan §5.
- 3,955 reports labelled; 28 have empty findings + impression.

**Positive counts** (usable = the 3,690 studies with a frontal image; uncertain counted
negative):

| Observation | Positives (usable) | % | Uncertain | % positive among the 161 excluded lateral-only studies |
|---|---|---|---|---|
| No Finding | 1,431 | 38.8 | – | 27.3 |
| Lung Opacity | 517 | 14.0 | 3 | 18.0 |
| Cardiomegaly | 415 | 11.2 | 53 | 19.9 |
| Atelectasis | 216 | 5.9 | 134 | 9.9 |
| Lung Lesion | 165 | 4.5 | 15 | 4.3 |
| Support Devices | 163 | 4.4 | 0 | 15.5 |
| Pleural Effusion | 129 | 3.5 | 36 | 5.6 |
| Fracture | 127 | 3.4 | 5 | 2.5 |
| Enlarged Cardiomediastinum | 70 | 1.9 | 199 | 0.6 |
| Pneumonia | 68 | 1.8 | 56 | 3.1 |
| Pleural Other | 56 | 1.5 | 20 | 1.2 |
| Edema | 49 | 1.3 | 47 | 1.9 |
| Consolidation | 31 | 0.8 | 6 | 1.2 |
| Pneumothorax | 20 | 0.5 | 7 | 0.0 |

As the dataset paper's outpatient focus suggested (§2.2), most findings are rare. Only seven
observations have more than ~100 positives (Lung Opacity through Fracture above).

The excluded lateral-only studies are somewhat sicker: more Cardiomegaly (19.9% vs 11.2%) and
Support Devices (15.5% vs 4.4%), fewer No Finding (27.3% vs 38.8%). They are 4% of studies with
images, so the effect on the comparison is small, but it is a selection effect (§6).

**Check against the dataset's manual codes.** IU reports carry manual MeSH "major" codes
assigned by the dataset authors, an independent reference. Agreement with CheXbert positives
(`scripts/data_prep/label_stats.py`, usable studies):

| Observation | Both | CheXbert only | MeSH only | Agreement | κ |
|---|---|---|---|---|---|
| Cardiomegaly | 299 | 116 | 20 | 96.3% | 0.79 |
| Pleural Effusion | 121 | 8 | 17 | 99.3% | 0.90 |
| Atelectasis | 203 | 13 | 93 | 97.1% | 0.78 |
| Edema | 35 | 14 | 6 | 99.5% | 0.78 |
| Pneumothorax | 18 | 2 | 4 | 99.8% | 0.86 |
| Pneumonia | 32 | 36 | 4 | 98.9% | 0.61 |
| Consolidation | 26 | 5 | 2 | 99.8% | 0.88 |
| Fracture | 80 | 47 | 3 | 98.6% | 0.76 |
| Lung Opacity (≈ Opacity / Airspace Disease / Infiltrate / Consolidation / Pulmonary Edema) | 417 | 100 | 108 | 94.4% | 0.77 |
| No Finding (≈ "normal") | 1,227 | 204 | 116 | 91.3% | 0.82 |

Agreement is substantial (κ 0.76–0.90) except Pneumonia (0.61).

**A systematic CheXbert error on IU (Cardiomegaly).** Reading the 116 CheXbert-only Cardiomegaly
cases:
- **55 come from one templated normal report**: "The heart and lungs have XXXX XXXX in the
  interval. Both lungs are clear and expanded. Heart and mediastinum normal." CheXbert labels
  55 of the 62 reports with this template as Cardiomegaly, although the text states the heart is
  normal. That is **13% of all 415 Cardiomegaly positives**.
- Other errors include a missed negation ("no cardiomegaly with bilateral small pleural
  effusions", CXR546 → positive).
- 36 say "upper limits of normal" or "borderline", which can reasonably be read either way.
- Removing the de-identification masks (`XXXX`) before labelling (`--drop-masks`) does **not**
  fix it (Cardiomegaly κ 0.79 → 0.79; other observations within ±0.03). CheXbert misreads the
  phrase itself. No simple preprocessing fix was found.

**S0 label coverage** (relevant to finding selection): TorchXRayVision outputs cover Lung
Opacity, Cardiomegaly, Atelectasis, Lung Lesion, Effusion and Fracture, but **not Support
Devices**.

**Decisions (2026-10-16):**
1. **Ground truth = the manual MeSH codes.** CheXbert stays as the labeller for the systems'
   free-text outputs, as the plan intended. Its agreement with the ground truth then measures
   the labeller's error directly.
2. **Six findings**, all scored by S0: Lung Opacity, Cardiomegaly, Atelectasis, Lung Lesion,
   Pleural Effusion, Fracture. Support Devices is dropped.

**MeSH → finding mapping** (`scripts/data_prep/mesh_ground_truth.py`; fixed before any system was run).
A study is positive if any of its codes matches:

| Finding | MeSH codes |
|---|---|
| Lung Opacity | Opacity or Density at a lung location (or no location given); Airspace Disease; Infiltrate; Consolidation |
| Cardiomegaly | Cardiomegaly; Cardiac Shadow/enlarged. **"borderline" → negative** (consistent with uncertain → negative, plan §5) |
| Atelectasis | Pulmonary Atelectasis |
| Lung Lesion | Nodule, Mass or Cavitation at a lung location (or no location given). Calcified granulomas excluded (benign, incidental; CheXbert's count shows it does not count them either) |
| Pleural Effusion | Pleural Effusion |
| Fracture | Fractures, Bone (healed included) |

Studies coded "No Indexing" (never coded) are excluded: 86 of the 3,690 usable studies.
**Ground-truth set: 3,604 studies** (`data/processed/ground_truth.csv`); 2,623 have none of the
six findings.

| Finding | Positives | % | CheXbert vs. ground truth: sensitivity | specificity | κ |
|---|---|---|---|---|---|
| Lung Opacity | 519 | 14.4 | 0.77 | 0.96 | 0.73 |
| Cardiomegaly | 295 | 8.2 | 0.99 | 0.98 | 0.87 |
| Atelectasis | 296 | 8.2 | 0.69 | 1.00 | 0.78 |
| Lung Lesion | 104 | 2.9 | 0.87 | 0.98 | 0.66 |
| Pleural Effusion | 138 | 3.8 | 0.88 | 1.00 | 0.90 |
| Fracture | 83 | 2.3 | 0.96 | 0.99 | 0.76 |

Most of the templated reports CheXbert misread as Cardiomegaly were never coded ("No Indexing")
and are therefore excluded. On the remaining set, CheXbert's Cardiomegaly κ is 0.87 (0.79
before). The labeller error that remains, and that will affect the starter comparison, is
mainly lower sensitivity for Atelectasis (0.69) and Lung Opacity (0.77), and lower agreement
for Lung Lesion (κ 0.66).

### 2.5 Splits

`scripts/data_prep/make_splits.py` (seed 2026) → `data/processed/splits.csv`. The collection has one
study per patient (§2.2), so this study split is a patient split.
- **Test (400):** positive-enriched. Findings are processed rarest first, sampling positives
  until each has ≥ 35; the rest is filled at random.
- **Validation (400):** random, natural prevalence. The 50-study pilot is drawn from it.
- **Train (2,804):** everything else; the only source for retrieval indexes.
- Code assertions: the three sets are disjoint and cover all 3,604 studies; pilot ⊂ validation.

| Split | n | Lung Opacity | Cardiomegaly | Atelectasis | Lung Lesion | Pleural Effusion | Fracture | None of the six |
|---|---|---|---|---|---|---|---|---|
| Train | 2,804 | 396 | 206 | 221 | 49 | 83 | 43 | 2,112 |
| Validation | 400 | 39 | 23 | 26 | 11 | 12 | 2 | 319 |
| Pilot (⊂ val) | 50 | 5 | 2 | 1 | 1 | 1 | 0 | 42 |
| Test | 400 | 84 | 66 | 49 | 44 | 43 | 38 | 192 |

**Issue:** validation has too few positives for rare findings (Fracture 2, Lung Lesion 11,
Pleural Effusion 12) to tune per-finding thresholds there, as plan §2 intends. Enriching the
test set also leaves the train index thin for the same findings (Lung Lesion 49, Fracture 43).

**Decision (2026-10-16): keep these splits; tune thresholds as follows.**
- **S0:** per-finding thresholds tuned on the train split. S0 was never trained on IU data
  (§2.3b), so train is unseen data for it, with enough positives.
- **R0:** per-finding thresholds and k tuned on train by leave-one-out (each train study is
  scored from its neighbours, excluding itself).
- **G and G−ind:** one decision threshold on the confidence, pooled across the six findings, tuned
  on validation (~113 positives in total).
- **Starter arms:** no tuning (their free text is labelled by CheXbert).

The thin train index for Lung Lesion and Fracture is a known limitation for the retrieval
systems (R0, G, P2, P3) on those two findings.

---

## 3. Q1 — The starter projects and their fitness for the task

### 3.1 Availability of the starters' models (checked 2026-10-16)

`scripts/outdated/check_api_key.py`, with the project's Gemini API key (free tier):

| Starter | Model hard-coded in the starter | Result |
|---|---|---|
| P1 | `gemini-2.5-pro` | **Not available**: "no longer available to new users" (Google suggests `gemini-3.1-pro-preview`) |
| P2 | `gemini-2.5-flash` | **Not available**: same message (Google suggests `gemini-3.8-flash`) |
| P3 | `gemini-3-flash-preview` | Available |
| P3 (embeddings) | `gemini-embedding-2` | Available (default dimension 3,072) |

Both models are still *listed* by the API but refuse new users. The plan's risk "starter models
retired" (§3, §7) applies to P1 and P2.

Successor candidates, tested with a one-word request:
- `gemini-3.1-pro-preview`: quota error on the free tier. Google's pricing page lists **no free
  tier** for it; paid $2.00 / $12.00 per 1M input/output tokens.
- `gemini-3.8-flash`: temporarily unavailable ("high demand", 503). Free tier available; paid
  $0.75 / $3.75, rising to $1.50 / $7.50 on 2027-01-01.
- `gemini-3-flash-preview`: works; free tier; paid $0.50 / $3.00.
- **Requests for `gemini-3.5-flash` were answered by `gemini-3.6-flash`.** Some model names are
  aliases, so every cached response records the model that actually answered
  (`model_version`).

**Decision (2026-10-16):** P1 and P2 both use **`gemini-3-flash-preview`**,
the model P3 already uses.
- P2 → **P2-adapted** (already renamed for the embedder swap; the model change is a second
  adaptation).
- P1 → **P1-adapted**: its tier changes from Pro to Flash, a larger departure from the shipped
  app than a same-tier successor would be.
- Consequence: all three starters now run on the same generation model. Their differences come
  from design (what reaches the model, prompt, retrieval) rather than from model versions. That
  removes the model confound the plan noted between starters, but the starters are less
  faithful to their shipped versions.

**Free-tier limit, measured (2026-10-16):** after about 20 generation requests to
`gemini-3-flash-preview` in one day, the API returned 429 with quota
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, **limit 20**, model `gemini-3-flash`.
Requests that failed with 503 (overload) during retries appear to count toward it. At 20 per
day, the ~3,850 requests planned on this model would take about 190 days. **The free tier is
not usable for this study beyond a handful of test calls.**

### 3.2 P2-adapted (from `vision_rag`)

Runner: `scripts/starters/p2_adapted.py`.

**Kept as shipped:**
- embed → top-1 retrieval (`np.dot`, `argmax`, no similarity threshold) → generate;
- P2's image preparation (`resize_image`, copied verbatim). It never triggers on IU: all
  frontal images are 512 px wide and at most 0.34 MP, against P2's 2.46 MP limit;
- P2's answer prompt.

**Adaptations:**
1. A batch runner replaces the Streamlit UI.
2. Embedder: Gemini Embedding 2 at default settings (3,072 dimensions) instead of Cohere
   embed-v4.0. Images are embedded one per request, with no task type. The returned vectors
   have length 1.0, so P2's plain `np.dot` is a true cosine similarity, as in the shipped code.
3. Query: the patient's frontal image (image → image retrieval) instead of a text question.
4. Generation input: the patient's image **and** the retrieved image. The prompt's "the
   following image" becomes "the following images", plus one added sentence: "The first image
   is the patient's chest X-ray; the second image is the most similar image retrieved from the
   database."
5. Model: `gemini-3-flash-preview` (§3.1).

**Other details:**
- Index: train-split frontal images only. An assertion stops the run if any query study is in
  the index.
- The runner refuses an incomplete index. Smoke tests run on a partial index go to a separate
  folder (`data/runs/p2_adapted_smoke/`) and never mix with real results.
- Every response is cached in full: raw text, prompt, retrieved case, similarity, the model
  that actually answered, and token usage.

**Smoke test** (20-image index, 2 pilot studies; not results):
- Both calls succeeded, answered by `gemini-3-flash-preview`.
- Tokens per study: about 2,270 input, 490–610 output, 820–2,180 thinking. That is roughly
  $0.007 per study at paid prices, about $3 for the planned ~450 P2 calls.
- For CXR8 (ground truth: none of the six findings) the answer correctly reported no findings,
  addressing both images separately.

**Train index** (2026-10-17): 2,804 / 2,804 train images embedded (3,072-d, unit length, no
duplicates), $0.33. Gemini Embedding 2 also packs chest X-rays into a narrow similarity band
(median nearest-neighbour similarity 0.979).

**Pilot** (50 validation studies):

| | Original question | Checklist question (§3.5) |
|---|---|---|
| Answered | 50 / 50 | 50 / 50 |
| Checklist complete and at the top | — | 50 / 50 |
| Refusals / empty answers | 0 | 0 |
| Cost per study | ≈ $0.0091 (2,268 input / 495 output / 2,163 thinking tokens) | ≈ $0.0094 |

Top-1 retrieval similarity: 0.956–0.992 (median 0.981). Following its shipped prompt ("Be as
elaborate as possible giving extra relevant information"), P2 explains each finding at length,
including textbook definitions (38 / 50 answers name a finding in a definition sentence).

### 3.3 P1-adapted (from `ai_medical_imaging_agent`): pilot

Runner: `scripts/starters/p1_adapted.py`. Kept as shipped: the agno agent with Gemini
(`markdown=True`), the 5-section prompt verbatim, the width-500 resize. Adaptations: batch runner;
DuckDuckGo web search **off** (plan §3, reproducibility); the findings question appended;
`gemini-3-flash-preview` instead of the retired `gemini-2.5-pro` (§3.1).

**Finding: with web search off, P1 fails on most studies.** Section 5 of P1's verbatim prompt
says "IMPORTANT: Use the DuckDuckGo search tool to: …". With no tool available, the model still
tries to call one, and the API ends the response with `MALFORMED_FUNCTION_CALL`. agno then
retries once by itself with a "the previous function call was malformed" message; that fails
too.

| Pilot (50 validation studies) | Studies |
|---|---|
| Answer (the 5-section report) | 19 (38%) |
| `MALFORMED_FUNCTION_CALL` (no answer) | **31 (62%)** |

- This is how the starter behaves when its tool is removed: the prompt and the tool are coupled.
  The shipped app always had the tool. Its behaviour with the tool would have been shown by the
  thesis plan's 50-study web-search sensitivity arm (`--web-search`); that arm is not run
  (decision 2026-10-18, "Resolved decisions" #3).
- **Handling (follows the plan, no deviation):** a malformed call is recorded as the study's
  outcome, like a refusal (plan §3: "refusals are recorded, not prompted away"). **Scoring rule,
  fixed before any scoring:** it counts as "no findings reported" (all six negative). Leaving
  those studies out would overstate P1.
- **Cost:** a malformed call is billed for its input only (~1,500 tokens, ≈ $0.0007; 2 calls
  per study because of agno's retry). An answer costs ≈ $0.0067 (mean 1,457 input / 1,090 output
  / 987 thinking tokens). The pilot cost $0.18 in total.
- **Invented references (to check in the Q1 write-up):** answers still fill section 5 with
  "references" and URLs (e.g. Radiopaedia, StatPearls, ACR), although no search happened. They
  come from the model's memory and are unverified.

**New arm: P1-adapted-ns (decision 2026-10-17; deviation).** Because P1-adapted fails on most
studies, it is not run further; its pilot stays as the evidence above. It is replaced by
**P1-adapted-ns** ("no search"): the same runner, agent, model, image preparation and appended
findings question, with **section 5 ("Research Context") removed** from the prompt. Sections 1–4
and the closing format line stay verbatim (`--no-search-section`, folder `p1_adapted_ns`).
- Why the whole section, not just the "IMPORTANT: Use the DuckDuckGo search tool" line: the
  section's other bullets (find literature, give links, include 2–3 references) only make sense
  with search; without a tool they invite invented references.
- What the arm is: **image + a fixed structured prompt → answer**, with no tools, retrieval or
  clinical text. It is the image-only reference among the starters.
- Scoring risk noted at the time (plan §5): section 3 asks for differential diagnoses, which name
  conditions as possible rather than present, and CheXbert might read them as positive. This
  risk no longer applies: the starters are now scored on their own checklist (§3.5).
- The malformed-call scoring rule (counts as "no findings") still applies if one occurs.
- The thesis plan's 50-study web-search sensitivity arm for P1 as shipped is not run (decision
  2026-10-18, "Resolved decisions" #3).

| Pilot (same 50 validation studies) | P1-adapted (verbatim) | **P1-adapted-ns** |
|---|---|---|
| Answered | 19 (38%) | **50 (100%)** |
| `MALFORMED_FUNCTION_CALL` | 31 (62%) | **0** |
| API calls | 83 | 50 |
| Cost per study | $0.0036 (mostly failed, input-only calls) | $0.0062 |
| Web links in answers | invented from memory (no search) | none |

P1-adapted-ns answers have all four sections (median 2,939 characters; mean 1,390 input /
720 output / 1,110 thinking tokens). With the checklist question (§3.5): 50 / 50 checklists
complete and at the top; ≈ $0.0064 per study.

### 3.4 P3-orig and P3-adapted (from `multimodal_agentic_rag`): pilot

Runner: `scripts/starters/p3_adapted.py` (first named `p3_orig.py`), which imports and calls the
starter's own code (`rag_store`, `app_state.RAG_STORE`, `build_agent`). Its model
(`gemini-3-flash-preview`), embedder (`gemini-embedding-2`, 768 dimensions), image + annotation
blend (0.68 / 0.32), top-6 retrieval and agent (prompt, tools, temperature 0.25) are as shipped.
**P3-orig** also keeps the shipped `max_output_tokens=900`; **P3-adapted** removes it (below).

Adaptations (minimal, as planned):
1. Batch runner instead of the FastAPI server and React UI.
2. Retrieval by the patient's frontal-image embedding (the store's own image path) instead of
   the embedded text question. Scoring and ranking as in `search()`; the UI's 3-D projection
   is skipped (it does not affect the matches).
3. Index = the 2,804 train frontal images, each annotated with its report (FINDINGS +
   IMPRESSION), built as `add_file_source` builds entries (2 embedding calls per study; $0.37).
4. The agent's model gets the project's metered client (same model ID).

As shipped, the server retrieves first and gives the agent a retrieval tool that returns that
packet. **The generator never sees the patient's image, only the six retrieved reports.**

| Pilot (50 validation studies) | Result |
|---|---|
| Answered | 50 / 50 |
| Tool use | all 50: inspect workspace, then retrieve (3 calls per study; 1 study 2) |
| **Truncated answers** | **4 / 50 (8%)**: the 900-token output cap includes thinking on this model; those answers used 733–863 thinking tokens and stop mid-sentence |
| Cost | ≈ $0.0030 per study ($0.15 for the pilot) |

- Kept as shipped: truncated answers are scored as written; the truncation rate is a Q1
  result.
- Answers describe the **retrieved cases**, not the patient (e.g. CXR8: "The retrieved reports …
  indicate that none of the requested findings are present"). This is the design gap noted in
  the plan: P3 is retrieval plus a text model, not image analysis.

**P3-adapted (decision 2026-10-18; deviation).** With the checklist question (§3.5), P3-orig's
answering call thinks much more (median 861 tokens, before ≈ 0), and the 900-token cap, which
includes thinking, cut **45 / 50** answers inside the checklist. The cap is removed (the model's
default, as for P1 and P2); the arm is renamed **P3-adapted**. P3-orig's runs are kept.

| Pilot (50 validation studies, checklist question) | P3-orig (cap 900) | **P3-adapted** (no cap) |
|---|---|---|
| Answering call ended by the token cap | 45 / 50 | 0 / 50 |
| Checklist complete and at the top | 9 / 50 (44 entries missing) | **50 / 50** |
| Answering-call thinking (tokens) | ≤ 900 (cut) | median 1,959; p90 7,493; 3 studies ≈ 62,900 |
| Cost per study | ≈ $0.0045 | median $0.0087, **mean $0.021** |

- **Runaway thinking:** 3 / 50 studies (CXR1995, CXR3693, CXR2843) thought until the model's
  own limit (~62,900 tokens, ≈ $0.19 each); their answers are complete and normal. Not tied to
  case type in the pilot (2 of the 3 are normal studies). Rate 6% (95% CI 1.3–16.5%).
- Google documents no token budget for thinking on Gemini 3 (only `thinking_level`), and an
  output cap would cut answers again. P3-adapted keeps the default thinking level (decision
  2026-10-18); every run instead has a **USD 10 spend limit**.

**Cost tracking.** Every API call (including failed ones and calls a library makes internally)
goes through one metered client (`scripts/common.py`) that enforces each run's request budget
and **USD 10 spend limit** (a run stops cleanly once its logged cost reaches the limit; it can
overshoot by at most one call) and logs the call's token usage. `scripts/tools/spend.py` totals the
log against the current prices (monthly cap USD 45).

### 3.5 Scoring the starters: from CheXbert to a findings checklist

**Plan:** every starter's free text goes through CheXbert (plan §5). **What happened:** on the
pilot answers CheXbert produced implausible labels (P2 "called" Atelectasis in 45 / 50 studies).
Direct tests showed why: CheXbert was trained on prose radiology reports and misreads the
answers' format.

| Sentence from an answer | CheXbert | Meaning |
|---|---|---|
| "Cardiomegaly: Not present." | positive | negative |
| "Fracture: Not present." | negative | negative (inconsistent with the line above) |
| "Atelectasis: There are no signs of atelectasis, which is …" | positive | negative |
| "Cardiomegaly refers to an enlarged heart …" (definition) | positive | not about the patient |
| "Pleural Effusion was specifically ruled out …" | positive | negative |
| "None of the findings listed, Lung Opacity, Cardiomegaly, … are present." | Cardiomegaly positive | all negative |
| "No cardiomegaly." / "A subtle Lung Opacity is visible …" | correct | — |

The answers echo the six finding names because the findings question lists them. Rewrite rules
before CheXbert were considered and rejected: definitions and "ruled out" need judgements of
meaning, not form (38 / 50 P2 answers contain definition sentences). A second model reading the
answers was considered and not needed.

**Solution (deviation, 2026-10-17/18): the checklist question.** The starters' own prompts are
unchanged; the findings question they all receive (plan §3) gains one instruction:
*"Begin your answer with exactly six lines, one per finding, in the form "<finding>: present",
"<finding>: absent" or "<finding>: uncertain". Then continue as instructed."* Each system is
scored on its **own stated checklist** (`scripts/evaluation/parse_checklist.py`): present → 1; absent,
uncertain (plan §5), missing or conflicting → 0. Non-compliance is visible to the parser and
counted, unlike a labeller's silent misreading. The pilots were re-run with it; the earlier
answers and their CheXbert labels are kept as evidence.

| Pilot, checklist question | P1-adapted-ns | P2-adapted | P3-adapted |
|---|---|---|---|
| Checklist complete and at the top | 50 / 50 | 50 / 50 | 50 / 50 |
| Missing / conflicting entries | 0 / 0 | 0 / 0 | 0 / 0 |

**Hand check** (plan §5; 15 answers per starter, 8 with a "present" where available; first pass
by Claude, an AI assistant; sheets kept in `data/runs/review/`): does the checklist agree with the rest of
the same answer?

| | P1-adapted-ns | P2-adapted | P3-adapted |
|---|---|---|---|
| Agree | 89 / 90 | 90 / 90 | 89 / 90 |
| Unclear | 1 (Atelectasis given only as a differential, marked present) | 0 | 1 ("possible" atelectasis in one retrieved report, marked absent) |
| Disagree | 0 | 0 | 0 |

The checklists represent what the answers say. Two patterns matter for reading the scores:
- **Lung Opacity alongside Lung Lesion:** when the only abnormality is a nodule or mass, P1-ns
  and P2 often mark both. The ground truth codes nodules and masses as Lung Lesion only (§2.4b),
  so this shows up as Lung Opacity false positives (a definition mismatch).
- **P3 aggregates over its neighbours:** a finding is "present" if any of its six retrieved
  cases has it.

### 3.6 Q1 summary: how the starters work and how fit they are (pilot)

| | P1 (`ai_medical_imaging_agent`) | P2 (`vision_rag`) | P3 (`multimodal_agentic_rag`) |
|---|---|---|---|
| What reaches the model | the patient's image + a structured radiology prompt | the patient's image + the single most similar train image (top-1) | the **six most similar train cases' reports**; never the patient's image |
| Arm run | P1-adapted-ns | P2-adapted | P3-adapted |
| Changes needed to run it here | model retired (Pro → Flash); web search off; **section 5 removed** (the prompt ordered the missing tool: 62% malformed calls) | model retired; Cohere → Gemini Embedding 2; query by image; patient image added | output cap removed (the cap, incl. thinking, cut the checklist) |
| Behaviour on the pilot (50 studies, 10 positive findings) | 31 findings called, 5 correct; 34 / 42 normal studies called normal | **11 called, 3 correct; 40 / 42 normal called normal** (cautious) | **75 called, 6 correct; 16 / 42 normal called normal** (over-calls) |
| Failure modes seen | tool coupling (fixed by the new arm); references invented from memory (verbatim prompt); Lung Opacity + Lung Lesion counted together | long textbook explanations; Lung Opacity + Lung Lesion counted together | answers about the neighbours, not the patient; 8% truncated as shipped; runaway thinking (6%) without the cap |
| Cost per study (checklist) | ≈ $0.006 | ≈ $0.009 | ≈ $0.021 (mean; median $0.009) |

For comparison on the same 50 studies: S0 called 93 findings (6 correct, 12 / 42 normal),
R0 53 (5 correct, 19 / 42 normal); both are tuned for high sensitivity (Youden thresholds).

**Caveat:** the pilot has only 10 positive findings, so these are behaviours, not accuracy
estimates. Accuracy (sensitivity, specificity, F1 with bootstrap CIs) comes from the
positive-enriched test set in week 6. Full pilot table: `data/runs/pilot_summary_checklist.md`.

**Fitness, in short:** none of the starters ran on chest X-rays as shipped without at least one
change beyond the minimal adaptations the plan foresaw (a retired model, a tool-dependent
prompt, an output cap that includes thinking). P1 and P2 read the patient's image; P3 does not,
so its answer is effectively a vote over retrieved reports in prose. P2 is the most
conservative, P3 the most liberal.

## 4. Q2 — Reusable strengths and gaps; design of the built system (G)

Evidence: the starters' code (`../codebase_analysis.md`), their pilots (§3), the anchor systems
(§5.1, §5.2) and a retrieval comparison run for this question (§4.2). The design of G itself
follows in §4.4.

### 4.1 Component review

| From | Component | Verdict | Evidence |
|---|---|---|---|
| P1 | The model reads the **patient's image** directly | **Keep** | P1-ns and P2 answer about the patient's film; P3, which never sees it, answers about its neighbours and over-calls (75 calls, 16 / 42 normal studies called normal; §3.6) |
| P1 | Structured, systematic radiology prompt (technique, findings, assessment) | **Keep the idea, adapt** | Gives systematic answers (§3.3). Its patient-friendly section and free-text "confidence level" add nothing measurable; differentials blur present vs. possible (hand check, §3.5) |
| P1 | agno agent with a tool | **Drop** | Hidden internal retries (one attempt = 2 billed calls); tool coupling made 62% of answers fail (§3.3). A direct API call is simpler and fully metered |
| P1 | Web search for references | **Drop** | Not reproducible (plan §3); without it, references are invented from memory (§3.3) |
| P2 | Embed → retrieve → generate, with the patient's image and the evidence in one call | **Keep** | The only starter pattern in which the model sees the patient and the retrieved evidence together (with adaptation 4, §3.2) |
| P2 | Top-1 retrieval | **Upgrade to top-k** | One neighbour carries little signal: k = 1 macro AUROC 0.572, k = 5 0.651 (R0, §5.2) |
| P2 | General-purpose embedder (Cohere; here Gemini Embedding 2) | **Replace by BiomedCLIP** | At k = 5, −0.077 [−0.100, −0.054] macro AUROC vs. BiomedCLIP (§4.2) |
| P2 | Retrieved **image** passed without its report | **Gap** | The retrieved image carries no label or text; the model must reinterpret another patient's film. The neighbour's report is the informative part (P3) |
| P2 | "Be as elaborate as possible" | **Drop** | Long textbook definitions in 38 / 50 answers (§3.2); cost without content |
| P3 | Retrieve first, then hand the packet to the model (citations = what the answer used) | **Keep** | The server's design guarantees the cited cases are the ones the answer saw (`server.py`, §3.4) |
| P3 | Neighbours' **report text** as evidence | **Keep** | Carries the labels and wording the image alone lacks |
| P3 | Structured citations with similarity scores | **Keep, make per finding** | Strongest explainability element of the three (codebase analysis); per-finding citations make them checkable against the cited cases' labels (plan §5) |
| P3 | Instruction to flag weak evidence | **Keep** | Cheap; neighbours are weak evidence at small k (§5.2) |
| P3 | Agent loop with an "inspect workspace" tool | **Drop** | 3 calls per study, the inspect step adds nothing; runaway thinking in 6% of studies (§3.4) |
| P3 | Patient never reaches the model; "present" if any neighbour has it | **Gap (the main one)** | Answers describe the neighbours, not the patient (§3.4, §3.5) |
| P3 | Image + report blend in the index | **Not needed** | Helps the Gemini embedder (+0.019 at k = 5) but stays below BiomedCLIP on images alone (§4.2) |
| P3 | Output cap that includes thinking | **Drop** | Truncated 8% (original question) and 90% (checklist) of answers (§3.4) |

### 4.2 Retrieval comparison (train, leave-one-out; no API calls)

R0's kNN vote with five embeddings of the same train images (`scripts/evaluation/q2_retrieval_compare.py`,
report `data/runs/q2/retrieval_compare.md`):

| Macro AUROC | k = 1 | k = 5 | k = 10 | k = 50 | k = 150 |
|---|---|---|---|---|---|
| **BiomedCLIP** (centre crop; R0's index) | 0.572 | **0.651** | 0.671 | 0.712 | 0.741 |
| BiomedCLIP, padded to square | 0.584 | 0.655 | 0.680 | 0.732 | 0.731 |
| Gemini Embedding 2, 3,072-d image (P2's index) | 0.529 | 0.574 | 0.600 | 0.667 | 0.720 |
| Gemini Embedding 2, 768-d image (P3's index) | 0.531 | 0.576 | 0.601 | 0.664 | 0.712 |
| P3's image + report blend, image query | 0.550 | 0.595 | 0.635 | 0.703 | 0.737 |

- At the number of cases a prompt can hold, **BiomedCLIP retrieves clearly more informative
  neighbours** than the starters' embedder (k = 5: −0.077 [−0.100, −0.054] for Gemini 3,072-d).
- **Crop vs. pad makes no measurable difference for retrieval** (k = 5: +0.004 [−0.012, +0.020];
  k = 150: −0.010 [−0.031, +0.013]), unlike for S0 (§5.1). There is no reason to change the index.
- At large k the embedders converge (k = 150: all within 0.03); small-k retrieval is where they
  differ, and small-k retrieval is weak for every embedder.

### 4.3 Gaps none of the starters fills

1. **Clinical text** (the indication) as an input: none takes it (plan §4; G vs. G−ind measures it).
2. **The patient's image and textual evidence together:** P2 gives image + image, P3 text only.
3. **A per-finding answer with a score:** the checklist gives present / absent / uncertain (§3.5),
   but no continuous score, which AUROC needs (plan §5).
4. **Finding definitions aligned with the ground truth:** P1-ns and P2 count a nodule or mass as
   Lung Opacity too, while the ground truth codes it as Lung Lesion only (§3.5).
5. **Machine-checkable citations per finding** (citation label agreement, plan §5).
6. **Cost control by design:** a fixed thinking level instead of an output cap; one call per study.
7. Calibration, claim-level verification and de-identification (codebase analysis) remain out of
   scope this semester (plan §1).

### 4.4 Design of G (decided 2026-10-18)

Full design with options and evidence: `g_design.md`. Summary:

- **Pipeline:** the patient's frontal image → BiomedCLIP (R0's index: train split, centre crop;
  validation / test queries already embedded) → the **10** most similar train studies →
  **one direct Gemini call** with the patient's image, the clinical indication and the 10
  neighbours' reports (FINDINGS + IMPRESSION, with case ID and similarity; not their images).
  No agent, no tools, no web search, no output cap. Model `gemini-3-flash-preview` (as the
  starters), default temperature (1.0, Google's recommendation for Gemini 3) and default
  thinking level (high, as the starters).
- **Prompt:** decision-support framing; the six findings **defined as in the ground truth**
  (§2.4b: e.g. a nodule or mass is Lung Lesion, not Lung Opacity; borderline heart size is not
  Cardiomegaly; healed fractures count); read the patient's image first; the neighbours are
  other patients, weak evidence to support or question the reading and to cite, not the answer;
  short rationales. Developed on the 50 pilot studies (at most 3 versions), then frozen before
  any validation run.
- **Output:** JSON enforced by the API's response schema: per finding a status (present /
  absent / uncertain), a probability 0–1, cited case IDs and a short rationale; plus an
  impression. Status → binary with the starters' rule (§3.5); probability → AUROC and the pooled
  Youden threshold tuned on validation (§2.5).
- **Prompt development (pilot, 50 studies):** prompt 1 gave valid JSON for all 50 studies but
  cited neighbours by list position in 3 answers, and its probabilities were coarse (66% exactly
  0.05). **Prompt 2** lists cases by ID and asks for citation by ID and two-decimal probabilities:
  0 invalid citations, 19 instead of 13 distinct probability values (most common 57%), binary
  calls 296 / 300 the same as prompt 1 (within run-to-run noise at temperature 1.0). **Prompt 2
  is frozen** (2026-10-18); the runner refuses other versions outside pilot runs. Pilot
  behaviour: 10 findings called, 4 correct, 39 / 42 normal studies called normal; ≈ $0.0105 per
  study, no runaway thinking (max 6,943 tokens). Details: `g_design.md` §D.
- **Arms:** **G** (as above), **G−ind** (identical except the indication line and the one step that
  refers to it are omitted),
  **G-img** (G's model, prompt and output with the image only: no indication, no neighbours;
  test set only, scored by status and AUROC). G+S0 is not run.
- **Cost estimate:** G family ≈ $19; project ≈ $37.5 of the USD 45 cap, re-projected after G's
  first pilot.

## 5. Q3 — Comparison

§5.1–5.3 set up the anchor systems (S0, R0) on train and the built system's threshold on
validation; §5.4 fixes the evaluation method; §5.5 gives the test-set results (week 6, run once)
and §5.6 the week-7 analyses. Every script that can touch the test split refuses it unless
`--test` is given.

**Rules shared by all tuned systems** (fixed 2026-10-17, before any scoring):
- Metrics code: `scripts/metrics.py` (rank-based AUROC with ties, threshold metrics, paired
  bootstrap over studies, 2,000 resamples), tested against brute force
  (`scripts/tests/test_metrics.py`).
- Decision threshold per finding = **Youden's J** (max sensitivity + specificity − 1); ties go to
  the higher threshold. Reason: the test set is positive-enriched, and Youden's J, like
  sensitivity and specificity, does not depend on prevalence. An F1-optimal threshold tuned at
  train prevalence would be too conservative for the test set. G's pooled threshold will use
  the same criterion.
- F1 below is at natural prevalence (train, validation) and is low for rare findings by
  construction; it is not comparable to test-set F1 (plan §2).

### 5.1 S0 — specialist image model (TorchXRayVision, no training)

**Weights loading.** The library loads weights with `torch.load(weights_only=False)`, which can
run code stored in the file. `scripts/anchors/s0_weights.py` instead uses a restricted unpickler that
allows only the 14 classes and functions the files were found to use (every pickle opcode was
scanned without executing anything). Its logits are identical to the library's (max difference
0.0). Weight files and SHA-256 hashes are in the diary (2026-10-17).

**Preprocessing** (`scripts/anchors/s0_score.py`): the library's documented pipeline (channel mean,
scale to [−1024, 1024], make square, resize to 224 or 512). The documented "make square" step is
a centre crop. On IU it keeps a median 82% of the long side (2,109 of 3,690 frontal films are
landscape), cutting off the lateral chest walls or the apices and bases. **Pad-to-square** (black
borders, nothing cut) was therefore scored as an alternative. Inference is deterministic
(TF32 off): scores reproduce across batch sizes and GPU / CPU to within 4 × 10⁻⁷.

**Selection on train** (S0 never saw IU data, so train is unseen data for it). Rule fixed
beforehand: among the three DenseNets × {crop, pad}, the highest macro AUROC over the six
findings. The ResNet-50 is reported only (§2.3b, and plan §3 names a DenseNet).

| Weights | Prep | Lung Opacity | Cardiomegaly | Atelectasis | Lung Lesion | Pleural Effusion | Fracture | Macro | Δ macro vs. selected [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| DenseNet chex | crop | 0.714 | 0.807 | 0.721 | 0.490 | 0.822 | 0.592 | 0.691 | −0.037 [−0.059, −0.014] |
| DenseNet chex | pad | 0.696 | 0.809 | 0.706 | 0.511 | 0.820 | 0.558 | 0.683 | −0.044 [−0.066, −0.023] |
| DenseNet mimic_nb | crop | 0.743 | 0.789 | 0.733 | 0.557 | 0.842 | 0.616 | 0.713 | −0.014 [−0.026, −0.002] |
| **DenseNet mimic_nb** | **pad** | 0.743 | 0.798 | 0.734 | 0.547 | 0.868 | 0.676 | **0.728** | selected |
| DenseNet mimic_ch | crop | 0.712 | 0.784 | 0.742 | 0.554 | 0.832 | 0.639 | 0.711 | −0.017 [−0.033, −0.001] |
| DenseNet mimic_ch | pad | 0.705 | 0.779 | 0.734 | 0.552 | 0.860 | 0.639 | 0.712 | −0.016 [−0.034, +0.003] |
| ResNet-50 res512 (reference) | crop | 0.723 | 0.894 | 0.811 | 0.689* | 0.939 | 0.715 | 0.795 | +0.067 [+0.040, +0.093] |
| ResNet-50 res512 (reference) | pad | 0.730 | 0.913 | 0.819 | 0.697* | 0.943 | 0.746 | 0.808 | +0.080 [+0.052, +0.106] |

\* Lung Lesion = max(Nodule, Mass) (proxy; the model's own Lung Lesion output is untrained).

Findings:
- **Selected: DenseNet `mimic_nb`, padded.** Its lead over the next candidates is small but
  outside the bootstrap noise for one of them (−0.014 [−0.026, −0.002]) and borderline for the
  other (−0.016 [−0.034, +0.003]).
- **Padding helps where the crop cuts anatomy:** Pleural Effusion (costophrenic angles) and
  Fracture (lateral ribs) gain with padding for mimic_nb (0.842 → 0.868, 0.616 → 0.676) and for
  the ResNet (0.939 → 0.943, 0.715 → 0.746).
- **Lung Lesion is near chance for every DenseNet** (0.49–0.56). The cause is not established,
  but it is **not a difference in definition** (checked 2026-10-31; this corrects the earlier
  explanation, see "Corrections"). S0's training labels (`mimic_nb`) were extracted from
  MIMIC-CXR reports by NegBio using the CheXpert labeller's mention phrases: Lung Lesion = mass,
  nodule, nodular density / opacity / opacification, lump, cavitary lesion, carcinoma, neoplasm or
  tumor, with "calcified nodule" mentions excluded. That is essentially this project's ground
  truth (nodule, mass or cavity in the lung, calcified granulomas excluded, §2.4b). The remaining
  differences are in how the labels were made (keyword rules on the IMPRESSION section vs. human
  coding of the report; uncertain mentions ignored in training), not in what counts as a lesion.
  Possible reasons for the weakness are noisy text-mined training labels, small lesions at
  224 px and poor transfer between hospitals; which of these applies is not known.
- **The ResNet-50 reference is clearly better than every DenseNet** (+0.080 macro AUROC
  [+0.052, +0.106]), most of all on Cardiomegaly, Atelectasis, Lung Lesion (via the proxy) and
  Pleural Effusion. See "Resolved decisions": the DenseNet is kept.

**Selected S0 (DenseNet mimic_nb, pad): thresholds tuned on train (Youden), results:**

| Finding | Train positives | Train AUROC [95% CI] | Threshold | Val positives | Val AUROC [95% CI] | Val sensitivity | Val specificity |
|---|---|---|---|---|---|---|---|
| Lung Opacity | 396 | 0.743 [0.715, 0.770] | 0.2108 | 39 | 0.716 [0.615, 0.806] | 0.51 | 0.81 |
| Cardiomegaly | 206 | 0.798 [0.767, 0.826] | 0.1509 | 23 | 0.868 [0.790, 0.929] | 0.87 | 0.70 |
| Atelectasis | 221 | 0.734 [0.702, 0.765] | 0.1373 | 26 | 0.672 [0.545, 0.783] | 0.69 | 0.60 |
| Lung Lesion | 49 | 0.547 [0.465, 0.627] | 0.0072 | 11 | 0.482 [0.320, 0.653] | 0.64 | 0.40 |
| Pleural Effusion | 83 | 0.868 [0.827, 0.907] | 0.2262 | 12 | 0.899 [0.843, 0.951] | 0.92 | 0.81 |
| Fracture | 43 | 0.676 [0.600, 0.748] | 0.0078 | 2 | 0.906 [0.847, 0.955] | 1.00 | 0.76 |
| **Macro** | | **0.728** [0.706, 0.749] | | | **0.757** [0.695, 0.802] | 0.77 | 0.68 |

Validation has very few positives for the rare findings (Fracture 2, Lung Lesion 11), so their
validation numbers say little; the CI for Fracture is narrow only because it rests on two cases.
On near-chance scores, Youden's J picks degenerate thresholds (Lung Lesion: sensitivity 0.71,
specificity 0.38 on train). Full report: `data/runs/s0/selection_report.md`; choice and
thresholds: `data/runs/s0/selection.json`.

### 5.2 R0 — retrieval alone (kNN vote, no LLM)

`scripts/anchors/r0_knn.py`. Index = the train split's BiomedCLIP frontal-image embeddings (the index G
will use). Score per finding = the share of the k nearest training studies (cosine similarity)
that have it, unweighted (similarities sit in a narrow band, §2.3). One k for all findings,
chosen by leave-one-out on train (each train study scored from the other 2,803) = highest macro
AUROC; per-finding Youden thresholds on the leave-one-out scores. Leakage assertions: no study
retrieves itself; no validation study is in the index.

**Choice of k** (leave-one-out, train). The first grid ended at k = 100, which was the best value
with AUROC still rising, so the grid was extended (train only):

| k | 1 | 3 | 5 | 10 | 20 | 30 | 50 | 75 | 100 | **150** | 200 | 300 | 500 | 750 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Macro AUROC | 0.572 | 0.626 | 0.651 | 0.671 | 0.704 | 0.712 | 0.712 | 0.732 | 0.741 | **0.741** | 0.736 | 0.740 | 0.736 | 0.739 | 0.739 |

**k = 150** (0.7413 vs. 0.7405 at k = 100). Performance plateaus from k ≈ 100 to 1,000, so the
exact k hardly matters. More importantly, **R0's signal is the prevalence of a finding in a
broad neighbourhood, not a few close matches**: the single nearest case gives 0.572 and five
cases 0.651.

| Finding | Train positives | Train AUROC (LOO) [95% CI] | Threshold (vote share) | Val positives | Val AUROC [95% CI] | Val sensitivity | Val specificity |
|---|---|---|---|---|---|---|---|
| Lung Opacity | 396 | 0.792 [0.766, 0.816] | 0.100 | 39 | 0.721 [0.630, 0.806] | 0.54 | 0.75 |
| Cardiomegaly | 206 | 0.891 [0.868, 0.911] | 0.053 | 23 | 0.959 [0.929, 0.982] | 0.96 | 0.80 |
| Atelectasis | 221 | 0.747 [0.711, 0.783] | 0.080 | 26 | 0.694 [0.578, 0.802] | 0.46 | 0.84 |
| Lung Lesion | 49 | 0.555 [0.475, 0.633] | 0.013 | 11 | 0.435 [0.256, 0.615] | 0.55 | 0.44 |
| Pleural Effusion | 83 | 0.898 [0.854, 0.935] | 0.020 | 12 | 0.896 [0.760, 0.985] | 0.75 | 0.90 |
| Fracture | 43 | 0.565 [0.477, 0.647] | 0.013 | 2 | 0.496 [0.236, 0.755] | 0.50 | 0.40 |
| **Macro** | | **0.741** [0.718, 0.763] | | | **0.700** [0.624, 0.777] | 0.63 | 0.69 |

- Retrieval works for the large, global findings (Cardiomegaly, Pleural Effusion, Lung Opacity)
  and is **near chance at every k for Lung Lesion and Fracture**: a whole-image embedding does
  not capture small focal findings (and the train index has only 49 and 43 positives, §2.5).
- On train, retrieval alone (0.741) already matches the selected zero-shot DenseNet (0.728),
  mainly through Cardiomegaly (0.891 vs. 0.798). R0 uses IU's own labels through its
  neighbours; S0 transfers from other hospitals' data.
- Outputs: `data/runs/r0/train.csv` (leave-one-out) and `val.csv`, each with the 150 neighbour
  IDs per study for the week-7 retrieval-quality and copying-rate analyses;
  `data/runs/r0/selection.json`; report `data/runs/r0/selection_report.md`.

### 5.3 G and G−ind on validation (pooled threshold)

`scripts/g/g_system.py` (frozen prompt 2; the 50 pilot answers reused for G), `g_export.py`,
`g_select.py`. One threshold per arm, pooled across the six findings: Youden's J over all 2,400
(study, finding) pairs of the stated probabilities (113 positive pairs), ties → higher (§2.5,
rule details fixed before the runs, diary 2026-10-18). Both binary readings are reported: the
probability at the threshold, and the stated status ("present", the starters' rule, §3.5).

**Runs:** both arms 400 / 400 valid JSON; no validation problems (all citations valid), no missing
statuses or probabilities. Thinking tokens per answer: G median 1,979, max 7,903; G−ind median
1,967, max 8,263 (no runaways, unlike P3-adapted, §3.4). Cost: G $3.44 (350 new calls), G−ind
$4.12 (400 calls), ≈ $0.010 per study.

| Finding | Val positives | G AUROC [95% CI] | G−ind AUROC [95% CI] | G sens. / spec. @thr | G sens. / spec. stated | G−ind sens. / spec. @thr | G−ind sens. / spec. stated |
|---|---|---|---|---|---|---|---|
| Lung Opacity | 39 | 0.703 [0.611, 0.798] | 0.701 [0.611, 0.794] | 0.51 / 0.86 | 0.31 / 0.95 | 0.46 / 0.87 | 0.28 / 0.96 |
| Cardiomegaly | 23 | 0.932 [0.878, 0.971] | 0.916 [0.856, 0.965] | 0.96 / 0.76 | 0.48 / 0.97 | 0.91 / 0.80 | 0.48 / 0.97 |
| Atelectasis | 26 | 0.678 [0.546, 0.809] | 0.689 [0.557, 0.820] | 0.58 / 0.64 | 0.50 / 0.92 | 0.65 / 0.69 | 0.42 / 0.93 |
| Lung Lesion | 11 | 0.700 [0.542, 0.855] | 0.622 [0.448, 0.801] | 0.45 / 0.86 | 0.18 / 0.98 | 0.36 / 0.86 | 0.09 / 0.98 |
| Pleural Effusion | 12 | 0.893 [0.796, 0.980] | 0.840 [0.691, 0.959] | 0.75 / 0.91 | 0.33 / 1.00 | 0.58 / 0.92 | 0.33 / 0.99 |
| Fracture | 2 | 0.713 [0.489, 0.938] | 0.573 [0.550, 0.597] | 0.50 / 0.88 | 0.00 / 0.99 | 0.00 / 0.95 | 0.00 / 0.99 |
| **Macro** | | **0.770** [0.701, 0.837] | **0.724** [0.678, 0.782] | 0.63 / 0.82 | 0.30 / 0.97 | 0.50 / 0.85 | 0.27 / 0.97 |

**Pooled threshold: 0.08 for both arms** (G: pooled sensitivity 0.64, specificity 0.82; G−ind:
0.59, 0.85). Prompt 2 asks for two-decimal probabilities, so the candidate thresholds are coarse.

The trade-off at other cut-offs (G, validation, macro over the six findings; computed
2026-10-18 for the presentation, from `data/runs/g__prompt2/val_checklist.csv`; not used for
any choice):

| Rule | Macro sensitivity | Macro specificity |
|---|---|---|
| G's own "present" | 0.30 | 0.97 |
| probability ≥ 0.05 | 0.94 | 0.22 |
| **probability ≥ 0.08 (chosen)** | **0.63** | **0.82** |
| probability ≥ 0.10 | 0.61 | 0.83 |
| probability ≥ 0.15 | 0.43 | 0.92 |
| probability ≥ 0.30 | 0.36 | 0.95 |
| probability ≥ 0.50 | 0.30 | 0.97 |

G's own "present" behaves like a cut-off near 0.5. At 0.05 specificity collapses, because
1,399 of the 2,400 answers are exactly 0.05. (Macro values differ slightly from the pooled ones
above, which are the basis of the choice.)

- **The stated status is very conservative:** "present" catches 30% (G) / 27% (G−ind) of
  positives at specificity 0.97. The probability at 0.08 roughly doubles sensitivity at
  specificity ≈ 0.82–0.85. The two readings disagree on 377 (G) / 307 (G−ind) of 2,400 pairs.
  Both are carried to test (plan §5: binary metrics comparable with the starters; AUROC and
  threshold metrics comparable with S0 and R0).
- **For orientation only (validation; the planned comparison is on test):** macro AUROC G 0.770,
  S0 0.757 (§5.1), G−ind 0.724, R0 0.700 (§5.2); the confidence intervals overlap widely. On Lung
  Lesion (11 positives), G reaches 0.700 [0.542, 0.855] where S0 (0.482) and R0 (0.435) are near
  chance. With 2 positives, Fracture's numbers carry no information.
- G vs. G−ind (indication line removed): G is higher on 5 of 6 findings (Lung Opacity equal
  within 0.002; Atelectasis lower), with the largest gaps on Lung Lesion and Pleural Effusion. No paired test on validation; the G vs. G−ind difference
  is a planned test-set comparison (plan §5).
- Outputs: `data/runs/g__prompt2/` and `data/runs/g_ind__prompt2/`: `val_checklist.csv`,
  `selection.json` (threshold + validation results), `selection_report.md`.

**G-img pilot** (50 pilot studies, frozen prompt 2, image only; g_design.md D7: test-only arm,
no threshold, so no validation run). 50 / 50 valid JSON, no validation problems, no missing
statuses or probabilities, **no case IDs cited** (none were given). Thinking median 1,180,
max 7,179. Cost $0.32. Descriptive comparison with G on the same 50 studies (10 positive
pairs, so not a performance estimate):

| | "Present" calls (correct / false) | "Uncertain" | Mean probability, positives / negatives | Present per finding (LO, CM, AT, LL, PE, FR) |
|---|---|---|---|---|
| G-img | 16 (3 / 13) | 1 | 0.315 / 0.087 | 4, 1, 7, 2, 1, 1 |
| G | 10 (4 / 6) | 3 | 0.353 / 0.067 | 2, 2, 3, 0, 1, 2 |

The stated statuses agree on 286 of 300 pairs. Without cases, the model calls "present" more
often, mainly Atelectasis (7 vs. 3), and gives negatives higher probabilities. The schema and
behaviour are fit for the test run.

### 5.4 Evaluation method for the test set (decided 2026-10-18)

Full proposal with option tables: `eval_design.md`. Plan §5 fixes *what* is measured (per-finding
and macro F1, sensitivity and specificity for all systems; AUROC where a system gives scores;
precision@k for the retrieval systems; G's copying rate; G's citation label agreement; G vs.
G−ind; paired bootstrap CIs over studies). It leaves five details open. They are fixed here,
before any test-set result exists, so that the test analysis cannot be shaped by its own
results. All five use only cached outputs; none needs an API call.

**Already fixed by earlier decisions (restated, not new):** ground truth = MeSH-derived labels
(§2.4b). Binary reading: each starter's checklist and G-family's stated status, "present" → 1
and absent / uncertain / missing → 0 (§3.5); S0 and R0 at their train-tuned per-finding
thresholds (§5.1, §5.2); G and G−ind additionally at their pooled validation threshold 0.08
(§5.3). AUROC for S0, R0, G, G−ind and G-img (G-img has no threshold, as it has no validation run;
g_design.md D7). Macro = unweighted mean over the six findings. Bootstrap: 2,000 study
resamples, seed 2026, paired, percentile 95% CI. Unanswered or unparsable outputs are counted and
scored as all-negative, so every system is scored on the same 400 test studies.

#### E1 — Which differences get confidence intervals

*The question.* Plan §5 asks for "paired bootstrap confidence intervals … for the differences"
but does not say which differences. With eight systems (S0, R0, P1-adapted-ns, P2-adapted,
P3-adapted, G, G−ind, G-img), two binary readings for G and G−ind, and several metrics, there are
dozens of possible pairs.

*Options considered.* (a) A fixed, pre-declared list with one primary comparison; (b) every pair
of systems on every metric; (c) per-system CIs only, no CIs on differences.

**Decision: (a).** The comparisons, each a difference "first − second" in the macro metric:

| # | Comparison | Metric(s) | Question it answers |
|---|---|---|---|
| **1 (primary)** | **G − S0** | macro AUROC | Does image + retrieval + clinical text beat the specialist image model? |
| 2 | G − R0 | macro AUROC | Does the LLM add anything over retrieval alone (same index, same neighbours)? |
| 3 | G − G−ind | macro AUROC; macro F1 (stated) | What does the clinical indication add? (plan §5) |
| 4 | G − G-img | macro AUROC; macro F1 (stated) | What do retrieval + indication add, with model, prompt and output format held constant? |
| 5 | G-img − S0 | macro AUROC | A general LLM reading the image alone vs. the specialist reading the image alone |
| 6–8 | G − P1-adapted-ns, G − P2-adapted, G − P3-adapted | macro F1, sensitivity, specificity (stated) | The built system vs. each starter (Q3) |

*Reasoning.*
- The thesis question (plan §1, thesis_plan.md) is whether a multimodal / RAG system beats a
  single image-recognition model. G − S0 on macro AUROC is the direct test of that: both give a
  continuous score for every finding, so AUROC compares their ranking ability without depending
  on either system's threshold. Naming one primary comparison before the test run means the
  answer to the thesis question is fixed in advance and cannot be chosen after seeing which
  comparison came out best.
- Comparisons 2–5 are the steps that explain the primary result: whether the LLM adds anything
  beyond its neighbours (2), what the text adds (3), what retrieval + text add within one design
  (4), and whether a general LLM can read an X-ray as well as a specialist at all (5). Together
  they separate the parts of G's design that plan §5 says Q3 cannot attribute on its own.
- Comparisons 6–8 answer Q3 for the starters. The starters give no scores, only verdicts, so
  F1, sensitivity and specificity of the stated verdicts are the only common metrics; G's stated
  status is used (not its threshold reading) so that both sides are scored by the same rule.
- Option (b) would produce dozens of intervals; at a 95% level some would exclude zero by chance
  alone, and the write-up would have to explain which ones mattered after the fact. Option (c)
  would not answer "is G better than X", since two overlapping per-system intervals do not show
  that a paired difference is zero or not.
- No multiple-comparison correction is applied. Instead, only #1 is called the main result;
  #2–#8 are reported as secondary and interpreted with that caution. Per-finding differences are
  shown in tables with their CIs but are not singled out as findings.

#### E2 — Retrieval quality (precision@k)

*The question.* Plan §5 defines precision@k as "share of retrieved training cases that share the
query's labels", for P2, P3, R0 and G. The definition is ambiguous with six rare findings: most
studies are normal, so "shares the query's labels" is dominated by normal studies retrieving
normal studies. A retriever that only ever returns normal cases would score well on normal
queries and the metric would barely notice whether it finds cases with the same *disease*.

*Options considered.* (a) Per finding, on queries that have the finding, with prevalence as
baseline; (b) exact match of the full six-label vector; (c) Jaccard overlap of label sets.

**Decision: (a).** For each finding f and each test study that has f: the share of its k
retrieved cases that also have f. This is averaged per finding and shown next to f's prevalence
in the train split, which is the expected share if neighbours were drawn at random; the ratio
of the two is reported as lift (1 = no better than random). A separate row gives, for normal
test studies, the share of their neighbours that are normal.

*Reasoning.*
- The retrieval systems exist to put relevant evidence in front of the LLM (or, for R0, to vote).
  Relevant evidence for a patient with an effusion means cases that *also* have an effusion.
  Option (a) measures exactly that, finding by finding.
- Raw precision is uninterpretable without a baseline: 0.15 would be excellent for Fracture
  (train prevalence about 1.5%) and poor for a common finding. Lift relative to prevalence makes
  findings comparable.
- Option (b) is the literal reading of the plan, but almost all exact matches are normal–normal
  pairs, so it mostly measures how often normals retrieve normals. Option (c) is undefined when
  both label sets are empty (normal–normal) and is hard to explain in a report.
- Normal queries still get a row of their own, so the "normals retrieve normals" behaviour is
  shown rather than hidden.

**Depth k.** Each system at its own k (P2: 1, P3: 6, G: 10, R0: 150), and all four at **k = 1**.
k = 1 is the only depth every system has (P2 retrieves a single case), so it is the only
like-for-like comparison of the retrievers themselves. G and R0 search the same BiomedCLIP index,
so R0's first 10 neighbours are exactly G's 10; they are reported once, under G.

This reads the plan's wording more narrowly than written; it is listed under deviations.

#### E3 — Copying rate

*The question.* Plan §5: "How often G's answer equals the neighbours' majority vote, especially
when that vote is wrong", G vs. R0. The point is to find out whether G actually reads the
patient's image or largely repeats what its 10 retrieved cases say. Two practical problems:
(1) a strict majority (at least 6 of 10 neighbours with the finding) is almost never reached for
these findings: R0's tuned thresholds are vote shares of only 0.013–0.100 (§5.2), so the
"majority vote" would be "absent" for nearly every study, and G agreeing with it would just
measure G's specificity; (2) agreement alone cannot tell copying from coincidence: G and its
neighbours may agree simply because both reflect the same image.

*Options considered.* (a) The neighbours' decision = R0's rule applied at k = 10, with G-img as a
control; (b) strict majority vote (≥ 6 of 10); (c) "any neighbour has the finding".

**Decision: (a).**
- The neighbours' decision for each finding = the share of G's 10 neighbours that have it,
  compared with a per-finding Youden threshold tuned by leave-one-out on train at k = 10 (the
  same procedure that set R0's thresholds at k = 150; train only, no API).
- Reported: how often G's binary verdict equals that decision, over all studies and separately
  where the decision is wrong (the plan's "especially when that vote is wrong").
- **The same numbers for G-img.** G-img uses G's model, prompt and output format but never sees
  the neighbours. Its agreement with the neighbours' decision is the agreement expected from
  reading the same image, without any influence from the cases. **G's agreement minus G-img's
  agreement estimates how much the neighbours pull G**, above all where they are wrong.
- Descriptive addition: the rate of "present" verdicts by number of positive neighbours
  (0, 1, 2, 3 or more), for G and G-img side by side.

*Reasoning.*
- Option (b) is the plan's wording, but on these data it is nearly constant ("absent"), so it
  carries almost no information. Option (c) fires for most studies on the common findings
  (Lung Opacity: about 14% of train studies are positive, so most sets of 10 contain one).
- Option (a) gives the neighbours a decision rule that is tuned to be informative, by the same
  method already used for R0, so it is not a new invention. The threshold is set on train, so
  the test set plays no part in it.
- The G-img control is what turns agreement into a measure of influence. Without it, high
  agreement could mean G copies or that the image is simply clear. G-img was added to the design
  (g_design.md D7) partly for this kind of contrast.
- The neighbours' vote uses G's k = 10, not R0's k = 150, as noted in "Observations" below: G is
  influenced by the 10 cases it sees, not by 150.

This changes the plan's operational definition ("majority vote"); it is listed under deviations.

#### E4 — Citation label agreement

*The question.* Plan §5: "Share of G's cited cases that actually have the finding G claims".
G cites case IDs for each finding. When G says "present", do the cases it points to as evidence
actually have that finding according to the ground truth? This checks whether G's explanations
are verifiable, the original motivation for grounding.

*Options considered.* (a) The plan's share, plus a baseline from all 10 neighbours; (b) the
plan's share alone.

**Decision: (a).** For every "present" verdict of G (and of G−ind), the share of its cited cases
that have that finding, pooled per finding. Baseline: the share of *all* 10 neighbours given to
G for those same studies that have the finding. Also counted: "present" verdicts with no
citation, and "absent" or "uncertain" verdicts with citations (counted, not scored).

*Reasoning.*
- Without a baseline the number cannot be read. If 40% of G's cited cases have the finding, that
  is selective citation if only 10% of its neighbours had it, and no selection at all if 40% did.
  The baseline separates "G picks the relevant cases" from "G cites whatever it was given".
- "Present" calls without any citation are reported because a claim with no evidence is itself
  an explanation failure.
- Citations under "absent" have no clear meaning to check against ground truth (a case cited as
  "showing no effusion" is correct whenever the case is normal), so they are counted only.

#### E5 — Implementation and checks

**Decision:** one script, `scripts/evaluation/evaluate.py <split> [--test]`, reading every
system's existing tables and cached answers (no API) and writing `data/runs/eval/<split>/report.md`
(all tables) and `results.json`. Offline tests in `scripts/tests/test_evaluate.py` on small
synthetic data with answers computed by hand: precision@k, copying rate, citation agreement and
the alignment of studies across systems.

*Reasoning.*
- One script means the whole Q3 analysis is reproduced by one command, and the study alignment
  across eight systems (the most likely source of silent errors) is written once.
- Tests with hand-computed answers check the new metrics before they touch real data, as was done
  for `metrics.py` and the checklist parser.
- The script is run on validation first: S0, R0, G and G−ind on all 400 studies, starters and
  G-img on the 50 pilot studies, systems compared on their common studies. This checks it end to
  end; those numbers are not results. On test the script asserts that every system has all 400
  studies, and it refuses the test split without `--test` (plan §6: once).

**Implemented and checked (2026-10-18).** `scripts/evaluation/evaluate.py`; 18 offline tests
(`scripts/tests/test_evaluate.py`, all pass). End-to-end run on validation
(`data/runs/eval/val/report.md`); its numbers are a check, not results. Checks made:
- Every per-system row reproduces the earlier, independently computed reports: S0 and R0
  (`selection_report.md`, all findings), G and G−ind (§5.3), the starters' pilot counts
  (`pilot_summary_checklist.md`).
- **A rounding error was found and fixed by this check.** R0's per-study table stores vote shares
  to 4 decimals, while its thresholds are exact fractions (e.g. 2/150 = 0.01333… is stored as
  0.0133, just below the threshold 0.01333…). Read from the table, studies exactly at a threshold
  flip to negative: R0's validation Lung Lesion came out at sensitivity 0.27 / specificity 0.70
  instead of 0.55 / 0.44. The script now recomputes R0's scores exactly from the 150 stored
  neighbour IDs and asserts that they agree with the table to within rounding. R0's own report
  (§5.2) was computed from exact values and is unaffected.
- G's 10 neighbours equal R0's first 10 for all 400 validation studies (same index, confirmed).
- A bootstrap CI is reported as undefined ("–") when no resample defines it (a finding with no
  positives among the studies, e.g. Fracture in the 50 pilot studies).

### 5.5 Test-set results (week 6, run once, 2026-10-18)

Every system was run once on the 400 test studies with the choices, prompts and thresholds fixed
above, and scored once with `evaluate.py test --test` (method §5.4). Full tables:
`data/runs/eval/test/report.md`; every individual check: `per_study.csv` (24,000 rows).

**Runs.** All eight systems answered all 400 studies; no unanswered or missing verdicts.
P3-adapted: checklist at the top in 396 / 400 (all 400 parsable); 11 runaway answers (≈ 63k
thinking tokens) and 37 server timeouts (retried). G-img: 2 answers cite case IDs it was never
given ("Observations"). API cost of the test runs $23.21 (G $4.05, G−ind $4.22, G-img $2.75,
P1-adapted-ns $2.65, P2-adapted $4.18, P3-adapted $5.37; S0 and R0 local); project total $36.00
of the USD 45 cap.

#### Primary comparison (pre-declared, §5.4 E1): G vs. the specialist image model S0

**G ranks findings better than S0: macro AUROC 0.754 vs. 0.705, difference +0.049, 95% CI
[+0.013, +0.087].** The interval excludes zero, so on this test set the grounded image + text
system beats the zero-shot specialist model on the pre-declared criterion. The gain is
concentrated in two findings, Lung Lesion (+0.19 [+0.10, +0.29]) and Cardiomegaly
(+0.07 [+0.01, +0.12]); on Lung Opacity, Atelectasis, Pleural Effusion and Fracture the two are
level (all intervals include zero).

**Limits on this claim (stated with it, not as footnotes):**
1. **S0 is not the strongest specialist available.** The plan's DenseNet was kept ("Resolved
   decisions" #1); the ResNet-50 from the same library was far stronger on train (macro AUROC
   0.808 vs. 0.728) and was deliberately not run on test. The result is "G beats the planned
   zero-shot DenseNet", not "G beats the best image model".
2. **Gemini has probably seen IU X-Ray** in training (plan §8). Memorisation, if present, helps
   every Gemini system and not S0. It is checked after the fact in week 7 (memorisation check);
   until then the G − S0 difference is an upper bound on what grounding alone achieves.
3. **G is given finding definitions** (§4.4) that S0's labels do not share; part of the
   difference may be task specification.
4. **S0 is zero-shot** (never trained on IU data); a fine-tuned image model is not in this
   comparison.
5. **The gain depends on Lung Lesion,** a finding on which S0 is near chance in every split
   (§5.1; the cause is not established and is not a difference in definition; check promised before the test run, computed after it with the same bootstrap):

   | Macro AUROC difference [95% CI] | All six findings | Without Lung Lesion | Without Lung Lesion and Fracture |
   |---|---|---|---|
   | G − S0 | **+0.049 [+0.013, +0.087]** | +0.020 [−0.020, +0.063] | +0.008 [−0.028, +0.044] |
   | G − G-img | +0.039 [+0.008, +0.071] | +0.040 [+0.005, +0.077] | +0.030 [−0.003, +0.065] |

   Without Lung Lesion, G and S0 are not distinguishable. The pre-declared result stands as
   declared, but it rests largely on one finding on which the specialist is weak. What retrieval and context add within G's design (G − G-img) does not depend on Lung
   Lesion.

#### Where G's advantage comes from (secondary comparisons)

| # | Comparison | Macro AUROC difference [95% CI] | Reading |
|---|---|---|---|
| 5 | G-img − S0 | +0.010 [−0.029, +0.050] | Gemini reading the image alone ≈ the specialist |
| 4 | G − G-img | **+0.039 [+0.008, +0.071]** | Retrieved cases + indication add a clear gain |
| 3 | G − G−ind | +0.020 [−0.008, +0.046] | The indication's contribution is not shown |
| 2 | G − R0 | +0.028 [−0.004, +0.060] | The LLM's gain over retrieval alone is not shown (borderline) |

The primary difference decomposes: **G − S0 (0.049) = G-img − S0 (0.010) + G − G-img (0.039).**
Gemini reading the image alone does about as well as the specialist; most of G's edge comes from
the retrieved cases and the clinical context, not from reading the X-ray better. Within that,
neither the indication alone (#3) nor the LLM's reasoning over plain retrieval (#2) is shown on
its own; together, within one design, they are (#4). These are secondary comparisons (§5.4 E1):
read as descriptive, not as separate confirmed effects.

Per system (macro AUROC [95% CI]): G 0.754 [0.722, 0.784]; G−ind 0.734 [0.703, 0.763]; R0 0.726
[0.696, 0.755]; G-img 0.715 [0.683, 0.747]; S0 0.705 [0.671, 0.737]. All systems score lower on
test than on validation (e.g. S0 0.757, G 0.770 on validation). The positive-enriched test set
has many more positives of the hard, rare findings (e.g. Fracture 38 vs. 2 on validation), and
each finding weighs equally in the macro average.

#### G vs. the starter apps (stated verdicts, comparisons #6–#8)

| | Macro sensitivity | Macro specificity | Macro F1 | G − starter, F1 [95% CI] |
|---|---|---|---|---|
| P1-adapted-ns | 0.52 | 0.86 | 0.42 | −0.022 [−0.078, +0.029] |
| P2-adapted | 0.42 | 0.90 | 0.39 | −0.000 [−0.049, +0.049] |
| P3-adapted | 0.43 | 0.75 | 0.30 | **+0.097 [+0.040, +0.149]** |
| **G (stated)** | 0.32 | 0.95 | 0.39 | |
| G (threshold 0.08) | 0.62 | 0.78 | 0.42 | (not a pre-declared comparison) |

**G's stated verdicts are more cautious, not more accurate.** Against every starter, G calls
fewer findings: lower sensitivity (−0.09 to −0.20, all intervals exclude zero) and higher
specificity (+0.05 to +0.20). On F1 it is level with P1-adapted-ns and P2-adapted and better than
P3-adapted, whose generator never sees the patient's image (§3.4) and over-calls (specificity
0.75). G's advantage lies in its probabilities (AUROC, above) and in verifiable citations
(below), not in its yes / no answers. Fracture is missed by all LLM systems' stated verdicts
(sensitivity 0.00–0.18).

#### Mechanism checks

- **Retrieval quality (E2).** G's 10 neighbours contain the query's finding 1.2–7.2 times as
  often as random (lift): strongest for Pleural Effusion (×7.2) and Cardiomegaly (×3.3), weakest
  for Fracture (×1.2) and Lung Lesion (×2.9, but only 5% of neighbours). At k = 1, BiomedCLIP
  (R0 / G) retrieves better than P3's Gemini Embedding for Atelectasis (×1.6 vs. ×0.8), Lung
  Lesion (×2.6 vs. ×0.0) and Pleural Effusion (×10.2 vs. ×7.1) and about as well otherwise;
  P2's top-1 is weaker still (e.g. Pleural Effusion ×3.1), in line with Q2 (§4.2).
- **Copying (E3).** Where the neighbours' kNN decision is wrong (524 of 2,400 pairs), G's stated
  verdict agrees with it 0.426 of the time and G-img's, which never saw the neighbours, 0.399.
  **The neighbours pull G by about 3 points overall**, more for Atelectasis (0.64 vs. 0.54) and
  Cardiomegaly (0.30 vs. 0.23). G largely does not copy its neighbours; its "present" rate still
  rises with the number of positive neighbours (0.01 with none, 0.67 with 3+), but G-img's rises
  similarly (0.05 → 0.67), as both read the same image.
- **Citations (E4).** G cites selectively: of the cases it cites for a "present" call, 0.81–0.96
  have the finding (Lung Opacity, Cardiomegaly, Atelectasis, Pleural Effusion), against
  0.24–0.43 of all its neighbours. Lung Lesion is the weak spot (0.43 vs. 0.07) and Fracture is
  too rare to judge (6 calls). 8 of G's 210 "present" calls cite no case.

#### Summary

On the pre-declared criterion, the built grounded system beats the planned zero-shot specialist
model (+0.049 macro AUROC). The gain rests largely on Lung Lesion, a finding on which S0 is
near chance; without it the two are not distinguishable (limit 5). The decomposition points
to the retrieved cases and context, not to better image reading, as the source of G's edge, and
that part holds without Lung Lesion. The comparison is further bounded by S0 not being the
strongest available model and by possible Gemini memorisation of IU X-Ray. Against the starter apps, G's
stated verdicts are more cautious rather than more accurate, while its probabilities rank
better and its citations are largely verifiable. Week 7 (failure review, memorisation check,
hand check of G's rationales) tests these readings (§5.6).

### 5.6 Week 7: failure review, memorisation check, hand check of G (decided 2026-10-18)

Method: `week7_design.md` (W1–W3, decided before any case was looked at). Test set, cached
outputs, no API. **Judgements are a first pass by Claude, an AI assistant, made from text only**
(the study's report and indication, the answers, G's neighbours' reports); no radiologist, no
image reading. Samples drawn by `scripts/evaluation/week7_samples.py` (seed 2026); sheets and
judgements in `data/runs/review/` (`failure_review*.{md,csv}`, `g_rationale*.{md,csv}`).

#### W1 — Failure review (23 errors)

| Primary category | n | Systems (cases) |
|---|---|---|
| Missed finding | 6 | G 4 (three Fracture, one Lung Opacity), P1-ns 1, P2 1 |
| Copied neighbour | 6 | G 4, P3 2 |
| Invented finding | 4 | G 2, S0 1, P1-ns 1 |
| Definition mismatch | 3 | G 2, G-img 1 |
| Other | 4 | threshold artefact 2 (G at 0.08, S0), real abnormality misclassified 1 (P2), invented citation with correct verdicts 1 (G-img) |
| Hedged, text-led, label noise | 0 | (label noise once as a secondary note) |

Patterns (23 sampled cases describe patterns; they do not estimate rates):
- **Fracture: old fractures and bones beyond the ribs are missed.** Three of G's four sampled
  misses are fractures: two old rib fractures and one humerus fracture. G's rationales check for
  "acute" fractures although its definition counts old ones. This matches G's stated Fracture
  sensitivity of 0.08 (§5.5). P2 and P3 miss old rib fractures in the same way.
- **Copying is visible in the wording when it happens.** In all four G cases classed as copied,
  the rationale gives the neighbours as the reason ("a finding reinforced by its presence in the
  majority of matched similar cases"; "mirroring the clear lung findings in many similar cases"),
  or repeats a neighbour's phrase almost verbatim ("streaky opacities in the right medial lung
  base", from CXR357). Three of the four come from the stratum in which G-img, which never saw
the neighbours, was right (that is how they were sampled); the fourth (CXR1445) was a sampled
G false alarm. This
  is the mechanism behind the ≈ 3-point neighbour pull measured in §5.5 (E3). P3, whose generator
  sees only the retrieved reports, copies by design: it calls a Fracture because a neighbour's
  report mentions a "fractured sternotomy wire".
- **Cardiomegaly false alarms on low-volume / AP films.** In the two other G Cardiomegaly false
  alarms, G calls the heart enlarged "accentuated by low lung volumes / AP technique" while citing
  cardiomegaly neighbours: an image reading pushed in the same direction as the neighbours.
- **Definition mismatch around Lung Opacity.** Nodules, masses and scarring coded as Lung
  Opacity in the ground truth (or not), against G's definition that excludes nodules and masses.
  The starters showed the same pattern (§3.5).
- **The pooled threshold produces false alarms from G's own "absent" answers.** G often answers
  "absent" with probability 0.10; the pooled threshold 0.08 (§5.3) turns these into positives
  (case CXR2711). This explains why G's threshold reading has lower specificity (0.78) than its
  stated verdicts (0.95). S0's near-degenerate Lung Lesion threshold (0.0072, §5.1) does the same.
- **Label noise exists but is rare in the sample:** once (CXR3165), basal opacities described in
  the report are not coded, so G's correct Lung Opacity and Atelectasis calls count as false alarms.

#### W2 — Memorisation check (plan §8)

`scripts/evaluation/memorisation.py`; offline tests `scripts/tests/test_memorisation.py` (10,
all pass). Reference = the study's FINDINGS + IMPRESSION (INDICATION excluded); pool = up to 20
reports with the identical six-label vector, excluding every report any system was shown for
that study; 384 of 400 test studies (16 with fewer than 5 same-label reports left out).

| System | 3-gram containment own / pool | Own − pool [95% CI] | Own above pool max (chance 5.0%) | TF-IDF cosine own − pool [95% CI] |
|---|---|---|---|---|
| P1-adapted-ns | 0.053 / 0.048 | +0.004 [−0.001, +0.010] | 6.8% | +0.010 [+0.006, +0.015] |
| P2-adapted | 0.035 / 0.034 | +0.001 [−0.003, +0.005] | 4.9% | +0.009 [+0.005, +0.013] |
| P3-adapted | 0.033 / 0.031 | +0.002 [−0.002, +0.007] | 4.9% | +0.006 [+0.002, +0.010] |
| G | 0.059 / 0.053 | +0.006 [−0.000, +0.012] | 6.5% | +0.019 [+0.013, +0.025] |
| G−ind | 0.054 / 0.051 | +0.003 [−0.003, +0.009] | 5.7% | +0.011 [+0.006, +0.017] |
| G-img | 0.041 / 0.038 | +0.003 [−0.002, +0.010] | 6.8% | +0.015 [+0.010, +0.021] |

- **Primary measure: no sign of verbatim recall in any system.** The own-report excess in word
  3-grams is at most +0.006, every interval includes zero, and the own report beats all 20
  same-label reports in 4.9–6.8% of studies against 5% by chance. That includes G-img, which sees
  only the image and would be the clearest sign of image-based recall.
- **Secondary measure: a small word-level excess in every system** (TF-IDF +0.006 to +0.019, all
  intervals above zero). This is not by itself a sign of memorisation: reports describe many
  case-specific details beyond the six labels (granulomas, scoliosis, devices, locations), and a
  system that reads the case correctly uses the same content words. The ordering fits that
  reading: P3, which never sees the image, has the smallest excess; G, which also receives the
  indication (whose wording can recur in the impression), the largest.
- **Limit:** a negative 3-gram result rules out recall of *wording*, not recall of the *labels*
  without the wording. Caveat 2 of §5.5 is therefore weakened but not removed.

#### W3 — Hand check of G: status vs. rationale (15 answers, 90 findings)

| Judgement | n |
|---|---|
| Status agrees with its own rationale | 88 |
| Unclear | 2 |
| Disagrees | 0 |

The two unclear calls are Lung Opacity "present": one rationale describes interstitial markings /
early edema rather than the airspace opacity of the definition (CXR1202), one hedges "airspace
disease or overlying soft tissue" at probability 0.60 (CXR2570).

Basis of the 17 "present" calls, as stated in their rationales: **image description alone 10;
image + cited cases 4; image + indication 3; cases or indication alone 0.** Every "present" call
describes an image appearance; the cases and the indication appear as support, never as the sole
reason. This partly answers the limit noted in E4 (a relevant citation does not show that G
reasoned from it): in this sample G's stated reasoning starts from the image. The rationale is
G's self-report and may not reflect how the answer was actually produced; and W1 shows that when
G errs by copying, the wording does change ("mirroring … similar cases").

#### Week 7 summary

- G's errors have recognisable causes: missed old / non-rib fractures, a visible pull towards its
  neighbours in a minority of cases, definition mismatches around Lung Opacity, and a low pooled
  threshold that turns cautious "absent" answers into positives.
- No system shows verbatim recall of the IU reports; the G − S0 result (§5.5) is therefore not
  explained by recalled report wording, though memorised labels cannot be excluded.
- G's statuses match its own rationales (88 / 90), and its "present" calls are stated in terms of
  the image, with cases and indication as support.

## 6. Threats to validity

Final list (2026-10-18). The study-level split is not a threat: the collection has one study per
patient (§2.2).

- **Gemini has probably seen IU X-Ray** (plan §8). Memorisation would favour every Gemini system
  over S0. The after-the-fact check (§5.6, W2) found no verbatim recall of report wording in any
  system, including the image-only G-img; recalled labels without wording cannot be excluded.
- **One collection, one test sample.** 400 positive-enriched test studies from one hospital
  collection; per-finding intervals are wide (e.g. Fracture, Lung Lesion). No external
  validation (the planned MIMIC replication belongs to the thesis).
- **Test-set enrichment:** sensitivity, specificity and AUROC do not depend on prevalence, but F1
  does; test-set F1 is not comparable with natural-prevalence F1 (§5.5).
- **Hand judgements are a first pass by an AI assistant, from text only** (the starters' checklist
  check, §3.5; week 7, §5.6). No radiologist reviewed them and no image was read; the failure
  categories and the rationale-basis labels are therefore interpretations of the written reports
  and answers.
- **Rationales are self-reports.** That G's "present" calls are stated in terms of the image
  (§5.6, W3) shows how G explains itself, not necessarily how its answer was produced.
- **Selection effect:** the 161 lateral-only studies excluded for lacking a frontal image are
  somewhat sicker than the rest (§2.4b). They are 4% of studies with images.
- **Thin retrieval index for rare findings:** test enrichment leaves only 49 Lung Lesion and
  43 Fracture positives in train, which limits the retrieval systems on those findings (§2.5).
- **Label noise:** CheXbert misreads a templated normal report as Cardiomegaly (55 false
  positives, 13% of Cardiomegaly positives), and agreement with the manual codes is lowest for
  Pneumonia (κ 0.61) (§2.4b).
- **Lung Lesion and Fracture are hard for the image-based anchors here:** S0's DenseNets and R0
  are near chance on them on train and validation (§5.1, §5.2). For S0's Lung Lesion this is a
  model weakness, not a definition difference: its training label defines Lung Lesion essentially
  as the ground truth does (§5.1; an earlier version of this point suggested a label-definition
  mismatch, see "Corrections"). **This changes the reading of the primary result**
  (check promised here on 2026-10-17, before any test run; computed 2026-10-18, §5.5): G − S0 is
  +0.049 [+0.013, +0.087] over six findings, but +0.020 [−0.020, +0.063] without Lung Lesion and
  +0.008 [−0.028, +0.044] without Lung Lesion and Fracture. Old fractures are missed by all LLM
  systems (§5.6).
- **S0 is not the strongest available image model.** Following the plan, S0 is a DenseNet; a
  TorchXRayVision ResNet-50 (also never trained on IU) scored clearly higher on train (macro
  AUROC 0.808 vs. 0.728, §5.1). G beat S0 on test (+0.049, §5.5); that does not show it beats the
  best single image model, and the margin is smaller than the DenseNet–ResNet gap on train. S0 is
  also zero-shot (never trained on IU data).
- **G is given finding definitions the starters are not** (§4.4): G vs. a starter also differs in
  how precisely the task is specified. Q3 measures how much results differ, not why (plan §5).
- **Thresholds were tuned outside the test distribution:** S0 and R0 on train (natural
  prevalence), G on validation. Youden's J is prevalence-free, but the score distributions of
  the enriched test set may still differ.

## 7. Further research (possibilities)

Added 2026-10-24, after the report was finalised. Three questions raised after the semester
work: (7.1) how the two embedders compare, (7.2) what the results suggest for improving G,
(7.3) whether a system like G is easier to carry over to other imaging tasks than a specialist
model. **Nothing in this section was run; it rests on the results above and lists
possibilities, not results.** None of it changes §1–§6. Any new run on the test set would be
after the fact (the test set was declared "run once", §5.4) and would have to be reported as
such; validation is the place to try these first.

### 7.1 Gemini Embedding 2 vs. BiomedCLIP

**What was measured.** Only retrieval quality, not the effect on a full system. The embedder was
never swapped inside G, and P2 (the only arm that retrieves with Gemini Embedding 2) also differs
from G in k (1 vs. 10), in what it passes on (an image without its report vs. reports) and in its
prompt, so P2's results do not show what the embedder contributes.

- **Retrieval (train, leave-one-out kNN vote, §4.2): BiomedCLIP is clearly better at small k.**
  Macro AUROC at k = 5: 0.651 vs. 0.574 (−0.077 [−0.100, −0.054]); at k = 10, G's k: 0.671 vs.
  0.600. At k = 150 the embedders converge (0.741 vs. 0.720). The difference is largest exactly
  where an LLM system works, with the handful of cases a prompt can hold.
- **Test set (§5.5, E2):** at k = 1 BiomedCLIP retrieves the query's finding more often for
  Atelectasis (lift ×1.6 vs. ×0.8), Lung Lesion (×2.6 vs. ×0.0) and Pleural Effusion (×10.2 vs.
  ×7.1), about as well for the other findings.
- **Settings of the Gemini embedder matter little:** 3,072-d and 768-d are level; P3's image +
  report blend helps (+0.019 at k = 5) but stays below BiomedCLIP on images alone (§4.2).
- **Both pack chest X-rays into a narrow similarity band** (median nearest-neighbour similarity
  0.979 Gemini, 0.970 BiomedCLIP; §2.3, §3.2), which is why similarity-weighted votes were not
  used.

**Cost.** Small for both, against $4–5 per 400-study generation run:

| | BiomedCLIP | Gemini Embedding 2 |
|---|---|---|
| Train index | $0, local on the RTX 4060 (run time not logged) | $0.33 for 2,784 images, ≈ 28 min (P2); $0.38 for P3's image + report index |
| Per query image | $0 | $0.00012 ($0.048 per 400 test studies) |
| Share of the project's API spend | 0% | ≈ $0.83 of $36.00 (≈ 2.3%) |

The practical differences outweigh the cost: BiomedCLIP runs locally (no image leaves the
machine) with pinned weights, while the Gemini embedder sends every image to an API whose model
version can change.

**Reading.** On retrieval quality, cost, privacy and reproducibility, BiomedCLIP is the better
choice for this task. Whether the retrieval gap carries through to G's scores is not measured.

**Possible check:** G with Gemini Embedding 2 retrieval (k = 10, prompt 2, everything else
frozen) on the 400 validation studies, ≈ $4 at G's cost per study, compared with G by the same
paired bootstrap.

### 7.2 Possible improvements to G

Ordered by how directly the results point to them. Items 2–4 can be computed from cached
outputs, without API calls.

1. **Fracture definition.** G's rationales check for "acute" fractures although its definition
   counts old ones; three of its four sampled misses were old rib or humerus fractures (§5.6,
   W1); stated Fracture sensitivity 0.08 (§5.5). Possible change: state that healed and old
   fractures and bones beyond the ribs (clavicle, humerus, spine) count, and add a step that
   reviews the bones.
2. **Calibration and thresholds.** G often answers "absent" with probability 0.10, and the pooled
   threshold 0.08 turns these into false alarms (§5.6, W1); on validation its probabilities are
   overconfident (stated ≈ 0.84 → present 35%; diary, 2026-10-18). Options: per-finding
   thresholds (limited by Fracture's 2 validation positives, §2.5), isotonic or Platt
   calibration on validation, or averaging several answers per study for finer probabilities
   (cost × the number of answers).
3. **Combine G with S0, or with the stronger ResNet-50.** G-img and S0 are level (+0.010
   [−0.029, +0.050], §5.5), so the two may err on different studies. An average of G's and S0's
   probabilities is the plan's dropped G+S0 arm and the thesis plan's hybrid; the ResNet-50
   (0.808 vs. 0.728 on train, §5.1) is the obvious stronger anchor.
4. **Give G the large-k signal.** The kNN vote is much stronger at k = 150 (0.741) than at k = 10
   (0.671) (§4.2), but G sees only 10 reports. One structured line per finding with R0's vote
   share at k = 150 adds few tokens and keeps the 10 citable reports.
5. **Reduce copying.** The neighbours pull G by ≈ 3 points, more for Atelectasis and Cardiomegaly
   (§5.5, E3), and the copied cases show it in the wording (§5.6, W1). Options: two stages (read
   the image first, as G-img does, then revise with the cases); point out low-volume and AP
   films, where the Cardiomegaly false alarms came from.
6. **Retrieval for rare findings.** Lung Lesion: 49 train positives (§6), 5% of neighbours carry
   it, cited-case agreement 0.43 (§5.5, E4). Options: retrieve also by the indication's text;
   a chest-X-ray-specific embedder (e.g. RAD-DINO, CXR Foundation); a larger index (MIMIC-CXR,
   planned for the thesis).
7. **Smaller items:** add the lateral image; test a lower thinking level for cost (G ≈ $0.010 per
   study); in an image-only arm, say that no cases are provided or drop the citation field (2 of
   450 G-img answers cited IDs never given, "Observations").

### 7.3 Is a system like G easier to carry over to other imaging tasks than a specialist?

**Reading: easier to re-target, yes; better when re-targeted, not shown.** This project covers
one modality, one collection and one test sample.

**For:**
- **No training.** Moving G to a new task means a new case index (images + reports), new finding
  definitions in the prompt, and possibly another embedder; a specialist needs labelled training
  data and a training run per task. BiomedCLIP was trained on figures from PubMed Central
  (PMC-15M) across many modalities, not only chest X-rays.
- **The task definition sits in the prompt, not in the weights.** Adapting to a new site's
  labelling conventions is a prompt change for G and retraining for a specialist. This project
  gives no direct evidence of how much that matters: S0's weakness on Lung Lesion was first
  attributed to a label mismatch, which the documentation does not support ("Corrections").
- **G's gain comes mostly from retrieval and context, not from reading the image better**
  (G − G-img +0.039, also without Lung Lesion; G-img − S0 +0.010; §5.5). That mechanism does not
  depend on the modality, only on a library of reported cases.

**Against:**
- **Chest X-ray is likely the most favourable case for a general model** (common in public data),
  and even here Gemini reading the image alone only matched a zero-shot specialist. In rarer
  modalities the general model may be weaker.
- **CT and MRI are volumes;** Gemini and BiomedCLIP take 2D images, so slice selection,
  windowing and summarising a volume become design problems of their own. Skin photographs and
  dermoscopy fit the current design far more directly.
- **G still needs labelled data,** as an index rather than for training, and its evidence is
  report text; many collections (e.g. dermoscopy) have a diagnosis label but no report.
- **Each new domain needs its own validation** like this one; easier to build is not safe to use.
- **Specialists also transfer:** S0 was never trained on IU data, and medical foundation models
  (e.g. MedGemma) narrow the gap between specialist and general models.
- **Running cost and data flow:** every study is an API call with a patient image, and the model
  version can change; a local model (MedGemma-4B, thesis plan) addresses this.

**Defensible wording:** "A retrieval-grounded vision-language system like G can be re-targeted
to a new imaging task without training, by changing its case index and task definitions. Whether this comes with competitive accuracy in other modalities was not tested."

**Possible test (thesis scale):** the same arms (G, G-img, a specialist, a kNN baseline) on a
second public modality, most directly dermoscopy (ISIC / HAM10000), with the same frozen-before-
test protocol.

---

## Open decisions

None at present.

## Resolved decisions

1. **S0: keep the selected DenseNet, or use the ResNet-50 (`resnet50-res512-all`)?** (raised
   2026-10-17, §5.1). **Decision (2026-10-17): keep the DenseNet (mimic_nb, pad), as the plan
   specifies; no further deviation from the plan.** The ResNet-50 stays in §5.1 as a reported
   reference only and is not run on test. Consequence: see §6, "S0 is not the strongest
   available image model".
   - *For the ResNet:* it is far better on train (macro AUROC 0.808 vs. 0.728, +0.080
     [+0.052, +0.106]). S0 is the "single image model" the thesis tests the multimodal systems
     against; a clearly weaker S0 than available would make "image + text beats the specialist"
     easier to claim than it should be. It was not trained on IU either (pc-nih-rsna-siim-vin).
   - *For the DenseNet:* plan §3 names "TorchXRayVision DenseNet"; the ResNet's Lung Lesion
     output is untrained, so its Lung Lesion score would be a proxy, max(Nodule, Mass), chosen
     in this project. That is an adaptation of S0.
   - The comparison was made on train only. The ResNet's validation results have deliberately
     not been looked at, so that validation plays no part in the choice.
   - Either way the choice and its reason go into the deviations table if it departs from the
     plan's wording.
2. **How the test set is evaluated: the details plan §5 leaves open** (raised 2026-10-18,
   `eval_design.md`). **Decision (2026-10-18): all five recommendations adopted** (§5.4):
   E1 a pre-declared list of compared differences, primary = G − S0 macro AUROC; E2 precision@k
   per finding on positive queries, with train prevalence as baseline, at each system's k and at
   k = 1; E3 copying rate against the neighbours' kNN decision at k = 10, with G-img as control;
   E4 citation agreement with an all-neighbours baseline; E5 one evaluation script with offline
   tests, checked on validation first.
3. **Run P1 with its DuckDuckGo web search on (P1 as shipped), 50 studies, ~$0.5–1.5?** (raised
   2026-10-17, §3.3; listed in `thesis_plan.md`'s systems table as "a 50-case sensitivity run with
   it on". The semester plan turns search off (§3) but does not list this arm; earlier entries
   here cited it as "plan §3", which was inaccurate.) **Decision (2026-10-18): not run.**
   - *Reproducibility (the reason already recorded for turning search off):* live web results
     change over time, so the arm could not be repeated.
   - *Search does not make the diagnosis:* section 5 of P1's prompt asks only for literature and
     references; the findings come from the model reading the image. The arm would mostly
     measure reference-gathering, not what the thesis compares.
   - *Possible route to the data itself:* IU X-Ray is publicly searchable online (Open-i). A
     search on the findings could return IU reports, possibly the study's own; this cannot be
     ruled out and would be an uncontrolled input no other system receives (in addition to
     the memorisation risk of plan §8).
   - *Not comparable, costlier:* it would be the only system with web access, with search results
     added to its context, so its results could not be read cleanly against the other arms.
   - *For running it:* it would be the only measurement of P1 exactly as shipped (prompt and tool
     together); without it, Q1 can say only that P1 fails when its tool is removed (§3.3), not how
     it performs with the tool. It would not have entered the test comparison (P1-adapted-ns is
     the P1 arm either way).
   - Consequence: Q1 describes P1 as shipped only through P1-adapted's pilot (tool removed) and
     P1-adapted-ns; the `--web-search` option of `p1_adapted.py` stays unused.

## Deviations from the plan

| Date | Plan said | What happened | Effect |
|---|---|---|---|
| 2026-10-16 | Split by patient if patient IDs exist, otherwise by study (§2, §6) | No patient IDs exist → split by study | The plan's "if not" branch applies: study-ID index assertion, repeat-patient screen, week-7 robustness check |
| 2026-10-16 | Plan §6 said to "say so plainly in §2 and list it in §8" | The plan is frozen; recorded here (§2.2) instead | None on the method |
| 2026-10-16 | "If not" branch: repeat-patient screen (week 1) and repeat-patient robustness check (week 7) | The dataset paper states one study per patient, so the study split is a patient split. The repeat-patient screen is reduced to a near-duplicate image check; the week-7 robustness check is dropped | Less work in weeks 1 and 7; no loss of rigour |
| 2026-10-16 | Every system gets the study's frontal image (§2), implicitly assuming views are labelled | The Open-i release has no frontal/lateral label | Extra week-1 step: classify views and check the classifier by eye on a sample (done, §2.4) |
| 2026-10-16 | (not anticipated) | 4 unusable images (3 blank, 1 noise) found and dropped | None: no study lost its only usable image |
| 2026-10-16 | Plan §3 assumed some S0 weights might include Open-i | Confirmed: `densenet121-res224-all` includes Open-i | That option is ruled out; S0 will use weights without Open-i (§2.3b) |
| 2026-10-16 | Ground truth = CheXbert labels on the reports (§2) | Ground truth = the dataset's manual MeSH codes; CheXbert kept for system outputs (decision after CheXbert showed a systematic Cardiomegaly error) | Independent reference; labeller error measured directly against it instead of only by the planned hand check. 86 uncoded studies excluded |
| 2026-10-16 | Keep findings with enough positives (≥ ~30 in test) | Six findings, chosen by count and by S0 coverage (Support Devices dropped) | All systems, S0 included, score the same six |
| 2026-10-16 | P2-orig: Cohere embed-v4.0 (§3) | **P2-adapted**: Gemini Embedding 2 (default settings, own embeddings, not shared with P3), decision to avoid a paid Cohere account | P2 no longer uses its shipped retrieval embedder |
| 2026-10-16 | Starters keep their hard-coded models; successor if retired (§3) | `gemini-2.5-pro` and `gemini-2.5-flash` retired for new users → P1 and P2 use `gemini-3-flash-preview` (decision); P1 renamed **P1-adapted** (Pro → Flash) | All starters share one model: no model confound between them; less faithful to shipped versions |
| 2026-10-16 | Decision thresholds tuned on validation (§2) | Validation has too few rare positives (Fracture 2). S0: tuned on train; R0: leave-one-out on train; G/G−ind: one pooled threshold on validation (decision) | Thresholds rest on enough positives; G's pooled threshold is coarser than per-finding tuning |
| 2026-10-17 | Starter prompts left as shipped; P1 with web search off (§3) | With search off, P1's verbatim prompt still orders a DuckDuckGo search; 62% of pilot studies ended in `MALFORMED_FUNCTION_CALL`. P1-adapted is kept as evidence and not run further; replaced by **P1-adapted-ns**: section 5 ("Research Context") removed, everything else unchanged (§3.3) | P1's prompt is no longer as shipped; the starter comparison uses P1-adapted-ns, an image + fixed-prompt arm |
| 2026-10-17 | Starters' free text labelled by CheXbert, uncertain → negative (§5); all get the same findings question (§3) | CheXbert misreads the answers' format (e.g. "Cardiomegaly: Not present" → positive; definitions such as "Cardiomegaly refers to an enlarged heart" → positive; 38 / 50 P2 answers contain such sentences), so the pilot labels were invalid. The findings question gets a format instruction (**checklist question**): each answer begins with six lines "<finding>: present / absent / uncertain", parsed directly (§5) | No labeller error for the starter comparison; all systems scored on their own stated verdicts (uncertain → negative as planned); unparsable verdicts counted and reported. The starters' own prompts are unchanged; pilots re-run (~$0.9) |
| 2026-10-18 | P3 run as shipped (only the minimal adaptations, §3) | With the checklist question, P3's answering call thinks much more (median 861 tokens) and its shipped `max_output_tokens=900` includes thinking: 45 / 50 pilot answers were cut off inside the checklist (original question: 4 / 50). The cap is removed (model default, as for P1 and P2); the arm is renamed **P3-adapted** | P3's checklist is complete; its configuration differs from the shipped app in one generation parameter. P3-orig's runs are kept as evidence |
| 2026-10-18 | G's k chosen on validation (§2, §6) | **k = 10 fixed in advance** from R0's leave-one-out curve on train (§5.2); tuning on validation would cost ≈ $7 per k value | One G configuration is run; the copying-rate analysis uses k = 10 |
| 2026-10-18 | Systems: P1–P3, S0, R0, G, G−ind, optional G+S0 (§3) | **G-img added** (G's design with the image only; test set); **G+S0 dropped** (plan's buffer rule) | G vs. G-img contrasts image-only and image + retrieval + text within one design |
| 2026-10-18 | Thesis plan (systems table): P1 with web search on, a 50-case sensitivity run (not in the semester plan) | Not run ("Resolved decisions" #3): irreproducible; search adds references, not findings; possible route to the public IU reports; an incomparable, costlier arm | Q1 has no measurement of P1 with its tool; the test comparison is unchanged |
| 2026-10-18 | Precision@k = "share of retrieved training cases that share the query's labels" (§5) | Measured **per finding on queries that have it**, with train prevalence as baseline (lift); normal queries in a separate row; at each system's k and at k = 1 (§5.4 E2) | Measures retrieval of the same disease instead of mostly normal–normal matches |
| 2026-10-18 | Copying rate = agreement with "the neighbours' majority vote" (§5) | Neighbours' decision = R0's rule at k = 10 (vote share ≥ per-finding Youden threshold, leave-one-out on train); **G-img as control** (§5.4 E3) | A strict majority is almost never positive for these findings; the control separates influence from shared image reading |
| 2026-10-18 | Paired bootstrap CIs "for the differences" (§5), no list | **Pre-declared list**, primary = G − S0 macro AUROC; no multiple-comparison correction, secondary comparisons labelled as such (§5.4 E1) | A specification, not a change of method; fixed before any test result |

## Observations relevant to later steps

- **G-img invented case citations once (test, CXR206; seen during the run).** G-img gets no
  similar cases (retrieval is skipped), but it shares G's prompt and output schema (g_design.md
  D7): step 2 still asks for "the IDs of the similar cases that support your call (an empty list
  if none)" and says "Cite cases by their ID exactly as written, e.g. CXR316", and the response
  schema has a required `cited_case_ids` list per finding. For CXR206 the model filled the slot
  with IDs in the right format that it was never given: CXR102 (Lung Opacity, present, 0.85),
  CXR205 (Atelectasis, uncertain; the patient's own ID minus one) and CXR331 (Lung Lesion,
  present); the three "absent" findings cited nothing. `validate()` flagged it ("cited cases not
  provided"); by the export rule the statuses still count. Final frequency: **2 of 450** G-img
  answers (50 pilot + 400 test); the second, CXR3080, cited CXR412, CXR882 (Lung Opacity) and
  CXR109 (Atelectasis) in the same way. All other answers left every list empty. Not scored (E4 covers only G
  and G−ind, whose cases are real). The model cannot check whether an ID is real; only the
  after-the-fact validation does. For the failure review and the discussion: an output slot for
  evidence can elicit fabricated evidence when none exists. A future image-only arm would need
  to say that no cases are provided, or drop the field, at the cost of identical prompts.

- Some indications state the finding outright (CXR1021: indication "Right pneumothorax."). This
  matters when interpreting G vs. G−ind: part of what the clinical text adds may be the answer
  itself.
- **For G's design (Q2): retrieval evidence at small k is weak.** R0 with the 5 nearest cases
  reaches macro AUROC 0.651 (leave-one-out, train), against 0.741 with 150 (§5.2). G's prompt can
  hold only a few cases, so the retrieved cases alone are weak evidence; G must rely on the
  image itself. For the copying-rate analysis (plan §5), the neighbours' vote should be computed
  at G's k, not R0's.
- **For G's design (Q2): BiomedCLIP's preprocessing centre-crops** (resize the short side to
  224, then crop 224 × 224), so the retrieval index does not see the lateral edges of landscape
  films. For S0, padding instead of cropping improved Pleural Effusion and Fracture (§5.1). A
  padded BiomedCLIP index is a cheap option to test when G's index is fixed; R0 would then be
  re-run on the same index (plan §3: "same BiomedCLIP index as G").

## Corrections

- **2026-10-31 — Why S0 is weak on Lung Lesion.** Earlier versions of this report (Summary,
  §5.1, §5.5, §6, §7.3) explained S0's near-chance Lung Lesion score as a label-definition
  mismatch: S0's "Lung Lesion" label was said not to match IU's MeSH nodule / mass codes, and
  the primary result was described as resting on a finding where the specialist is
  "handicapped by a label mismatch". This was an inference from the scores (first written
  2026-10-17) and was never checked. Checked now against primary sources:
  - MIMIC-CXR-JPG documentation (PhysioNet, v2.1.0): the NegBio labels, which S0
    (`densenet121-res224-mimic_nb`) was trained on, were produced by "NegBio … run using the
    custom mention patterns defined by the CheXpert tool"; labels come from the IMPRESSION
    section (FINDINGS if there is none).
  - CheXpert labeller phrase files: Lung Lesion is triggered by mass, nodular density /
    densities, nodular opacity / opacities / opacification, nodule, lump, cavitary lesion,
    carcinoma, neoplasm, tumor; "calcified nodul…" is excluded.
  - TorchXRayVision 1.5.5 `MIMIC_Dataset` (installed source): positive = 1; negated mention or a
    "No Finding" study = 0; uncertain mentions and unmentioned findings are ignored in training.
  - This project's ground truth (§2.4b): Nodule, Mass or Cavitation in the lung, calcified
    granulomas excluded; none of the 104 Lung Lesion positives rests only on a calcified
    nodule code.

  The two definitions are essentially the same. The differences lie in how the labels were made
  (keyword rules on report text vs. human coding), not in what counts as a lesion. **Corrected
  reading:** S0 is weak on Lung Lesion on IU for reasons not established (possible: noisy
  text-mined training labels, small lesions at 224 px, poor transfer between hospitals). The
  facts of the primary result are unchanged: the gain rests largely on Lung Lesion, and without
  it G − S0 is not distinguishable from zero. What changes is the stated reason: S0 is weak on
  this finding; it was not graded against a different definition. The same correction was made
  in `presentation/final_presentation.html`, `presentation/summary_presentation.html` and
  `presentation/qa_prep.md`.
