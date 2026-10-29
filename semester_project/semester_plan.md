# Semester Plan — Starter Agents vs. a Grounded Image + Text System (Chest X-ray)

Scope for this semester (~8 weeks). Drafted 2026-10-16.
The full thesis plan is `../thesis_plan.md`; this semester is its pilot study and first chapter.
Background on the starters: `../codebase_analysis.md`. **Assumptions to confirm with the supervisor are marked ⚠.**

---

## 1. Goal in one paragraph

Find out what the three starter projects from `main.txt` are, how they work and how well suited
they are to medical image analysis. Identify their strengths that can be combined into a
multimodal, grounded image + clinical-text system. Build that system and measure how much its
results differ from the starters' results on the same public chest-X-ray data. A single
specialist image model (S0) and a retrieval-only baseline (R0) give the comparison anchors.
Everything is decision support only and uses public, de-identified data.

**Questions**
- **Q1 (fitness):** How do the three starters work, and how well do they perform on chest-X-ray
  finding detection when run with minimal adaptation?
- **Q2 (strengths):** Which of their components are worth carrying into a grounded image + text
  system, and which gaps must be filled?
- **Q3 (comparison):** How much do the built system's results differ from the starters' results,
  from a single specialist image model, and from retrieval alone?

**Out of scope this semester** (moved to the thesis): the full one-change-per-step ladder, the
mechanism probes (blank image, shuffled text, wrong-label retrieval, image perturbation),
contamination testing, calibration analysis, claim-level checking of explanations, human rating,
MIMIC-CXR, local MedGemma. The copying rate (§5) is kept as a cheap, light version of the
thesis's anchoring analysis, because it needs no extra runs.

---

## 2. Data

- **IU X-Ray (Open-i):** images + report + "indication" field, fully public ✅.
- **Unit of analysis: the study.** Labels come from the study's report. Every image-based system
  receives the study's **frontal image only**, because P1 accepts a single image as shipped and
  this keeps all systems equal. Studies without a frontal image are excluded. Findings visible
  only on the lateral view will count as misses for every system; this is stated as a limitation.
- **Labels:** CheXpert-style findings extracted from the reports with CheXbert. Keep only
  findings with **≥ ~30 positive studies in the test set** (expected 5–8 findings).
- **Splits** (by patient, every study of a patient in one split, **if** the release has patient
  IDs; otherwise by study. See the week-1 patient-ID check in §6):

  | Split | Use |
  |---|---|
  | Train | Retrieval index for P2, P3, R0 and G. Nothing else |
  | Validation | Prompt development, choice of k, decision thresholds. The ~50-study pilot set is drawn from here |
  | Test | A few hundred studies, **enriched for positives** (stratified sampling) so each kept finding has enough cases. Run once, at the end |

  Enrichment changes prevalence. Sensitivity, specificity and AUROC are unaffected; F1 and
  precision are not comparable to real-world prevalence, and this is stated.
- **Leakage guards** (week 1, in code):
  - Assert that no validation/test patient ID (or study ID, if there are no patient IDs) appears
    in any retrieval index (an easy mistake with P3, which indexes report text).
  - Screen for near-duplicate images and repeat patients across splits with BiomedCLIP embedding
    similarity (§6, week 1).

---

## 3. Systems

| ID | System | Notes |
|---|---|---|
| **P1-orig** | `ai_medical_imaging_agent`: Gemini + 5-section prompt, image only | Batch runner instead of the UI; DuckDuckGo web search off (reproducibility); the findings question is appended to its fixed prompt (P1 has no question input) |
| **P2-orig** | `vision_rag`: Cohere embed-v4, top-1 retrieval, Gemini | Queries by the patient's image instead of a text question, and also passes that image to Gemini; as shipped, only the retrieved image reaches the model |
| **P3-orig** | `multimodal_agentic_rag`: Gemini Embedding 2 (image + report blend), top-6, ADK agent | Queries by image embedding; called directly without server/UI. Its generator sees only the retrieved *text*, never an image |
| **S0** | TorchXRayVision DenseNet, no training | The single-image-model reference point. ⚠ Use weights **not** trained on Open-i (= IU X-Ray); the combined "all" weights may include it. Checked in week 1 |
| **R0** | Vote over the k nearest training cases' labels (same BiomedCLIP index as G), no LLM | Retrieval alone; shows whether Gemini adds anything beyond copying neighbours. No API cost |
| **G** | **Built system** (§4) | The grounded image + text system |
| G−ind | G without the indication text | Measures what the clinical text adds |
| G+S0 (optional) | G with S0's probabilities in the prompt | ~1 extra day |

**Rules for the starter arms:**
- All three get the same findings question ("Which of these findings are present: …").
- Every adaptation is the minimum needed to run them on X-rays, and each is recorded with its
  reason. Their prompts are otherwise left as shipped. In particular, refusals are recorded, not
  prompted away.
- If an adaptation changes a core component (e.g. Cohere is replaced for cost reasons), the arm
  is renamed (e.g. **P2-adapted**) and the change is reported.
- ⚠ The starters' hard-coded model IDs (`gemini-2.5-pro`, `gemini-2.5-flash`,
  `gemini-3-flash-preview`) may be retired. If so, use the nearest successor and document it.

---

## 4. The built system (G)

```
frontal image ──► BiomedCLIP embedding ──► top-k similar training cases (image + report)
                                                     │
frontal image + indication text + retrieved reports ─┴─► Gemini (pinned model ID)
                                                     │
   JSON: {findings: [{finding, present, confidence 0–1, cited_case_ids[], rationale}],
          impression: "<short free-text summary>"}
```

What comes from each starter:
- **P1:** the prompt structure (adapted to JSON output) and the Gemini call pattern.
- **P2:** the embed → retrieve → generate pattern, upgraded from top-1 to top-k and from a
  general-purpose to a medical embedder.
- **P3:** structured citations with similarity scores and the instruction to hedge on weak
  evidence.

Gaps G fills that none of the starters do: clinical-text input, per-finding scores, the patient's
image *and* the retrieved evidence reaching the model together, and machine-checkable citations.
G's prompt may use a decision-support framing to reduce refusals; this is part of G's design.
Every API response is cached **in full** (raw text, not just parsed labels) with its exact model
ID. §8 needs the raw text.

---

## 5. Evaluation

**Two scoring paths, so the starter comparison is fair:**
- **Starter comparison (Q3 vs. P1–P3):** every system's free text goes through the same CheXbert
  labeller. For G that is its `impression` field. The labeller's accuracy can still depend on
  writing style (e.g. P1's differential-diagnosis list mentions conditions it does not assert).
  CheXbert's "uncertain" labels count as negative, fixed in advance.
- **Anchored comparison (Q3 vs. S0, R0):** G's JSON scores are used directly, alongside S0's and
  R0's continuous scores.

| What | How | Applies to |
|---|---|---|
| Finding accuracy | Per-finding and macro F1, sensitivity, specificity vs. report-derived labels | All systems |
| Ranking quality | AUROC | S0, R0, G (JSON path) |
| Retrieval quality | precision@k: share of retrieved training cases that share the query's labels | P2, P3, R0, G |
| Copying rate | How often G's answer equals the neighbours' majority vote, especially when that vote is wrong | G vs. R0 |
| Citation label agreement | Share of G's cited cases that actually have the finding G claims | G only (the starters don't cite per finding) |
| Effect of clinical text | G vs. G−ind | G |
| Failure cases | ~20 hand-reviewed errors: missed finding, invented finding, laterality, copied a wrong citation, refusal | All systems |
| Qualitative fitness | Output format, refusals, hedging, invented content, on the pilot set | Starters (Q1) |

The CheXbert labelling of free text is checked by hand on ~15–20 outputs per system, with
agreement reported per system.
Statistics: paired bootstrap confidence intervals (resampling studies) for the differences.

**What the comparison can and cannot conclude:** G differs from each starter in model,
retrieval, prompt and output format at once. Q3 measures *how much* the results differ, not
*which change caused it*. Attribution is the thesis ladder's job.

---

## 6. Timeline (8 weeks)

| Week | Work | Deliverable |
|---|---|---|
| 1 | IU X-Ray download; **patient-ID check** (below); train/val/test split with leakage assertions and repeat-patient screen; CheXbert labels; choose findings; check model availability, S0 weights' training data and Cohere pricing; API keys | Data module + label stats + split note (patient or study level) |
| 2 | P1 batch runner + pilot run (**early warning**, §8); build the P2 (Cohere) and P3 (Gemini Embedding) indexes over the train split; run S0 (local) | P1 pilot results; indexes; S0 baseline |
| 3 | **Q1:** P2 and P3 runners + pilot runs; starter fitness write-up. **Q2:** strengths/gaps synthesis → design of G | Q1 write-up; G design |
| 4–5 | Build G: BiomedCLIP index, R0, prompt, JSON parsing, caching; tune k and thresholds on validation | G and R0 running on validation |
| 6 | Test-set runs, once: P1–P3, S0, R0, G, G−ind (+ G+S0) | Raw results, cached |
| 7 | **Q3:** metrics, bootstrap CIs, retrieval quality, citation agreement, copying rate, failure review, threats-to-validity check (§8), repeat-patient robustness check (if split by study) | Results tables + plots |
| 8 | Write-up + supervisor presentation | Semester report |

**Week 1 — patient-ID check (hypothesis to confirm).** The public IU X-Ray release may have no
patient identifier, only a study/report ID (`CXR####`), removed by de-identification. Not yet
verified; check the downloaded files first.
- **If a patient ID exists:** split by patient as in §2; the assertion checks patient IDs.
- **If not:**
  - The split is **by study**; say so plainly in §2 and list it in §8. A patient with several
    studies could appear in both train and test, so retrieval could return their earlier exam.
    That would favour the retrieval systems (R0, G, P2, P3) over P1 and S0.
  - Keep the index assertion, relabelled as a **study-ID** check. It still catches indexing the
    wrong split, the likeliest bug (P3 indexes report text). It does not catch repeat patients.
  - Find out whether repeats exist at all: the dataset paper (Demner-Fushman et al., 2016) for
    how exams were selected, and the reports' COMPARISON field and "unchanged / stable since"
    wording. These show a prior exam existed, not that it is in the collection.
  - **Repeat-patient screen:** BiomedCLIP embedding similarity between train and test images
    (replaces the perceptual hash). Inspect the ~20 most similar train–test pairs by eye for the
    same patient (matching devices, wires, unusual anatomy) and report the count.
  - Not used: a looser perceptual-hash threshold (chest X-rays hash alike, so it would flag
    different patients and still miss re-exams), and report-text matching (IU reports are too
    templated; matches would be mostly normal–normal pairs).
  - **Robustness check (week 7):** re-compute R0 and G without the test studies whose nearest
    training neighbour is unusually similar. If the conclusions hold, residual repeats don't
    matter, even if detection was imperfect.

Buffer: if weeks 4–5 run over, drop G+S0 and shrink the test set; keep S0, R0 and G−ind.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Gemini refuses or hedges on diagnostic questions | Starters: record the refusal rate as a Q1 finding. G: decision-support framing in its prompt |
| Starter models retired | Nearest successor, documented |
| CheXbert mislabels free text | Same labeller path for all systems in the starter comparison; manual check of ~15–20 outputs per system |
| Too few positives per finding | Positive-enriched test set; keep only findings with ≥ ~30 positives |
| API cost / rate limits | Pilot set first, caching, a few hundred test studies |
| Cohere account/cost for P2 | Price it in week 1; if replaced, rename to P2-adapted |
| S0 trained on IU X-Ray | Check the weights' training sets in week 1 (§3) |
| Retrieval leaks test cases | Patient (or study) split + ID assertions + embedding screen (§2) |
| No patient IDs, so repeat patients cross splits | Week-1 check; repeat-patient screen; week-7 robustness check (§6) |

---

## 8. Threats to validity (closing consideration)

**Gemini has probably seen IU X-Ray.** The dataset has been public since ~2015, so it is likely
in Gemini's training data. Memorised report text is plausible; recalling a specific image's report
is less so. This semester has no clean test for it, so it is discussed, not ruled out:
- **Direction of the bias:** only G receives the test study's *own* text (its indication), the
  one input that could prompt recall of that study's report. Indications are short and generic,
  so this is a weak route. Memorised images, if any, would help every Gemini system that sees the
  image (P1, P2, G, G−ind). The likely net effect is a modest push toward "image + text wins",
  and any memorisation favours all Gemini systems over S0, which does not use Gemini. The Q3
  write-up states this caveat next to the G vs. S0, G vs. G−ind and G vs. P1 numbers, not only
  here.
- **After-the-fact check (no extra API calls):** compare each system's free text with the test
  study's *own* report, which it never saw. Baseline: overlap with other reports carrying the same
  labels, since IU reports are heavily templated. Overlap with its own report clearly above that
  baseline points to memorisation.
- **Early warning:** if any Gemini system scores implausibly high on the pilot set in week 2,
  look at contamination then, not in week 8.
- A clean test (local model on credentialed data) is left to the thesis.

Other threats: a study-level split, if the release has no patient IDs (repeat patients may cross
splits; see the week-1 check and the robustness check); report-derived labels are noisy (CheXbert); frontal-only input; the positive-enriched
test set; the starters' possibly substituted model versions.

---

## 9. Questions for the supervisor

1. Is this semester scope (starters → grounded system → comparison) right, with the full study
   left for the thesis?
2. Is a hosted API (Gemini, Cohere) acceptable? Any API budget?
3. Is chest X-ray an acceptable modality?

## 10. Immediate next steps

1. Download IU X-Ray; check whether it has a patient ID field; write the loader and the
   train/val/test split (by patient if possible, otherwise by study).
2. Get a Google AI Studio key; check that the starters' model IDs are still served.
3. Run P1 (web search off) on ~10 X-rays to see the raw output format.
4. Install TorchXRayVision + BiomedCLIP; confirm they run on the RTX 4060; check which datasets
   each TorchXRayVision weights file was trained on.
