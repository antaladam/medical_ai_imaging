# Thesis Plan — Does Multimodal / RAG Context Beat a Single Image Model? (Chest X-ray)

Plan for the MSc topic in `main.txt`. It builds on the gaps listed in `codebase_analysis.md`.
v2, revised 2026-10-16. v1, which centred on building an explainable prototype, is kept in
`thesis_plan_v1_2026-10-13.md`. **Assumptions to confirm with the supervisor are marked ⚠.**

---

## 1. Scope in one paragraph

This is a **comparative, hypothesis-testing study**, not a product build. A strong single
image-recognition model is the reference point. Against it, the study tests whether systems that add
clinical text, retrieved similar cases, and an LLM/VLM reasoning layer actually do better, and if
they do, *why*. Components are added one at a time so each one's contribution can be measured on
its own. A null or negative result ("the specialist image model is as good or better") is a valid
outcome and gets reported as such. The prototype exists only as the instrument for the
experiments. Everything is decision support only and uses public, de-identified data.

**Main question:** Is a multimodal, retrieval-augmented system more effective than a single
image-recognition model for chest-X-ray decision support? "Effective" means more accurate, better
calibrated and easier to verify.

**Hypotheses** (stated before running the experiments; each can be rejected)
- **H1 (accuracy):** Adding clinical text and/or retrieved cases improves per-finding accuracy
  (macro AUROC) over the image-only specialist model.
- **H2 (calibration):** The multimodal/RAG system's confidence is better calibrated (lower ECE/Brier)
  than the image model's probabilities.
- **H3 (verifiability):** Evidence-cited explanations (retrieved cases + reports) are more
  verifiable than the image model's saliency-map explanations.
- **H4 (mechanism):** Any gains come from *relevant* evidence. They should disappear under random
  retrieval or corrupted text, and should not come from label leakage in the clinical text or from
  copying retrieved labels.

**Secondary, descriptive question (no hypothesis):** how do the three assigned starter apps
(`main.txt`), run nearly as shipped, compare with the specialist model and with the improved
systems? See §4.1.

**What "understanding the models" means here:** for every system, find out what signal it
actually uses. That includes image vs. text vs. retrieved labels, where it fails, and whether
its confidence can be trusted. Probing experiments (§5) answer this, not just headline scores.

---

## 2. Background to master (lit review, weeks 1–3)

Required before the comparison is credible:
- **Specialist image classifiers:** CNN/ViT chest-X-ray classifiers (CheXNet line, TorchXRayVision),
  multi-label training on report-derived labels, known label-noise and shortcut-learning problems.
- **Vision–language embedders:** CLIP-style contrastive models (BiomedCLIP, CheXzero-type
  zero-shot), and how image-to-image and image-to-text retrieval work.
- **Generative VLMs:** general (Gemini) and medical (MedGemma, LLaVA-Med, CheXagent). Covers
  evidence on how well they classify compared with specialists, and on whether they really use
  the image.
- **Retrieval-augmented generation for radiology:** retrieval-augmented report generation, kNN
  label transfer.
- **Evaluation:** calibration (ECE, reliability, selective prediction), saliency-map evaluation
  (sanity checks, pointing game), citation and attribution faithfulness.

Deliverable: a literature table (paper, model type, dataset, image-only vs. multimodal comparison?,
result) that sets the **prior expectation** for each hypothesis.

---

## 3. Data

| Dataset | Role | Access |
|---|---|---|
| **IU X-Ray (Open-i)**, ~3.9k studies with reports and an "indication" field | Primary: image + clinical text + report, fully public | Open ✅ |
| **NIH ChestX-ray14** (incl. ~1k bounding boxes) / **CheXpert** | Training/validating the image model; bounding boxes for testing saliency localization | Public ✅ |
| **MIMIC-CXR(-JPG)** | External replication with local models only (licence bars third-party APIs) | PhysioNet credentialing ⚠ |

**Labels:** CheXpert-style findings extracted from the reports (CheXbert), plus a manually
audited subset. The analysis may restrict to the 5–8 findings with enough positives in IU X-Ray.
**Splits:** split by patient. The retrieval index uses the training split only, and a patient is
never retrieved for themselves.
**Text-leakage check:** the indication field can contain the answer ("follow-up pneumonia").
Before using it, tag each indication as *neutral*, *suggestive*, or *answer-leaking*, and report
results per tag. Otherwise any "multimodal gain" could just be the system reading the answer.
**Contamination:** hosted LLMs have probably seen IU X-Ray. This is a threat to validity, so a
local model replication on MIMIC serves as a check.

---

## 4. Systems compared (a ladder: one change per step)

| ID | System | Isolates |
|---|---|---|
| **S0** | **Specialist image model**: TorchXRayVision DenseNet, optionally fine-tuned on the IU training split, explained with Grad-CAM. ⚠ Use weights not trained on Open-i (= IU X-Ray); the combined "all" weights may include it | The reference point |
| S0+T | S0 image features + text-embedding of the indication → small fused classifier (late fusion, no LLM) | Text value *without* an LLM |
| R0 | kNN label transfer: retrieve top-k similar training images (BiomedCLIP), vote their labels; no LLM | Retrieval value *without* an LLM |
| S1 | VLM, image only | The LLM layer vs. the specialist |
| S2 | VLM + indication text | Adding text |
| S3 | VLM + top-k retrieved cases (images + reports) | Adding retrieval |
| S4 | VLM + text + retrieval | Full multimodal RAG |
| S5 | VLM + text + retrieval + **S0's probabilities as a tool input** | Whether the LLM adds value *on top of* the specialist (hybrid) |

VLMs: Gemini API (main, IU X-Ray) and MedGemma-4B locally, 4-bit on the RTX 4060 ⚠ (check memory
and licence). Model IDs are pinned and every response is cached.

**Fairness rules:** same test cases and label set for every system; decision thresholds tuned on
the same validation split; every ladder system gets a continuous score (for VLMs: verbalized
probability or the frequency across n=5–10 samples) so AUROC is comparable. The starter arms in
§4.1 produce free text only and are scored on binary metrics.

### 4.1 Role of the starter projects

The three starter repos from `main.txt` are used in two ways: as **near-as-shipped systems under
test** (a separate block, not rungs of the ladder) and as **sources of components** for the
improved ladder systems. Their Streamlit/React UIs are not used for the experiments; each one is
driven by a batch runner that calls its core code.

| ID | Starter | Kept as shipped | Minimum adaptation (and why) | What it effectively tests |
|---|---|---|---|---|
| **P1-orig** | `ai_medical_imaging_agent` | Model ID from the code (`gemini-2.5-pro`), agno agent, the 5-section prompt | Batch runner instead of the UI. DuckDuckGo tool **off** for the main run, because live web search makes results irreproducible; a 50-case sensitivity run with it on | An off-the-shelf image-only LLM app |
| **P2-orig** | `vision_rag` | Cohere `embed-v4.0`, top-1 cosine retrieval, `gemini-2.5-flash`, the original prompt | (a) Query by the patient's image instead of a text question: image→image retrieval over the IU training images. (b) Also pass the query image to Gemini. As shipped, only the *retrieved* image goes to the model, so it would describe a different patient | One retrieved neighbour image, no report text |
| **P3-orig** | `multimodal_agentic_rag` | Gemini Embedding 2, image + annotation blend at weight 0.32 (annotation = the training case's report), top_k = 6, the ADK agent prompt, `gemini-3-flash-preview` | Query by image embedding instead of text (`search()` only embeds text); `rag_store` and the agent are called directly, without the server/UI | **LLM reasoning over neighbours' reports.** The generator only receives retrieved text, never the patient's image (`server.py:217-219`, `retrieval_payload`). So this is R0 + an LLM, *not* S4 |

All three receive the same fixed question ("Which of these findings are present: …"). None
gets the indication text, because P1 has no input for it.

**Scoring the starter arms:** their free-text output is turned into finding labels with the same
CheXbert labeler used for the ground truth, checked on ~50 manually labelled outputs (P1's
patient-friendly section and differential list may confuse a labeler trained on reports).
Metrics are F1 and sensitivity/specificity only. No AUROC or calibration, because they produce no
per-finding score. P1's prose "confidence level" is reported as an exploratory note only. No
claim-level verifiability either: P1 has no citations, P2 shows one source image, and P3 cites
per answer, with a prompt that forbids inline IDs.

**What this block can and cannot conclude:**
- *Can:* how assigned, ready-made LLM health apps perform against a specialist model, and the
  total gap between them and the improved systems.
- *Cannot:* attribute that gap to a single design choice. Going from as-shipped to improved
  changes the model, embedder, k, prompt and output format at once. Attribution is the ladder's
  job (§4). The three starters also use three different Gemini versions, so P1 vs. P2 vs. P3 is
  not a design comparison.
- *Bridge (optional, cheap):* re-run each P*-orig with the ladder's VLM. This separates "older
  model" from "weaker design".

**Components carried into the ladder:** P1's prompt sections, adapted to JSON output, become S1's
prompt. P2's embed → retrieve → generate pattern becomes S3 (with a medical embedder and top-k).
P3's citation objects, similarity scores and weak-evidence hedging instruction become S3–S5's
output schema, and its 0.32 blend weight becomes an ablation. P1's Streamlit shell is the base
for the week-19 demo; its temp-file race and DICOM crash (`codebase_analysis.md`) need fixing
only there.

**Correction to `codebase_analysis.md`:** it lists P3 as the best basis for the image+text arm.
Its *retrieval* is multimodal, but its *generator* is text-only, so it can't serve as S4 without
adding the image to the agent's input.

---

## 5. Experiments

**E1 — Main comparison (H1, H2):** all ladder systems on the test set, with the starter arms
(§4.1) in a separate table on binary metrics. Per-finding and macro AUROC, F1 at
the tuned threshold, ECE/Brier and reliability diagrams.

**E2 — Mechanism probes (H4, "understanding the models"):**
- *Image dependence:* give the VLM a blank or mismatched image. If performance barely drops, it is
  not really using the image.
- *Text dependence:* remove, shuffle across patients, or neutralise the indication. Report results
  separately for neutral vs. leaking indications.
- *Retrieval relevance:* random retrieval and deliberately wrong-label retrieval. Measure the
  **anchoring rate** (how often the output copies the retrieved labels even when they are wrong).
- *Retrieval quality vs. gain:* is per-case improvement correlated with retrieval precision@k?
- *k sweep* (1, 3, 5, 10) and image vs. text vs. blended retrieval (incl. P3's 0.32 blend weight).
- *Embedder:* R0 and S3 with BiomedCLIP vs. the starters' general-purpose embedders (Gemini
  Embedding 2, Cohere embed-v4). IU X-Ray only, because the latter are hosted APIs.

**E3 — Explanation verifiability (H3):**
- S0: Grad-CAM localization vs. NIH bounding boxes (pointing game / IoU) plus a randomization
  sanity check.
- S3–S5: share of explanation claims supported by the cited evidence (NLI/LLM-judge, validated
  on ~100 manually audited claims, Cohen's κ).
- A small rubric rating of both explanation types on ~50 cases (clinicians ⚠, else author +
  LLM-judge, stated as a limitation).

**E4 — Failure analysis:** a taxonomy (missed finding, hallucinated finding, laterality, anchoring
on wrong retrieval, text-led error, overconfidence). Risk–coverage curves: which system degrades
more gracefully when allowed to abstain? Includes cases where S0 is right and S4 is wrong, and
the reverse.

**E5 — Replication (if access):** S0, R0, S1, S4 with local MedGemma on MIMIC-CXR. The starter
arms are excluded, because they depend on hosted APIs that the MIMIC licence rules out.

**Statistics:** DeLong / paired bootstrap for AUROC differences, McNemar for per-case correctness,
Holm correction across findings. A **minimum meaningful difference** (e.g. ΔAUROC ≥ 0.02 with the
CI excluding 0) is fixed in advance, so "effective" is decided by a pre-stated criterion, not
after the fact.

---

## 6. Expected outcomes (to be stated honestly, not assumed)

Plausible results, all of them reportable:
1. The specialist S0 matches or beats generalist VLMs on pure classification, while RAG/text add
   verifiability but not accuracy.
2. Apparent multimodal gains shrink sharply on neutral indications, meaning leakage drove them.
3. The hybrid S5 is best, because the LLM adds value only when it is grounded on the specialist's
   output.

Each of these answers the brief's research question either way.

---

## 7. Safety & data-protection chapter (required by the brief)

- Intended use: second reader / decision support; human in the loop; no autonomous diagnosis.
- Data flow: public data only to hosted APIs; credentialed data only to local models.
- GDPR / EU AI Act (high-risk) / MDR positioning, not a legal analysis.
- What the experiments imply for safety, e.g. the anchoring rate and the behaviour of
  overconfident failures, as evidence for or against deploying such systems.

---

## 8. Timeline (⚠ assumes ~two semesters; compress E3–E5 if one semester)

| Weeks | Phase | Deliverable |
|---|---|---|
| 1–3 | Literature review (§2) + setup; apply for PhysioNet | Lit table with prior expectations; hypotheses + thresholds fixed |
| 4–5 | Data pipeline: IU loader, patient split, labels, indication-leakage tagging, embedding index | Data module + label/leakage stats |
| 6–7 | S0 (+ fine-tune), S0+T, R0 | Non-LLM baselines: first numbers; supervisor check-in |
| 8–10 | VLM systems S1–S5 with cached, structured JSON output | All ladder systems running on the validation set |
| 11 | Starter arms P1–P3-orig (batch runners, minimum adaptations) + output labeler check | Starter arms running; labeler validated on ~50 outputs |
| 12–14 | E1 + E2 (main comparison + mechanism probes) | Core results tables and plots |
| 15–16 | E3 + E4 (explanations, failure analysis) | Verifiability results, failure gallery |
| 17–18 | E5 MIMIC replication (if access) | External validation |
| 19 | Minimal demo on P1's Streamlit shell (side-by-side S0 vs. S4) + safety chapter | Demo, data-flow documentation |
| 20–24 | Writing (+ optional workshop paper: MIDL / ML4H / MICCAI workshop) | Thesis |

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| VLMs give poor continuous scores, so AUROC is unfair to them | Use sampling frequency, and also report threshold metrics (F1, sensitivity/specificity) |
| Few positives per finding in IU X-Ray | Restrict to well-populated findings; report CIs; add MIMIC if available |
| Indication leakage inflates multimodal gains | Leakage tagging + neutral-only analysis (§3) |
| PhysioNet delayed or denied | IU + NIH are enough for the core thesis |
| API cost, model deprecation | Cache responses, pin model IDs, pilot on 50 cases first |
| Scope creep into app building | The UI is a week-19 extra only |
| Starters' hard-coded models retired (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3-flash-preview`) | Check availability in week 1; if a model is gone, use the nearest successor and document the substitution (this weakens the "as shipped" claim) |
| Starter adaptations drift into redesign | Only the adaptations listed in §4.1; each one recorded with its reason |
| CheXbert mislabels free-text app output | Manual check of ~50 outputs; report labeler agreement |
| Extra API accounts/costs (Cohere for P2) | Price the embedding of the IU training images in week 1; drop P2-orig's Cohere embedder for Gemini Embedding 2 if needed, and say so |

---

## 10. Questions for the supervisor

1. Is this framing (a comparative study with a specialist model as the reference) acceptable,
   given the brief's emphasis on "explainable prototype"? The prototype becomes the experimental
   instrument.
2. Is using the starter projects as near-as-shipped systems under test plus a component source
   (§4.1) what was intended, or do you expect one of them to be extended into the final system?
3. Submission date: one semester, or spanning the thesis?
4. Is chest X-ray an acceptable modality? Department support for PhysioNet, API budget or GPU?
5. Any clinician contacts for the explanation rating (E3)?
6. Is a hosted API acceptable for the main experiments, or must everything run locally?

## 11. Immediate next steps

1. Start PhysioNet credentialing (CITI course).
2. Download IU X-Ray; write the loader and patient split; inspect the indication field for leakage.
3. Run TorchXRayVision zero-shot on IU X-Ray to get the first S0 numbers (a reality check on the bar
   to beat).
4. Run P1 (web search off) on ~10 IU X-Ray images to see the raw output format, and check whether
   the three starters' hard-coded Gemini model IDs are still served.
5. Start the literature table, focused on existing specialist-vs-VLM and with/without-RAG comparisons.
