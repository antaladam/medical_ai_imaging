# Thesis Plan — Multimodal, Explainable Decision Support for Chest X-rays

Plan for the MSc topic in `main.txt`. It builds on the gaps listed in `codebase_analysis.md`.
Drafted 2026-10-13. **Assumptions to confirm with the supervisor are marked ⚠.**

---

## 1. Scope in one paragraph

Build a chest-X-ray decision-support prototype that outputs structured findings, a confidence for each
finding, and an explanation whose claims cite retrieved evidence (similar prior cases and their
reports). Use a public dataset to compare four input conditions, from image-only to image + clinical
text + retrieval. Measure five things: finding accuracy, calibration, grounding/verifiability,
explanation quality and failure modes. The system is decision support only. It never produces a
final diagnosis, and it only uses public, de-identified data.

**Research question (from the brief):** Can grounded multimodal retrieval improve the usefulness and
verifiability of LLM-generated explanations for medical-image decision support?

**Sub-questions**
- RQ1: Does adding clinical text (the indication / history) improve finding accuracy over image-only input?
- RQ2: Does retrieving similar cases improve accuracy and calibration, or does it mainly copy retrieved labels (anchoring)?
- RQ3: What share of explanation claims are actually supported by the evidence they cite, and does retrieval raise that share?
- RQ4: Where does the system fail, and can its confidence or retrieval similarity predict those failures (selective prediction)?

---

## 2. Data

| Dataset | Why | Access |
|---|---|---|
| **IU X-Ray (Open-i)** — ~3.9k studies / ~7.4k images with reports *and* an "indication" field | Primary: has image + clinical text + report, and is fully public | Open download ✅ |
| **MIMIC-CXR** (+ MIMIC-CXR-JPG) | Larger, more realistic; used for external validation if accessible | PhysioNet credentialing + CITI training ⚠; its licence forbids sending data to third-party APIs, so only local models can be used on it |
| **CheXpert Plus** (images + reports) | Alternative large set if MIMIC is not accessible | Stanford registration ⚠ (verify licence) |
| NIH ChestX-ray14 / CheXpert | Image-only label sets for training/checking the local classifier baseline | Public ✅ |

**Ground truth:** the 14 CheXpert-style findings, extracted from the reports with a labeler
(CheXbert, or the MIMIC-provided labels), plus a small manually checked subset.
**Splits:** split by patient. The retrieval index is built from the training split only, and the
same patient is never retrieved for itself, to avoid leakage.
**Contamination risk:** hosted LLMs have probably seen IU X-Ray reports. Note this as a threat to
validity, and use MIMIC/CheXpert Plus with local models as a cleaner check.

---

## 3. System architecture

```
 image ─┬─► [A] local CXR classifier (TorchXRayVision) ──► per-finding probabilities (calibrated baseline)
        ├─► [B] image embedder (BiomedCLIP / MedImageInsight-type) ─► kNN over train-split images ─┐
 indication text ─► [C] text embedder ─► kNN over train-split reports ─────────────────────────────┤
                                                                                                    ▼
                              [D] VLM reasoner (Gemini API  and/or  local MedGemma-4B, 4-bit)
                                  input: image (+ indication) (+ top-k retrieved cases)
                                  output: JSON {finding, present?, confidence, claims[{text, cited_case_ids}]}
                                                                                                    ▼
                              [E] verifier: does each claim follow from its cited evidence? (NLI model / LLM-judge)
                                                                                                    ▼
                              [F] logger → eval harness (metrics, plots, failure gallery)
                                                                                                    ▼
                              [G] Streamlit UI: image, findings + confidence, explanation with clickable citations
```

What comes from the starter repos:
- **Project 1** (`ai_medical_imaging_agent`): the prompt structure and the Streamlit pattern → becomes the *image-only arm* and the UI shell.
  Fix the temp-file race and the DICOM crash (`codebase_analysis.md`).
- **Project 2** (`vision_rag`): the embed → retrieve → generate pattern. Replace Cohere with a local medical embedder so retrieval runs offline and patient data never leaves the machine.
- **Project 3** (`multimodal_agentic_rag`): the structured citation objects + similarity scores + "hedge on weak evidence" prompting → the *grounded arm*. The FastAPI/React/3D view is optional; don't let it eat the time budget.

**Hardware:** RTX 4060 8 GB. TorchXRayVision, BiomedCLIP and an NLI model run comfortably.
MedGemma-4B should fit with 4-bit quantization (⚠ verify the memory need and the HAI-DEF licence terms).
Use the Gemini API for the main runs on IU X-Ray, and the local model for MIMIC and for the
privacy-preserving configuration.

---

## 4. Experimental conditions

| Arm | Input to the reasoner |
|---|---|
| A1 | image only |
| A2 | image + indication text |
| A3 | image + top-k retrieved similar cases |
| A4 | image + indication + top-k retrieved cases (full system) |
| B0 | local classifier alone (non-LLM baseline) |

**Ablations (run on A4):** k ∈ {1, 3, 5, 10}; image- vs text- vs blended retrieval; with/without the
verifier; Gemini vs MedGemma; **shuffled/random retrieval control** (tests whether gains come from
relevant evidence or just from more context).

---

## 5. Evaluation (maps to the brief's five axes)

| Axis | Metric |
|---|---|
| Image interpretation | Per-finding F1 / AUROC (macro + micro) vs report-derived labels; also report-generation metrics (RadGraph F1 / CheXbert F1) if free-text reports are produced |
| Calibration | ECE, Brier score, reliability diagrams. Compare three confidence sources: verbalized confidence, sampling consistency (n=5–10 samples), and classifier probability. Optionally temperature/Platt scaling on a validation split |
| Retrieval grounding | Label overlap of retrieved cases with the query (precision@k), and the anchoring rate (how often the output just copies the retrieved labels) |
| Explanation verifiability | Citation precision/recall: share of claims entailed by their cited evidence (automatic NLI/LLM-judge, validated against a manual audit of ~100 claims, Cohen's κ) |
| Explanation usefulness | Small rubric rating (correctness, completeness, clarity, appropriate hedging) on ~50 cases. Ideally 1–2 clinicians ⚠, otherwise the author + LLM-judge, with that limitation stated |
| Failure cases | Taxonomy (missed finding, hallucinated finding, laterality error, anchoring on wrong retrieval, overconfidence) + risk-coverage curves (accuracy if the system abstains on low-confidence cases) |

Statistics: paired bootstrap CIs between arms; McNemar's test on per-case correctness.

---

## 6. Safety & data-protection chapter (required by the brief)

- Intended use: second reader / decision support only; human in the loop; no autonomous diagnosis.
- Data flow diagram: which data goes to which API. Public data only to hosted APIs; credentialed data only to local models.
- GDPR / EU AI Act framing (a medical-device AI system would be high-risk under the Act) and the MDR — at the level of positioning, not a legal analysis.
- Prototype safeguards: abstain below a confidence threshold, an always-visible disclaimer, no image persistence beyond the session, per-session temp files.

---

## 7. Timeline (⚠ assumes ~two semesters to submission; compress phases 3–5 if one semester)

| Weeks | Phase | Deliverable |
|---|---|---|
| 1–3 | **Literature review + setup** — medical VLMs (MedGemma, LLaVA-Med, CheXagent), retrieval-augmented report generation, LLM calibration, citation/attribution evaluation. Apply for PhysioNet now (it takes weeks) | Lit-review draft; API keys; datasets downloaded |
| 4–5 | **Data pipeline** — IU X-Ray loader, patient-level split, label extraction, embeddings + index | Reproducible data module + label stats |
| 6–7 | **Baselines** — B0 classifier, A1 image-only VLM with JSON output | First numbers; supervisor check-in |
| 8–10 | **Grounded system** — retrieval arms A2–A4, citations, verifier | Full pipeline running end to end on the val set |
| 11–13 | **Evaluation harness + main experiments** — all arms, calibration, ablations | Results tables + plots |
| 14–15 | **Failure analysis + human rating** | Failure taxonomy, rating study |
| 16–17 | **Local model / MIMIC replication** (if access was granted) | External-validation results |
| 18–19 | **Prototype UI + safety chapter** | Demo app, data-flow documentation |
| 20–24 | **Writing** (+ optional workshop paper, e.g. a MICCAI / MIDL / ML4H workshop) | Thesis draft → final |

Write as you go: methods notes after each phase, so the final writing is mostly editing.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| PhysioNet access delayed or denied | IU X-Ray is enough for the core thesis; MIMIC is a bonus |
| API cost / rate limits / model deprecation | Cache every response with the exact model ID; budget a pilot run first; pin model versions |
| Labeler noise in the ground truth | Manually audited subset; report label noise as a limitation |
| No clinician raters | Automatic + author ratings, clearly flagged; ask the supervisor about medical-school contacts early |
| Scope creep (React UI, 3D view, multi-turn chat) | Streamlit only; extras only after week 19 |
| Dataset contamination in hosted LLMs | Local-model + MIMIC replication; report it as a threat to validity |

---

## 9. Questions for the supervisor

1. Submission date / is this one semester or thesis-spanning?
2. Is chest X-ray an acceptable modality focus, or is a different imaging domain expected?
3. Can the department sponsor PhysioNet credentialing? Any API budget or GPU server?
4. Any clinician contacts for the explanation-rating study?
5. Is a hosted API (Gemini) acceptable for the main experiments, or must everything run locally?

## 10. Immediate next steps (this week)

1. Start PhysioNet credentialing (CITI course).
2. Get a Google AI Studio key; run Project 1 on a few public X-rays to see the raw behavior.
3. Download IU X-Ray; write the loader + patient split.
4. Install TorchXRayVision + BiomedCLIP locally; confirm they run on the 4060.
5. Start the literature table (paper, task, dataset, retrieval?, calibration?, grounding metric?).
