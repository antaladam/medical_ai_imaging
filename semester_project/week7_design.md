# Week 7 — analysis design (decided)

Status: **decided 2026-10-18 (W1–W3 adopted as recommended).** To-do step 5. All three analyses use the
cached test-set outputs; no API calls. The test results are final (results §5.5), so looking at
individual test cases now cannot influence any tuned choice.

The plan fixes the three analyses in outline: a failure review of ~20 hand-reviewed errors with a
taxonomy (plan §5); an after-the-fact memorisation check comparing free text with the study's own
report against same-label reports (plan §8); a hand check of G's statuses against its rationales
(as done for the starters' checklists, results §3.5). This fixes the details before any case is
looked at.

---

## A. Settled (no decision needed)

| # | Choice | Source |
|---|---|---|
| A1 | Test set only, cached outputs only, no API | plan §8 ("no extra API calls"), results §5.5 |
| A2 | First-pass judgements by Claude, an AI assistant, recorded as such; sheets and judgements kept in `data/runs/review/`; samples drawn by a script with seed 2026 | as the starters' hand check (results §3.5) |
| A3 | **Judged from text, not from images:** the study's own report (FINDINGS, IMPRESSION), its indication, the system's answer, and for G the neighbours' reports. No radiologist is available; an AI reading the X-rays would add an unvalidated second reader | limitation stated in the write-up |

---

## B. Decisions needed

### W1 — Failure review: which errors, which categories

| Option | For | Against |
|---|---|---|
| **Stratified, seeded sample of 23 errors from `per_study.csv` (recommended)**, see below | Covers every system and the error types the analysis questions are about; reproducible | 23 cases describe patterns, they do not estimate rates |
| ~20 errors drawn at random from all systems | Simplest | Dominated by the starters' and S0's many false alarms; may miss G's characteristic errors |

Recommended sample (stated verdicts; G's threshold reading for the G-vs-S0 strata, as AUROC
was the primary comparison):

| Stratum | n |
|---|---|
| G misses (truth present, G absent / uncertain) | 4 |
| G false alarms | 4 |
| G wrong where S0 is right, and S0 wrong where G is right (threshold readings) | 2 + 2 |
| G copied a wrong neighbours' decision while G-img (no neighbours) was right (E3) | 3 |
| Each starter: 1 miss + 1 false alarm | 6 |
| G-img's two invented-citation answers (CXR206, CXR3080) | 2 |
| **Total** | **23** |

Categories, fixed now (one primary per error, "other" allowed): **missed finding** (the report
describes it; the answer does not mention it); **invented finding** (the answer asserts something
the report does not describe); **definition mismatch** (e.g. a nodule called Lung Opacity, a
borderline heart called enlarged; the ground-truth coding differs from the answer's reading);
**copied neighbour** (the answer's reason is the retrieved cases, and they are wrong for this
patient); **text-led** (the indication drives the call); **hedged** (the finding is mentioned as
possible / uncertain and scored absent); **label noise** (the report describes the finding but the
MeSH code is missing, or the reverse). Laterality is dropped from the plan's list: the labels have
no side, so it cannot produce a scored error.

### W2 — Memorisation check (plan §8)

*Question:* does a system's text match the study's **own** report more than other reports with
the same labels? If so, it may recall the IU report rather than read the case.

| Choice | Recommended | Alternatives and why not |
|---|---|---|
| Systems | All six Gemini systems: P1-ns, P2, P3, G, G−ind, G-img (S0, R0 produce no text) | — |
| System text | G family: the impression + all rationales; starters: the answer without the six checklist lines | Only the impression: too short for the starters' comparison |
| Reference | The report's **FINDINGS + IMPRESSION**, **without INDICATION** (G received the indication, so including it would count G's own input as "recall") | Whole report: biased towards G |
| Normalisation | lower case, "XXXX" de-identification tokens and punctuation removed | — |
| Overlap measure | **Primary: containment of word 3-grams** = share of the reference report's 3-grams that appear in the system text. Secondary: unigram cosine (TF-IDF over all reports) | Exact sentence match: too strict; embedding similarity: measures meaning, not recall of wording |
| Baseline | For each test study: the same measure against **20 other reports with the identical six-label vector** (seed 2026, drawn from all splits, excluding the study itself); baseline = their mean | Random reports: IU reports are templated, so any normal report overlaps any normal answer; same-label reports control for that |
| Statistic | Per system: mean (own − baseline) with a paired bootstrap 95% CI over studies; plus the share of studies whose own-report overlap exceeds the **maximum** of its 20 same-label reports (chance level ≈ 1/21 ≈ 5%) | — |
| Reading | Own ≈ baseline and share ≈ 5%: no sign of memorisation. A clear excess, **above all in G-img** (image only, no text cue), would point to recall from the image; an excess only in G vs. G−ind would point to the indication route | — |

Known limit: a negative result shows no *verbatim* recall; it cannot rule out memorised labels
without memorised wording.

### W3 — Hand check of G: status vs. rationale

| Option | For | Against |
|---|---|---|
| **15 G test answers, 8 with at least one "present" (seed 2026), 90 finding judgements: does the status agree with its own rationale (agree / disagree / unclear)? Plus, per "present" call, the rationale's basis: image / cited cases / indication / mixed (recommended)** | Same protocol as the starters (comparable); the basis label addresses "did G reason from the evidence it cites" (the open limit of E4) | Basis is read from G's wording, which may not reflect how it actually decided |
| Status vs. rationale only | Exactly as the starters | Leaves the E4 limit open |

---

## C. Outputs

`scripts/evaluation/week7_samples.py` draws the W1 and W3 samples and writes review sheets to
`data/runs/review/` (each case in full: report, indication, answer, neighbours for G);
`scripts/evaluation/memorisation.py` computes W2 (with offline tests on synthetic text). Results
in results §5.6 (new); judgements in `data/runs/review/*.csv`.
