# Evaluation — design (decided)

Status: **decided 2026-10-18 (all recommendations E1–E5 adopted).** Summarised with full
reasoning in `results.md` §5.4 and the diary; this file keeps the option tables.

To-do step 3: one evaluation script for the test set, written and checked on validation data before any test run (plan §6: test once).

The plan (§5) fixes what is measured: per-finding and macro F1 / sensitivity / specificity for
all systems; AUROC where there are scores; precision@k for the retrieval systems; G's copying
rate; G's citation label agreement; G vs. G−ind; paired bootstrap CIs over studies. Since the
plan: starters and G are scored on their own stated verdicts, not CheXbert (results §3.5);
G-img added, G+S0 dropped (g_design.md D7). This proposal fixes the details the plan leaves open.

Out of scope here (week 7, to-do step 5): the ~20-case failure review, the memorisation check
(plan §8), the hand check of G's statuses against its rationales.

---

## A. Settled by earlier decisions (no decision needed)

| # | Choice | Source |
|---|---|---|
| A1 | **Ground truth:** MeSH-derived labels (`ground_truth.csv`), six findings | results §2.4b |
| A2 | **Binary reading per system:** starters = their checklist ("present" → 1; absent, uncertain, missing → 0); G, G−ind, G-img = stated status, same rule; S0, R0 = score ≥ their train-tuned per-finding thresholds; G, G−ind also = probability ≥ their pooled validation threshold (0.08) | results §3.5, §5.1–5.3 |
| A3 | **Scores for AUROC:** S0, R0, G, G−ind, G-img (stated probabilities). G-img has no threshold (validation run not made, D7) | g_design.md D7 |
| A4 | **Macro** = unweighted mean over the six findings | results §5 |
| A5 | **Bootstrap:** 2,000 resamples of studies, seed 2026, paired (same resample for both systems in a difference), percentile 95% CI; `metrics.py` | results §5 |
| A6 | **Unanswered / unparsable:** counted per system and reported; scored as all-negative (not dropped), so every system is scored on the same 400 studies | results §3.5 |
| A7 | **Test split only with `--test`**; on validation the script uses whatever studies each system has (starters, G-img: 50 pilot studies) and compares systems on their common studies; on test it asserts all 400 for every system | plan §6 |

---

## B. Decisions needed

### E1 — Which differences get confidence intervals (declared before test)

The plan asks for paired bootstrap CIs "for the differences" without listing them. Declaring
the list now keeps the test analysis from becoming a search for significant pairs.

| Option | For | Against |
|---|---|---|
| **Fixed list, one primary (recommended)** — see below | The thesis question gets one pre-declared answer; the rest is labelled secondary | Secondary comparisons still read as tests; stated as descriptive |
| All pairs of all systems (8 systems, 28 pairs × metrics) | Complete | Many comparisons; some will "differ" by chance |
| No CIs on differences, only per-system CIs | Simple | Plan §5 asks for paired CIs |

Recommended list (difference = first − second, macro over six findings):

| # | Comparison | Metric(s) | Question |
|---|---|---|---|
| **1 (primary)** | **G − S0** | macro AUROC | Does image + retrieval + text beat the specialist image model? (the thesis question) |
| 2 | G − R0 | macro AUROC | Does the LLM add anything over retrieval alone? |
| 3 | G − G−ind | macro AUROC; macro F1 (stated) | Effect of the clinical indication (plan §5) |
| 4 | G − G-img | macro AUROC; macro F1 (stated) | Effect of retrieval + indication within one design |
| 5 | G-img − S0 | macro AUROC | Image-only LLM vs. image-only specialist |
| 6–8 | G − P1-ns, G − P2, G − P3 | macro F1, sensitivity, specificity (stated) | Built system vs. each starter (Q3) |

Per-finding differences are shown in tables with CIs, but not singled out. No correction for
multiple comparisons; the write-up states that only #1 is the pre-declared main comparison.

### E2 — Retrieval quality: definition of precision@k

The plan: "share of retrieved training cases that share the query's labels". With six rare
findings, "share all labels" mostly means "both normal", which rewards retrieving normals.

| Option | For | Against |
|---|---|---|
| **Per finding, on queries that have it (recommended):** for each finding f and each test study positive for f, the share of its k neighbours that also have f; averaged per finding. Reported next to f's train prevalence (what random retrieval gives), as lift = precision / prevalence | Measures whether retrieval finds cases with *the same disease*, which is what the LLM is meant to use; prevalence makes it interpretable | Normal queries not covered (added as a separate row: share of normal neighbours for normal queries) |
| Exact label-vector match | Literal reading of the plan | Dominated by normal–normal matches |
| Jaccard of label sets | Partial credit | Undefined for normal–normal; hard to read |

**k:** each system at its own k (P2 1, P3 6, G 10, R0 150) and all at **k = 1** (the one depth
every system has). G and R0 share the index, so R0 at k = 10 is exactly G's neighbours; it is not
reported separately.

### E3 — Copying rate (G vs. its neighbours)

The plan: "how often G's answer equals the neighbours' majority vote, especially when that vote
is wrong". A strict majority (≥ 6 of 10 neighbours) is almost never positive for these rare
findings (R0's thresholds are vote shares of 0.01–0.10, §5.2), so it would measure little more
than G's specificity.

| Option | For | Against |
|---|---|---|
| **kNN decision at k = 10 + G-img as control (recommended):** the neighbours' decision = R0's rule at k = 10 (vote share ≥ a per-finding Youden threshold, tuned by leave-one-out on train, like R0; no API). Report how often G's binary equals that decision, overall and where the decision is wrong; **the same for G-img**, which never saw the neighbours. G minus G-img = how much the neighbours pull G | G-img separates "G follows the neighbours" from "G and the neighbours read the same image the same way"; uses an existing rule | One extra threshold set (train only, cheap) |
| Strict majority vote (≥ 6 / 10) | Plan's wording | Nearly always "absent"; uninformative |
| Any positive neighbour | Simple | Very common for frequent findings |

Also reported, descriptive: G's "present" rate by number of positive neighbours (0, 1, 2, 3+),
for G and G-img side by side.

### E4 — Citation label agreement

| Option | For | Against |
|---|---|---|
| **For each of G's "present" calls: share of its cited cases that have that finding; pooled per finding, with a baseline = share of all 10 neighbours that have it (recommended)** | Baseline shows whether G cites selectively or just cites whatever it was given. Also count "present" calls with no citation | — |
| Without a baseline | As plan | Cannot tell selective citation from random |

Same for G−ind. Absent calls with citations are counted but not scored.

### E5 — Outputs and tests

| Option | For | Against |
|---|---|---|
| **One script `scripts/evaluation/evaluate.py <split> [--test]` (recommended):** reads every system's existing tables and caches (no API); writes `data/runs/eval/<split>/report.md` (all tables) and `results.json`; offline tests `scripts/tests/test_evaluate.py` on small synthetic data with hand-computed answers (precision@k, copying, citation agreement, common-study alignment) | One command reproduces the whole Q3 analysis; tests before real data | — |
| One script per analysis | Smaller files | Alignment of studies across systems repeated |

Run on validation first (S0, R0, G, G−ind on 400; starters and G-img on the 50 pilot studies) to
check it end to end; those numbers are not a result.

---

## C. Cost

No API calls. Test-set runs (to-do step 4) are unchanged.
