# G — design (decided)

Status: **decided 2026-10-18 (all recommendations D1–D7 adopted).** Summarised in
`results.md` §4.4 and the diary; this file is the full decided version.

The plan (§4) fixes G's outline: the patient's frontal image → BiomedCLIP → top-k similar training
cases (image + report); the image, the indication and the retrieved reports → Gemini (pinned
model) → per-finding JSON with confidence, cited case IDs and rationale, plus a short impression.
Q2 (`results.md` §4) says what to keep, drop and add. This proposal turns that into concrete
choices.

---

## A. Settled by the evidence (no decision needed)

| # | Choice | Evidence |
|---|---|---|
| A1 | **Index:** BiomedCLIP, centre crop, train split = R0's existing index (`biomedclip_frontal.npy`); queries use the same file's validation / test rows, so **no new embedding** is needed | Q2 §4.2: BiomedCLIP beats Gemini Embedding 2 at prompt-sized k; crop vs. pad makes no difference |
| A2 | **Retrieve first, then one direct generation call** (google-genai, metered client); no agent, no tools, no web search | P1 tool coupling, P3 agent loop (3 calls, runaways), agno hidden retries (§4.1) |
| A3 | **The patient's image reaches the model**, together with the neighbours' **reports** | The main gap of P3; P2 passes images without reports (§4.1) |
| A4 | **No output cap** | P3's cap, which includes thinking, cut answers (§3.4) |
| A5 | **Temperature: default (1.0)** | Google: "we strongly recommend keeping the temperature … at its default value of 1.0 … below 1.0 may lead to … looping". P1 and P2 also run at the default |
| A6 | **Model:** `gemini-3-flash-preview`, the starters' model | Holds the model constant across all Gemini systems (§3.1) |
| A7 | **Every response cached in full** (raw text, model version, usage), every call metered, USD 10 per-run limit | Plan §4; spend controls (§3) |
| A8 | **Leakage guards:** index = train only; assert no query study is in the index | Plan §2; as in R0, P2, P3 |

---

## B. Decisions needed

### D1 — How many neighbours (k)?

| Option | For | Against |
|---|---|---|
| **k = 10, fixed now (recommended)** | R0: k = 5 → 0.651, k = 10 → 0.671, k = 20 → 0.704 (more neighbours carry more signal); 10 reports ≈ 600 tokens ≈ $0.0003 per study | The plan says "choice of k on validation"; fixing it is a deviation |
| k = 5 (P3 used 6) | Shorter prompt, easier to cite | Weaker evidence |
| Tune k on validation (plan) | As planned | Each extra k value on 400 validation studies ≈ $3.6 per arm; with G and G−ind, ≈ $7 per k value: not affordable within the cap |

k also sets the copying-rate analysis (plan §5): the neighbours' majority vote is computed at
G's k.

### D2 — Define the six findings in G's prompt?

**Recommended: yes**, with the ground truth's definitions (`results.md` §2.4b):
- *Lung Opacity:* airspace opacity, consolidation, infiltrate or airspace disease in the lung;
  **not** a discrete nodule or mass.
- *Lung Lesion:* a nodule, mass or cavity in the lung; **not** a calcified granuloma.
- *Cardiomegaly:* an enlarged heart; **borderline or upper-normal heart size counts as absent**.
- *Atelectasis, Pleural Effusion:* as named.
- *Fracture:* any rib, clavicle, spine or other bony fracture, **healed fractures included**.

For: removes the Lung Opacity / Lung Lesion overlap the hand check found (§3.5) and states the
task precisely. Against: the starters do not get these definitions, so G vs. a starter also
differs in task specification. That is acceptable for Q3, which measures how much the results
differ, not why (plan §5), and is stated as a threat to validity.

### D3 — Output format and score

**Recommended:** JSON enforced by the API's response schema (no free-text parsing):

```json
{
  "findings": [
    {"finding": "Cardiomegaly", "status": "present | absent | uncertain",
     "probability": 0.0-1.0, "cited_case_ids": ["CXR123", "..."], "rationale": "one or two sentences"}
  ],
  "impression": "short free-text summary"
}
```

- **Starter comparison (binary):** status "present" → 1; absent / uncertain → 0, the same rule as
  the starters' checklist (§3.5).
- **Anchored comparison (vs. S0, R0):** `probability` gives AUROC; thresholded metrics use one
  pooled Youden threshold on validation, as decided on 2026-10-16.
- A response that fails the schema is recorded and counted (like an unparsable checklist).

### D4 — Thinking level

| Option | For | Against |
|---|---|---|
| **Default (`high`), recommended** | Same as P1, P2 and P3: the model's reasoning is held constant across systems, so differences come from design | Cost and runaway risk (P3: 6% runaways, but P3 ran at temperature 0.25, which Google warns can cause looping; P1 and P2 at 1.0 had none) |
| `medium` or `low` | Cheaper | G would reason less than the starters (a confound); untested |

Safeguards with the default: the USD 10 per-run limit; the pilot measures thinking and cost
before any validation or test run.

### D5 — What the prompt says

**Recommended outline** (wording drafted and checked on the pilot):
1. Role and framing: decision support for a radiologist; no refusal needed (plan §4 allows a
   decision-support framing; refusals were 0 in the pilots anyway).
2. The six findings with their definitions (D2).
3. Read the **patient's** image systematically first.
4. Then consider the k similar cases: they are **other patients**, similar-looking but weak
   evidence; use them to support or question what the image shows, **not** as the answer; cite
   the cases that support each call.
5. The clinical indication: `Clinical indication: <text>` (G−ind: this line is omitted;
   nothing else changes).
6. Output as in D3; rationales short; no textbook explanations.

Neighbours are shown as `CXR id · similarity · report (FINDINGS + IMPRESSION)`; their images are
**not** sent (≈ 1,100 tokens each, and a retrieved image without its report adds no label; §4.1).

### D6 — Prompt development and freezing

**Recommended:** develop on the 50-study pilot only; at most 3 prompt versions, each recorded
with its pilot outcome (schema compliance, cost, behaviour); **freeze** the prompt before any
validation run. Validation then sets G's pooled threshold; test is run once (plan §6).

### D7 — Optional arms

| Arm | What it shows | Cost | Recommendation |
|---|---|---|---|
| **G-img** (G's model, prompt and output, image only: no indication, no neighbours) | What retrieval + indication add *within the same design* (G vs. G-img), the cleanest image-only vs. multimodal contrast | test only ≈ $2.5 (AUROC and stated status need no threshold) + pilot ≈ $0.3 | **Yes**, test-only |
| G+S0 (S0's probabilities in the prompt) | Whether Gemini adds anything on top of the specialist | ≈ $7 (validation + test) | **No** (plan's buffer rule: dropped first) |

---

## C. Cost estimate

Per G study (from the pilots: one image ≈ 1,100 input tokens; P2-like output and thinking):
≈ 2,500 input + ≈ 2,500 output and thinking tokens → **≈ $0.009**.

| Run | Studies | Estimate |
|---|---|---|
| Prompt development (≤ 3 versions × 50 pilot studies, G only) | ≤ 150 | ≈ $1.4 |
| G and G−ind on validation | 800 | ≈ $7.2 |
| G and G−ind on test | 800 | ≈ $7.2 |
| G-img: pilot + test | 450 | ≈ $3 |
| **G family total** | | **≈ $19** |

Project total with the starters' test runs: ≈ $3.9 spent + $14.6 starters + $19 G ≈ **$37.5
(83% of the cap)**. Re-projected after G's first pilot.

---

## D. Prompt development (D6) — prompt 2 frozen, 2026-10-18

| | Prompt 1 | **Prompt 2 (frozen)** |
|---|---|---|
| Change | — | cases listed by ID without position numbers ("Case CXR316 (similarity …): …"); two sentences added to the output step: cite by ID exactly as written; two-decimal probabilities, intermediate values for subtle or borderline appearances |
| Valid JSON, all six findings | 50 / 50 | 50 / 50 |
| Answers with invalid citations | 3 (cited list positions) | **0** |
| Distinct probability values / share of the most common (0.05) | 13 / 66% | 19 / 57% |
| Findings called / correct / normal studies called normal | 8 / 4 / 40 of 42 | 10 / 4 / 39 of 42 |
| Binary agreement with prompt 1 | — | 296 / 300 (within run-to-run noise at temperature 1.0) |
| Cost per study; max thinking | $0.010; 5,525 | $0.0105; 6,943 |

Version 3 was not needed. `scripts/g/g_system.py` defaults to prompt 2 and refuses any other
version outside pilot runs.
