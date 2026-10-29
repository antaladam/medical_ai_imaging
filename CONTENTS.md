# MSc project #4: Multimodal, explainable AI decision support for chest X-rays

Does a multimodal, retrieval-augmented system beat a single image model for chest-X-ray decision
support? Decision support only; public, de-identified data only.

## Contents

| Path | What it is |
|---|---|
| `README.md` | The assigned project brief, headed with the author's name |
| `main.txt` | The assigned project brief, as received |
| `thesis_plan.md` | Long-term thesis plan (v2); `thesis_plan_v1_2026-10-13.md` is the earlier version |
| `codebase_analysis.md` | Analysis of the three starter projects |
| `starter_ai_agents/`, `rag_tutorials/` | The three starter projects, as received |
| `semester_project/semester_plan.md` | This semester's plan (frozen; deviations are recorded, not edited in) |
| `semester_project/results.md` | The report in progress, including all deviations from the plan |
| `semester_project/diary.md` | Log of every change and decision |
| `semester_project/scripts/` | Data preparation, labelling, splits, starter runners, tests |
| `semester_project/requirements.txt` | Pinned Python package versions |
| `semester_project/presentation/` | Semester progress review, self-contained HTML: `progress_review.html` (13 slides) and `progress_review_using_template.html` (the same slides in the faculty's NIK template). `old/` holds the earlier 18-slide English and Hungarian versions |

**Setting up on a new computer:** see [`SETUP.md`](SETUP.md).

## Not in the repository

- `semester_project/data/`: the IU X-Ray dataset (Open-i, CC BY-NC-ND 4.0) and everything derived
  from it. `SETUP.md` explains how to download it and regenerate (or copy) the derived files.
- `.env`: the Gemini API key (`GOOGLE_API_KEY=...`), local only.
- `.venv/`, `.cache/`: local Python environment and model caches.

## Third-party code

`starter_ai_agents/` and `rag_tutorials/` come from
[Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps); see that
repository for their licence. The CheXbert labeller in `semester_project/scripts/data_prep/chexbert_labels.py`
follows CheXbert (Smit et al. 2020) as packaged in `f1chexbert` (MIT).
