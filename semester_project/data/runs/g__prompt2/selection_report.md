## g__prompt2 — val (400 studies)

- Answers: 400 valid JSON, 0 invalid; 0 with validation problems; 0 missing statuses
- Thinking tokens per answer: median 1979, mean 2396, max 7903; > 20,000: 0
- **Pooled threshold 0.0800** (Youden J over 2400 pairs, 113 positive): sensitivity 0.64, specificity 0.82
- Threshold reading vs. stated status disagree on 377 of 2400 pairs

| Finding | Positives | AUROC [95% CI] | Sens. @thr | Spec. @thr | F1 @thr | Sens. stated | Spec. stated | F1 stated |
|---|---|---|---|---|---|---|---|---|
| Lung Opacity | 39 | 0.703 [0.611, 0.798] | 0.51 | 0.86 | 0.37 | 0.31 | 0.95 | 0.35 |
| Cardiomegaly | 23 | 0.932 [0.878, 0.971] | 0.96 | 0.76 | 0.32 | 0.48 | 0.97 | 0.49 |
| Atelectasis | 26 | 0.678 [0.546, 0.809] | 0.58 | 0.64 | 0.17 | 0.50 | 0.92 | 0.38 |
| Lung Lesion | 11 | 0.700 [0.542, 0.855] | 0.45 | 0.86 | 0.14 | 0.18 | 0.98 | 0.21 |
| Pleural Effusion | 12 | 0.893 [0.796, 0.980] | 0.75 | 0.91 | 0.33 | 0.33 | 1.00 | 0.47 |
| Fracture | 2 | 0.713 [0.489, 0.938] | 0.50 | 0.88 | 0.04 | 0.00 | 0.99 | 0.00 |
| **Macro** | | **0.770** [0.701, 0.837] | 0.63 | 0.82 | 0.23 | 0.30 | 0.97 | 0.32 |

wrote C:\Users\antal\Desktop\Project 4\semester_project\data\runs\g__prompt2\selection.json

