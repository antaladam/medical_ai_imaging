## g_ind__prompt2 — val (400 studies)

- Answers: 400 valid JSON, 0 invalid; 0 with validation problems; 0 missing statuses
- Thinking tokens per answer: median 1967, mean 2514, max 8263; > 20,000: 0
- **Pooled threshold 0.0800** (Youden J over 2400 pairs, 113 positive): sensitivity 0.59, specificity 0.85
- Threshold reading vs. stated status disagree on 307 of 2400 pairs

| Finding | Positives | AUROC [95% CI] | Sens. @thr | Spec. @thr | F1 @thr | Sens. stated | Spec. stated | F1 stated |
|---|---|---|---|---|---|---|---|---|
| Lung Opacity | 39 | 0.701 [0.611, 0.794] | 0.46 | 0.87 | 0.34 | 0.28 | 0.96 | 0.34 |
| Cardiomegaly | 23 | 0.916 [0.856, 0.965] | 0.91 | 0.80 | 0.35 | 0.48 | 0.97 | 0.47 |
| Atelectasis | 26 | 0.689 [0.557, 0.820] | 0.65 | 0.69 | 0.21 | 0.42 | 0.93 | 0.35 |
| Lung Lesion | 11 | 0.622 [0.448, 0.801] | 0.36 | 0.86 | 0.11 | 0.09 | 0.98 | 0.10 |
| Pleural Effusion | 12 | 0.840 [0.691, 0.959] | 0.58 | 0.92 | 0.29 | 0.33 | 0.99 | 0.44 |
| Fracture | 2 | 0.573 [0.550, 0.597] | 0.00 | 0.95 | 0.00 | 0.00 | 0.99 | 0.00 |
| **Macro** | | **0.724** [0.678, 0.782] | 0.50 | 0.85 | 0.22 | 0.27 | 0.97 | 0.28 |

wrote C:\Users\antal\Desktop\Project 4\semester_project\data\runs\g_ind__prompt2\selection.json

