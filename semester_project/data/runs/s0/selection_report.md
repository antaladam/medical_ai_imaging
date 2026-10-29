## Train (2804 studies) — AUROC per config

| Weights | Prep | Lung Opacity | Cardiomegaly | Atelectasis | Lung Lesion | Pleural Effusion | Fracture | Macro | Δ macro vs. selected [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| DenseNet chex | crop | 0.714 | 0.807 | 0.721 | 0.490 | 0.822 | 0.592 | **0.691** | -0.037 [-0.059, -0.014] |
| DenseNet chex | pad | 0.696 | 0.809 | 0.706 | 0.511 | 0.820 | 0.558 | **0.683** | -0.044 [-0.066, -0.023] |
| DenseNet mimic_nb | crop | 0.743 | 0.789 | 0.733 | 0.557 | 0.842 | 0.616 | **0.713** | -0.014 [-0.026, -0.002] |
| DenseNet mimic_nb | pad | 0.743 | 0.798 | 0.734 | 0.547 | 0.868 | 0.676 | **0.728** | selected |
| DenseNet mimic_ch | crop | 0.712 | 0.784 | 0.742 | 0.554 | 0.832 | 0.639 | **0.711** | -0.017 [-0.033, -0.001] |
| DenseNet mimic_ch | pad | 0.705 | 0.779 | 0.734 | 0.552 | 0.860 | 0.639 | **0.712** | -0.016 [-0.034, +0.003] |
| resnet50-res512-all (reference) | crop | 0.723 | 0.894 | 0.811 | 0.689 | 0.939 | 0.715 | **0.795** | +0.067 [+0.040, +0.093] |
| resnet50-res512-all (reference) | pad | 0.730 | 0.913 | 0.819 | 0.697 | 0.943 | 0.746 | **0.808** | +0.080 [+0.052, +0.106] |

## Selected S0 = densenet121-res224-mimic_nb / pad — train (2804 studies) — thresholds tuned here (in-sample)

| Finding | Positives | AUROC [95% CI] | Threshold | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| Lung Opacity | 396 | 0.743 [0.715, 0.770] | 0.2108 | 0.64 | 0.77 | 0.42 |
| Cardiomegaly | 206 | 0.798 [0.767, 0.826] | 0.1509 | 0.79 | 0.66 | 0.26 |
| Atelectasis | 221 | 0.734 [0.702, 0.765] | 0.1373 | 0.78 | 0.56 | 0.23 |
| Lung Lesion | 49 | 0.547 [0.465, 0.627] | 0.0072 | 0.71 | 0.38 | 0.04 |
| Pleural Effusion | 83 | 0.868 [0.827, 0.907] | 0.2262 | 0.83 | 0.76 | 0.17 |
| Fracture | 43 | 0.676 [0.600, 0.748] | 0.0078 | 0.60 | 0.69 | 0.06 |
| **Macro** | | **0.728** [0.706, 0.749] | | 0.73 | 0.64 | 0.19 |

## Selected S0 = densenet121-res224-mimic_nb / pad — val (400 studies)

| Finding | Positives | AUROC [95% CI] | Threshold | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| Lung Opacity | 39 | 0.716 [0.615, 0.806] | 0.2108 | 0.51 | 0.81 | 0.31 |
| Cardiomegaly | 23 | 0.868 [0.790, 0.929] | 0.1509 | 0.87 | 0.70 | 0.26 |
| Atelectasis | 26 | 0.672 [0.545, 0.783] | 0.1373 | 0.69 | 0.60 | 0.19 |
| Lung Lesion | 11 | 0.482 [0.320, 0.653] | 0.0072 | 0.64 | 0.40 | 0.06 |
| Pleural Effusion | 12 | 0.899 [0.843, 0.951] | 0.2262 | 0.92 | 0.81 | 0.22 |
| Fracture | 2 | 0.906 [0.847, 0.955] | 0.0078 | 1.00 | 0.76 | 0.04 |
| **Macro** | | **0.757** [0.695, 0.802] | | 0.77 | 0.68 | 0.18 |

wrote C:\Users\antal\Desktop\Project 4\semester_project\data\runs\s0\selection.json
