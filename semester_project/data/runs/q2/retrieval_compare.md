# Q2 retrieval comparison (train, leave-one-out, 2804 studies)

Macro AUROC of the neighbours' vote (share of the k nearest train studies with each finding).

| Embedding | k=1 | k=5 | k=10 | k=50 | k=150 |
|---|---|---|---|---|---|
| biomedclip_crop | 0.572 | 0.651 | 0.671 | 0.712 | 0.741 |
| biomedclip_pad | 0.584 | 0.655 | 0.680 | 0.732 | 0.731 |
| gemini3072_image | 0.529 | 0.574 | 0.600 | 0.667 | 0.720 |
| gemini768_image | 0.531 | 0.576 | 0.601 | 0.664 | 0.712 |
| p3_blend | 0.550 | 0.595 | 0.635 | 0.703 | 0.737 |

## k = 5: per finding, and difference to biomedclip_crop (paired bootstrap 95% CI)

| Embedding | Lung Opacity | Cardiomegaly | Atelectasis | Lung Lesion | Pleural Effusion | Fracture | Macro | Δ vs crop [95% CI] |
|---|---|---|---|---|---|---|---|---|
| biomedclip_crop | 0.691 | 0.787 | 0.643 | 0.519 | 0.757 | 0.510 | 0.651 | — |
| biomedclip_pad | 0.703 | 0.750 | 0.654 | 0.555 | 0.756 | 0.510 | 0.655 | +0.004 [-0.012, +0.020] |
| gemini3072_image | 0.608 | 0.708 | 0.560 | 0.503 | 0.585 | 0.483 | 0.574 | -0.077 [-0.100, -0.054] |
| gemini768_image | 0.607 | 0.708 | 0.561 | 0.505 | 0.595 | 0.483 | 0.576 | -0.075 [-0.098, -0.052] |
| p3_blend | 0.638 | 0.720 | 0.595 | 0.513 | 0.617 | 0.484 | 0.595 | -0.056 [-0.079, -0.034] |

## k = 150: per finding, and difference to biomedclip_crop (paired bootstrap 95% CI)

| Embedding | Lung Opacity | Cardiomegaly | Atelectasis | Lung Lesion | Pleural Effusion | Fracture | Macro | Δ vs crop [95% CI] |
|---|---|---|---|---|---|---|---|---|
| biomedclip_crop | 0.792 | 0.891 | 0.747 | 0.555 | 0.898 | 0.565 | 0.741 | — |
| biomedclip_pad | 0.775 | 0.876 | 0.759 | 0.533 | 0.892 | 0.553 | 0.731 | -0.010 [-0.031, +0.013] |
| gemini3072_image | 0.730 | 0.864 | 0.735 | 0.600 | 0.804 | 0.589 | 0.720 | -0.021 [-0.049, +0.008] |
| gemini768_image | 0.727 | 0.861 | 0.735 | 0.596 | 0.802 | 0.550 | 0.712 | -0.029 [-0.058, -0.001] |
| p3_blend | 0.745 | 0.875 | 0.749 | 0.636 | 0.847 | 0.572 | 0.737 | -0.004 [-0.033, +0.024] |
