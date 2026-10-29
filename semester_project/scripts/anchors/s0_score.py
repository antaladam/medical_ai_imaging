"""S0: score frontal images with TorchXRayVision models (no training).

Preprocessing follows the library's documented pipeline: channel mean of the 8-bit image,
xrv.datasets.normalize(img, 255) (-> [-1024, 1024]), make square, XRayResizer to the model's
resolution (224 for the DenseNets, 512 for the ResNet). "Make square" has two variants, both
scored (diary 2026-10-17):
  crop: XRayCenterCrop, as documented (cuts the long side to the short one);
  pad:  pad the short side symmetrically with -1024 (black = air), so nothing is cut off.

Writes data/runs/s0/<weights>__<prep>__<split>.csv: study + sigmoid probability for every
trained output of the model. Weights are loaded by scripts/anchors/s0_weights.py (safe loading).
The test split is refused unless --test is given (plan §6: test is run once, in week 6); on
test only the selected configuration (selection.json) is scored.

Usage: python -I scripts/anchors/s0_score.py <split>[,<split>...] [--test]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
sys.path.insert(0, str(Path(__file__).resolve().parent))  # anchors/: s0_weights
import s0_weights  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
PROC = PROJECT / "data" / "processed"
IMAGES = PROJECT / "data" / "raw" / "iu_xray" / "images"
OUT = PROJECT / "data" / "runs" / "s0"
WEIGHTS = ["densenet121-res224-chex", "densenet121-res224-mimic_nb", "densenet121-res224-mimic_ch",
           "resnet50-res512-all"]
PREPS = ["crop", "pad"]
BATCH = 32


def to_square(img: np.ndarray, prep: str) -> np.ndarray:
    """img: (1, H, W) in [-1024, 1024]."""
    if prep == "crop":
        return xrv.datasets.XRayCenterCrop()(img)
    _, h, w = img.shape
    d = abs(h - w)
    pad = ((0, 0), (d // 2, d - d // 2), (0, 0)) if h < w else ((0, 0), (0, 0), (d // 2, d - d // 2))
    return np.pad(img, pad, constant_values=-1024.0)


def load_image(name: str) -> np.ndarray:
    img = np.asarray(Image.open(IMAGES / f"{name}.png"), dtype=np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(2)  # RGB copies of a greyscale image (results §2.1)
    return xrv.datasets.normalize(img, 255)[None, ...]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("splits")
    ap.add_argument("--test", action="store_true", help="allow scoring the test split (week 6)")
    args = ap.parse_args()
    splits = args.splits.split(",")
    if "test" in splits and not args.test:
        sys.exit("refusing to score the test split without --test (plan §6: once, in week 6)")
    weights, preps = WEIGHTS, PREPS
    if "test" in splits:
        # Only the selected S0 is scored on test (results "Resolved decisions" #1): the other
        # configurations, the ResNet-50 reference included, never get test scores.
        assert splits == ["test"], "score the test split on its own"
        sel = json.loads((OUT / "selection.json").read_text())
        weights, preps = [sel["weights"]], [sel["prep"]]

    with open(PROC / "frontal.csv") as fh:
        image_of = {r["study"]: r["image"] for r in csv.DictReader(fh)}
    with open(PROC / "splits.csv") as fh:
        split_of = {r["study"]: r["split"] for r in csv.DictReader(fh)}

    # Reproducible scores: cuDNN uses TF32 for convolutions by default on this GPU, which made
    # a study's probability depend on its batch by up to ~1e-4 (diary 2026-10-17).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {w: s0_weights.load(w).to(device) for w in weights}
    res = {w: 512 if w.startswith("resnet") else 224 for w in weights}
    resizers = {r: xrv.datasets.XRayResizer(r) for r in set(res.values())}
    OUT.mkdir(parents=True, exist_ok=True)

    for split in splits:
        studies = sorted((s for s, sp in split_of.items() if sp == split), key=lambda s: int(s[3:]))
        assert studies and all(s in image_of for s in studies), f"split {split}: missing images"
        rows = {(w, p): [] for w in weights for p in preps}
        for i in range(0, len(studies), BATCH):
            batch = studies[i:i + BATCH]
            imgs = [load_image(image_of[s]) for s in batch]
            for p in preps:
                sq = [to_square(im, p) for im in imgs]
                assert all(x.shape[1] == x.shape[2] for x in sq)
                for r, resize in resizers.items():
                    x = torch.from_numpy(np.stack([resize(im) for im in sq])).to(device)
                    with torch.no_grad():
                        for w in weights:
                            if res[w] == r:
                                prob = torch.sigmoid(models[w](x)).cpu().numpy()
                                rows[(w, p)].extend(zip(batch, prob))
            print(f"{split}: {min(i + BATCH, len(studies))}/{len(studies)}", end="\r")
        print()
        for (w, p), out in rows.items():
            keep = [j for j, t in enumerate(models[w].targets) if t]
            with open(OUT / f"{w}__{p}__{split}.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["study"] + [models[w].targets[j] for j in keep])
                wr.writerows([s] + [f"{pr[j]:.6f}" for j in keep] for s, pr in out)
            print(f"wrote {w}__{p}__{split}.csv ({len(out)} studies)")


if __name__ == "__main__":
    main()
