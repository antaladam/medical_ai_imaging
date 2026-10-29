"""Q2: which embedding retrieves the most informative neighbours? (design: diary 2026-10-18)

R0's leave-one-out kNN vote on the train split (no API calls; validation and test untouched),
for five embeddings:
  biomedclip_crop  - BiomedCLIP, its own preprocessing (centre crop); R0's index
  biomedclip_pad   - BiomedCLIP, image padded to a square (black) first; computed here once
                     and cached as data/processed/biomedclip_frontal_pad.npy
  gemini3072_image - Gemini Embedding 2, 3,072-d, image (P2-adapted's index)
  gemini768_image  - Gemini Embedding 2, 768-d, image (P3's index, image part)
  p3_blend         - P3's index (image + report, 0.68 / 0.32), queried by the 768-d image
Reports macro / per-finding AUROC per k, and a paired bootstrap CI of each embedding's macro
AUROC minus biomedclip_crop's at k = 5 and k = 150. Writes data/runs/q2/retrieval_compare.md.

Usage: python -I scripts/evaluation/q2_retrieval_compare.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
import metrics  # noqa: E402
from common import FINDINGS  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
PROC = PROJECT / "data" / "processed"
RUNS = PROJECT / "data" / "runs"
IMAGES = PROJECT / "data" / "raw" / "iu_xray" / "images"
K_GRID = [1, 5, 10, 50, 150]
N_BOOT, SEED = 2000, 2026


def biomedclip_pad() -> np.ndarray:
    """BiomedCLIP embeddings of the frontal images, padded to square first (frontal.csv order)."""
    path = PROC / "biomedclip_frontal_pad.npy"
    if path.exists():
        return np.load(path)
    import open_clip
    import torch
    from PIL import Image
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    with open(PROC / "biomedclip_frontal_ids.csv") as fh:
        images = [r["image"] for r in csv.DictReader(fh)]

    def padded(name):
        im = Image.open(IMAGES / f"{name}.png").convert("RGB")
        side = max(im.size)
        sq = Image.new("RGB", (side, side), (0, 0, 0))
        sq.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
        return preprocess(sq)  # resize to 224; the centre crop is now a no-op

    out = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            x = torch.stack([padded(n) for n in images[i:i + 64]]).to(device)
            f = model.encode_image(x)
            out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
            print(f"padded embeddings {min(i + 64, len(images))}/{len(images)}", end="\r")
    print()
    emb = torch.cat(out).numpy()
    np.save(path, emb)
    return emb


def loo_scores(query: np.ndarray, index: np.ndarray, y: np.ndarray, kmax: int) -> np.ndarray:
    """Neighbour-label shares for every k up to kmax; a study never retrieves itself."""
    sim = query @ index.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1, kind="stable")[:, :kmax]
    assert not (nn == np.arange(len(y))[:, None]).any()
    return y[nn]  # (n, kmax, findings)


def macro(y, s, idx=slice(None)):
    return np.nanmean([metrics.auroc(y[idx, j], s[idx, j]) for j in range(len(FINDINGS))])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with open(PROC / "splits.csv") as fh:
        train = sorted((r["study"] for r in csv.DictReader(fh) if r["split"] == "train"), key=lambda s: int(s[3:]))
    with open(PROC / "ground_truth.csv") as fh:
        gt = {r["study"]: [r[f] == "1" for f in FINDINGS] for r in csv.DictReader(fh)}
    y = np.array([gt[s] for s in train])
    with open(PROC / "biomedclip_frontal_ids.csv") as fh:
        row = {r["study"]: int(r["row"]) for r in csv.DictReader(fh)}

    def norm(a):
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    crop = np.load(PROC / "biomedclip_frontal.npy")[[row[s] for s in train]]
    pad = biomedclip_pad()[[row[s] for s in train]]
    g3072 = np.stack([np.load(RUNS / "p2_adapted" / "embeddings" / f"{s}.npy") for s in train])
    p3 = [np.load(RUNS / "p3_orig" / "index" / f"{s}.npz") for s in train]
    g768 = np.stack([z["media"] for z in p3])
    blend = norm(0.68 * g768 + 0.32 * np.stack([z["annotation"] for z in p3]))  # rag_store._blend_vectors
    variants = {"biomedclip_crop": (crop, crop), "biomedclip_pad": (pad, pad),
                "gemini3072_image": (norm(g3072), norm(g3072)), "gemini768_image": (norm(g768), norm(g768)),
                "p3_blend": (norm(g768), blend)}

    lab = {name: loo_scores(q, ix, y, max(K_GRID)) for name, (q, ix) in variants.items()}
    lines = [f"# Q2 retrieval comparison (train, leave-one-out, {len(train)} studies)\n",
             "Macro AUROC of the neighbours' vote (share of the k nearest train studies with each finding).\n",
             "| Embedding | " + " | ".join(f"k={k}" for k in K_GRID) + " |", "|---|" + "---|" * len(K_GRID)]
    for name, L in lab.items():
        lines.append(f"| {name} | " + " | ".join(f"{macro(y, L[:, :k].mean(1)):.3f}" for k in K_GRID) + " |")

    for k in (5, 150):
        lines += [f"\n## k = {k}: per finding, and difference to biomedclip_crop (paired bootstrap 95% CI)\n",
                  "| Embedding | " + " | ".join(FINDINGS) + " | Macro | Δ vs crop [95% CI] |",
                  "|---|" + "---|" * (len(FINDINGS) + 2)]
        base = lab["biomedclip_crop"][:, :k].mean(1)
        for name, L in lab.items():
            s = L[:, :k].mean(1)
            per = [metrics.auroc(y[:, j], s[:, j]) for j in range(len(FINDINGS))]
            if name == "biomedclip_crop":
                d = "—"
            else:
                diffs = metrics.bootstrap(lambda i: macro(y, s, i) - macro(y, base, i), len(train), N_BOOT, SEED)
                lo, hi = metrics.ci(diffs)
                d = f"{np.mean(per) - macro(y, base):+.3f} [{lo:+.3f}, {hi:+.3f}]"
            lines.append(f"| {name} | " + " | ".join(f"{a:.3f}" for a in per) + f" | {np.mean(per):.3f} | {d} |")

    out = RUNS / "q2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
