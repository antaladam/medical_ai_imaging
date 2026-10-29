"""Embed the selected frontal images with BiomedCLIP and look for near-duplicate pairs.

Writes data/processed/biomedclip_frontal.npy (L2-normalised image embeddings, one row per study
in frontal.csv order) and biomedclip_frontal_ids.csv, prints the similarity distribution of each
image's nearest other image, and saves a side-by-side sheet of the most similar pairs.

Usage: python -I scripts/data_prep/embed_and_dedup.py <processed_dir> <images_dir>
"""
import csv
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
TOP_PAIRS, THUMB = 30, 140


@torch.no_grad()
def embed(paths, device):
    model, preprocess = open_clip.create_model_from_pretrained(MODEL)
    model = model.to(device).eval()
    out = []
    for i in range(0, len(paths), 64):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + 64]]).to(device)
        f = model.encode_image(x)
        out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    return torch.cat(out).numpy()


def main(proc: Path, image_dir: Path) -> None:
    with open(proc / "frontal.csv") as fh:
        rows = [(r["study"], r["image"]) for r in csv.DictReader(fh)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = embed([image_dir / f"{img}.png" for _, img in rows], device)
    np.save(proc / "biomedclip_frontal.npy", emb)
    with open(proc / "biomedclip_frontal_ids.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "study", "image"])
        w.writerows((i, s, img) for i, (s, img) in enumerate(rows))

    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    nn = sim.max(1)
    print(f"Embedded {len(rows)} images, dim {emb.shape[1]}")
    print("Nearest-neighbour similarity percentiles 50/90/99/99.9/max:",
          np.percentile(nn, [50, 90, 99, 99.9, 100]).round(4))
    for t in (0.99, 0.98, 0.97, 0.95):
        print(f"  pairs with similarity >= {t}: {int((np.triu(sim, 1) >= t).sum())}")

    iu = np.triu_indices(len(rows), 1)
    vals = sim[iu]
    top = np.argsort(-vals)[:TOP_PAIRS]
    pairs = [(rows[iu[0][k]], rows[iu[1][k]], float(vals[k])) for k in top]
    with open(proc / "dedup_top_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "study_a", "image_a", "study_b", "image_b", "similarity"])
        w.writerows((k, a[0], a[1], b[0], b[1], round(s, 4)) for k, (a, b, s) in enumerate(pairs))

    cols = 5  # 5 pairs per row
    canvas = Image.new("RGB", (cols * 2 * THUMB + cols * 10, (TOP_PAIRS // cols) * (THUMB + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (a, b, s) in enumerate(pairs):
        x0, y = (k % cols) * (2 * THUMB + 10), (k // cols) * (THUMB + 18)
        for j, (_, img) in enumerate((a, b)):
            im = Image.open(image_dir / f"{img}.png").convert("RGB")
            im.thumbnail((THUMB, THUMB))
            canvas.paste(im, (x0 + j * THUMB, y))
        draw.text((x0 + 2, y + THUMB + 2), f"{k}: {a[0]}/{b[0]} {s:.3f}", fill="black")
    canvas.save(proc / "dedup_top_pairs.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
