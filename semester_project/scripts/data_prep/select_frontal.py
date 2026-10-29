"""Select one frontal image per IU X-Ray study.

Rule (from the by-eye checks recorded in results.md §2.4):
  1. Drop unusable images: near-blank (grey std < 15 on a 128 px copy) or noise
     (full-resolution high-frequency/contrast ratio > 0.2).
  2. Per study, take the image with the highest BiomedCLIP p_frontal.
  3. Accept it if p_frontal >= 0.05; otherwise the study has no frontal image and is excluded.
Writes data/processed/frontal.csv and a contact sheet of all accepted images with p_frontal < 0.5,
plus a sample of excluded studies' best images, for checking by eye.

Usage: python -I scripts/data_prep/select_frontal.py <processed_dir> <images_dir>
"""
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

BLANK_STD, NOISE_RATIO, MIN_P = 15.0, 0.2, 0.05
THUMB, COLS, SEED = 150, 10, 0


def sheet(items, image_dir: Path, out: Path) -> None:
    rows = max(1, (len(items) + COLS - 1) // COLS)
    canvas = Image.new("RGB", (COLS * THUMB, rows * (THUMB + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (img, pf) in enumerate(items):
        im = Image.open(image_dir / f"{img}.png").convert("RGB")
        im.thumbnail((THUMB, THUMB))
        x, y = (k % COLS) * THUMB, (k // COLS) * (THUMB + 18)
        canvas.paste(im, (x, y))
        draw.text((x + 2, y + THUMB + 2), f"{k}: {pf:.2f}", fill="black")
    canvas.save(out)


def main(proc: Path, image_dir: Path) -> None:
    with open(proc / "views.csv") as fh:
        rows = [(r["image"], r["study"], float(r["p_frontal"])) for r in csv.DictReader(fh)]
    quality = json.load(open(proc / "image_quality.json"))      # image -> [std, diff]
    hf_ratio = json.load(open(proc / "image_hf_ratio.json"))    # image -> ratio

    unusable = {img for img, _, _ in rows
                if quality[img][0] < BLANK_STD or hf_ratio[img] > NOISE_RATIO}
    by_study = {}
    for img, study, pf in rows:
        if img not in unusable:
            by_study.setdefault(study, []).append((pf, img))

    selected, excluded = [], []
    for study, imgs in by_study.items():
        pf, img = max(imgs)
        (selected if pf >= MIN_P else excluded).append((study, img, pf))

    with open(proc / "frontal.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "image", "p_frontal"])
        w.writerows(sorted(selected, key=lambda r: int(r[0][3:])))

    all_studies = {s for _, s, _ in rows}
    print(f"Unusable images dropped: {len(unusable)} {sorted(unusable)}")
    print(f"Studies with images: {len(all_studies)}; with a usable image: {len(by_study)}")
    print(f"Frontal selected: {len(selected)}; excluded (best p < {MIN_P}): {len(excluded)}")
    low = sorted((img, pf) for _, img, pf in selected if pf < 0.5)
    print(f"Selected with p_frontal < 0.5 (all shown on sheet): {len(low)}")
    sheet(low, image_dir, proc / "frontal_check_low_p.png")
    random.seed(SEED)
    ex = sorted(random.sample([(img, pf) for _, img, pf in excluded], min(40, len(excluded))))
    sheet(ex, image_dir, proc / "frontal_check_excluded.png")
    with open(proc / "frontal_check_index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sheet", "k", "image", "p_frontal"])
        w.writerows(("low_p", k, img, pf) for k, (img, pf) in enumerate(low))
        w.writerows(("excluded", k, img, pf) for k, (img, pf) in enumerate(ex))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
