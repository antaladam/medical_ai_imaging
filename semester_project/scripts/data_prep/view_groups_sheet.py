"""Contact sheets of the view-classification groups that need checking by eye.

Groups: studies with no image classified frontal; studies with 2+ frontal images; images with
uncertain p_frontal (0.2-0.8). Up to 40 images per sheet, chosen at random with a fixed seed.

Usage: python -I scripts/data_prep/view_groups_sheet.py <views.csv> <images_dir> <out_dir>
"""
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

THUMB, COLS, MAX_N, SEED = 160, 10, 40, 0


def sheet(items, image_dir: Path, out: Path) -> None:
    rows = (len(items) + COLS - 1) // COLS
    canvas = Image.new("RGB", (COLS * THUMB, rows * (THUMB + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (img, pf) in enumerate(items):
        im = Image.open(image_dir / f"{img}.png").convert("RGB")
        im.thumbnail((THUMB, THUMB))
        x, y = (k % COLS) * THUMB, (k // COLS) * (THUMB + 20)
        canvas.paste(im, (x, y))
        draw.text((x + 2, y + THUMB + 2), f"{k}: {'F' if pf >= 0.5 else 'L'} {pf:.2f}", fill="black")
    canvas.save(out)


def main(views_csv: Path, image_dir: Path, out_dir: Path) -> None:
    with open(views_csv) as fh:
        rows = [(r["image"], r["study"], float(r["p_frontal"])) for r in csv.DictReader(fh)]
    by_study = {}
    for img, study, pf in rows:
        by_study.setdefault(study, []).append((img, pf))

    groups = {
        "no_frontal": [x for imgs in by_study.values() if not any(p >= 0.5 for _, p in imgs) for x in imgs],
        "multi_frontal": [x for imgs in by_study.values() if sum(p >= 0.5 for _, p in imgs) >= 2
                          for x in imgs if x[1] >= 0.5],
        "uncertain": [(img, pf) for img, _, pf in rows if 0.2 < pf < 0.8],
    }
    random.seed(SEED)
    for name, items in groups.items():
        chosen = sorted(random.sample(items, min(MAX_N, len(items))))
        sheet(chosen, image_dir, out_dir / f"views_check_{name}.png")
        with open(out_dir / f"views_check_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["k", "image", "p_frontal"])
            w.writerows((k, img, pf) for k, (img, pf) in enumerate(chosen))
        print(f"{name}: {len(items)} images, sheet shows {len(chosen)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
