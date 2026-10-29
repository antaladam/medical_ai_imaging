"""Exact and near-exact pixel duplicates among the selected frontal images.

Checks (a) identical file bytes (MD5) and (b) identical 16x16 greyscale thumbnails after
quantising to 32 grey levels (catches resized/re-encoded copies of the same image).

Usage: python -I scripts/data_prep/pixel_duplicates.py <processed_dir> <images_dir>
"""
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def main(proc: Path, image_dir: Path) -> None:
    with open(proc / "frontal.csv") as fh:
        rows = [(r["study"], r["image"]) for r in csv.DictReader(fh)]
    md5, thumb = defaultdict(list), defaultdict(list)
    for study, img in rows:
        p = image_dir / f"{img}.png"
        md5[hashlib.md5(p.read_bytes()).hexdigest()].append(study)
        a = np.asarray(Image.open(p).convert("L").resize((16, 16), Image.BILINEAR)) // 8
        thumb[a.tobytes()].append(study)
    print("Identical files:", [v for v in md5.values() if len(v) > 1])
    print("Identical 16x16 quantised thumbnails:", [v for v in thumb.values() if len(v) > 1])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
