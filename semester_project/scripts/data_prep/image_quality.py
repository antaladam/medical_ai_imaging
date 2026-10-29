"""Image-quality measures used by select_frontal.py to drop unusable images.

Reproduces the two files that were first computed ad hoc on 2026-10-16 (results.md §2.4):
  image_quality.json   image -> [grey std, mean |horizontal neighbour difference|], both on a
                       128x128 greyscale copy (near-blank images: std < 15)
  image_hf_ratio.json  image -> mean |horizontal neighbour difference| / grey std at full
                       resolution (pure-noise image: ratio > 0.2)

Usage: python -I scripts/data_prep/image_quality.py <images_dir> <processed_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main(image_dir: Path, proc: Path) -> None:
    quality, hf_ratio = {}, {}
    paths = sorted(image_dir.glob("*.png"))
    for p in paths:
        grey = Image.open(p).convert("L")
        small = np.asarray(grey.resize((128, 128)), dtype=np.float32)
        quality[p.stem] = [float(small.std()), float(np.abs(np.diff(small, axis=1)).mean())]
        full = np.asarray(grey, dtype=np.float32)
        hf_ratio[p.stem] = float(np.abs(np.diff(full, axis=1)).mean() / (full.std() + 1e-6))
    json.dump(quality, open(proc / "image_quality.json", "w"))
    json.dump(hf_ratio, open(proc / "image_hf_ratio.json", "w"))
    print(f"Images: {len(paths)}; near-blank (std < 15): {sum(v[0] < 15 for v in quality.values())}; "
          f"noise (ratio > 0.2): {sum(v > 0.2 for v in hf_ratio.values())}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
