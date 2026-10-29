"""Check that every extracted PNG maps to exactly one report image ID, and the reverse.

Report XML image IDs look like CXR1_1_IM-0001-3001; PNG names look like CXR1_IM-0001-3001.png.
The middle "_<n>_" part is dropped in the PNG names.

Usage: python -I scripts/data_prep/check_image_mapping.py <reports_dir> <images_dir>
"""
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


def png_key(xml_id: str) -> str:
    return re.sub(r"^(CXR\d+)_\d+_", r"\1_", xml_id)


def main(report_dir: Path, image_dir: Path) -> None:
    xml_ids = {}
    for path in report_dir.glob("*.xml"):
        root = ET.parse(path).getroot()
        uid = root.find("uId").get("id")
        for img in root.iter("parentImage"):
            xml_ids[png_key(img.get("id"))] = uid

    pngs = {p.stem: p for p in image_dir.glob("*.png")}
    others = [p.name for p in image_dir.iterdir() if p.suffix != ".png"]
    print(f"Report image IDs: {len(xml_ids)}; PNG files: {len(pngs)}; other files: {others}")
    print(f"PNGs without a report: {len(set(pngs) - set(xml_ids))}")
    print(f"Report images without a PNG: {len(set(xml_ids) - set(pngs))}")
    bad_prefix = [k for k, uid in xml_ids.items() if k in pngs and not k.startswith(uid + "_")]
    print(f"PNG name prefix disagrees with its report ID: {len(bad_prefix)}")

    modes, shapes = Counter(), Counter()
    unreadable = 0
    for p in pngs.values():
        try:
            with Image.open(p) as im:
                modes[im.mode] += 1
                w, h = im.size
                shapes["landscape" if w > h else "portrait" if h > w else "square"] += 1
        except Exception:
            unreadable += 1
    print(f"Unreadable PNGs: {unreadable}; modes: {dict(modes)}; orientation: {dict(shapes)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
