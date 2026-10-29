"""Is the IM-#### part of IU X-Ray image names a patient link, or a sequential counter?

Test: sort report IDs as strings ("CXR1" < "CXR10" < "CXR100" ...). If IM numbers never decrease
in that order and shared values only occur between neighbours, IM-#### is a counter assigned
over string-sorted IDs, not a patient identifier. Also prints the reports of a few shared groups
so they can be compared by eye.

Usage: python -I scripts/data_prep/check_im_series.py data/raw/iu_xray/reports/ecgen-radiology
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main(report_dir: Path) -> None:
    rows = {}
    for path in report_dir.glob("*.xml"):
        root = ET.parse(path).getroot()
        uid = root.find("uId").get("id")
        ims = {int(m.group(1)) for img in root.iter("parentImage")
               if (m := re.search(r"_IM-(\d+)-", img.get("id")))}
        sec = {a.get("Label"): (a.text or "").strip() for a in root.iter("AbstractText")}
        rows[uid] = (ims, sec)

    with_images = [u for u in sorted(rows) if rows[u][0]]  # string sort
    multi = [u for u in with_images if len(rows[u][0]) > 1]
    print(f"Reports with images: {len(with_images)}; with >1 distinct IM value: {len(multi)}")

    seq = [(u, min(rows[u][0])) for u in with_images]
    decreases = [(a, b) for a, b in zip(seq, seq[1:]) if b[1] < a[1]]
    print(f"Decreases of IM number along string-sorted IDs: {len(decreases)}")
    for a, b in decreases[:10]:
        print(f"  {a} -> {b}")

    by_im = {}
    for u, im in seq:
        by_im.setdefault(im, []).append(u)
    pos = {u: i for i, (u, _) in enumerate(seq)}
    non_contiguous = [k for k, v in by_im.items()
                      if len(v) > 1 and max(pos[u] for u in v) - min(pos[u] for u in v) != len(v) - 1]
    print(f"Shared IM values whose reports are NOT neighbours in string order: {len(non_contiguous)}")

    print("\nSample shared groups (indication | impression):")
    for k in [k for k, v in by_im.items() if len(v) > 2][:3]:
        print(f" IM-{k:04d}")
        for u in by_im[k]:
            sec = rows[u][1]
            print(f"   {u}: {sec.get('INDICATION', '')[:50]!r} | {sec.get('IMPRESSION', '')[:60]!r}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
