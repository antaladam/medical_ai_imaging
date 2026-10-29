"""Label statistics for finding selection, plus a check of CheXbert against IU's manual MeSH codes.

Prints, per CheXpert observation: positives and uncertain counts among the usable (frontal)
studies, and positive rate among the excluded lateral-only studies. Then agreement of CheXbert
positives with the dataset's manual MeSH "major" codes for observations with a clear mapping.

Usage: python -I scripts/data_prep/label_stats.py <reports_dir> <processed_dir>
"""
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# CheXbert observation -> manual MeSH major-code heads that count as "present"
MESH_MAP = {
    "Cardiomegaly": {"Cardiomegaly"},
    "Pleural Effusion": {"Pleural Effusion"},
    "Atelectasis": {"Pulmonary Atelectasis"},
    "Edema": {"Pulmonary Edema"},
    "Pneumothorax": {"Pneumothorax"},
    "Pneumonia": {"Pneumonia"},
    "Consolidation": {"Consolidation"},
    "Fracture": {"Fractures, Bone"},
    "Lung Opacity": {"Opacity", "Airspace Disease", "Infiltrate", "Consolidation", "Pulmonary Edema"},
    "No Finding": {"normal"},
}


def main(report_dir: Path, proc: Path) -> None:
    with open(proc / "chexbert_labels.csv") as fh:
        labels = {r["study"]: r for r in csv.DictReader(fh)}
    with open(proc / "frontal.csv") as fh:
        usable = [r["study"] for r in csv.DictReader(fh)]
    with open(proc / "views.csv") as fh:
        with_images = {r["study"] for r in csv.DictReader(fh)}
    excluded = sorted(with_images - set(usable))
    conds = [c for c in labels[usable[0]] if not c.startswith("raw_") and c not in ("study", "empty_text")]

    print(f"Usable studies: {len(usable)}; excluded lateral-only: {len(excluded)}")
    print(f"{'observation':28s} {'pos':>5s} {'pos%':>6s} {'uncert':>6s} | {'excl pos%':>9s}")
    for c in conds:
        pos = sum(labels[s][c] == "1" for s in usable)
        unc = sum(labels[s][f"raw_{c}"] == "3" for s in usable) if c != "No Finding" else 0
        ex = sum(labels[s][c] == "1" for s in excluded)
        print(f"{c:28s} {pos:5d} {100 * pos / len(usable):5.1f}% {unc:6d} | {100 * ex / len(excluded):8.1f}%")

    mesh = {}
    for p in report_dir.glob("*.xml"):
        root = ET.parse(p).getroot()
        mesh[root.find("uId").get("id")] = {(m.text or "").split("/")[0].strip() for m in root.iter("major")}
    print("\nCheXbert vs manual MeSH codes (usable studies):")
    print(f"{'observation':20s} {'both':>5s} {'cxb only':>8s} {'mesh only':>9s} {'agree%':>7s} {'kappa':>6s}")
    n = len(usable)
    for c, heads in MESH_MAP.items():
        a = [labels[s][c] == "1" for s in usable]
        b = [bool(mesh[s] & heads) for s in usable]
        both = sum(x and y for x, y in zip(a, b))
        a_only = sum(x and not y for x, y in zip(a, b))
        b_only = sum(y and not x for x, y in zip(a, b))
        agree = sum(x == y for x, y in zip(a, b)) / n
        pa, pb = sum(a) / n, sum(b) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
        print(f"{c:20s} {both:5d} {a_only:8d} {b_only:9d} {100 * agree:6.1f}% {kappa:6.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
