"""Ground truth for the six study findings from IU X-Ray's manual MeSH "major" codes.

Decided 2026-10-16 (results.md §2.4b): the manual codes define ground truth; CheXbert is kept
for labelling system outputs, and its agreement with this ground truth measures labeller error.

Mapping (code = "Head/modifier/modifier..."; a study is positive if any code matches):
  Cardiomegaly      Cardiomegaly, or Cardiac Shadow with "enlarged"; "borderline" -> negative
  Pleural Effusion  Pleural Effusion
  Atelectasis       Pulmonary Atelectasis
  Fracture          Fractures, Bone (healed included)
  Lung Lesion       Nodule / Mass / Cavitation located in the lung (or no location given);
                    calcified granulomas excluded
  Lung Opacity      Opacity / Density located in the lung (or no location given),
                    Airspace Disease, Infiltrate, Consolidation
Studies coded "No Indexing" (never coded) are excluded.

Writes data/processed/ground_truth.csv for the usable (frontal) studies, prints counts and
CheXbert-vs-ground-truth agreement.

Usage: python -I scripts/data_prep/mesh_ground_truth.py <reports_dir> <processed_dir>
"""
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FINDINGS = ["Lung Opacity", "Cardiomegaly", "Atelectasis", "Lung Lesion", "Pleural Effusion", "Fracture"]


def lung_or_unlocated(mods):
    return not mods or "lung" in mods


def positives(codes):
    out = set()
    for code in codes:
        head, *mods = [x.strip() for x in code.split("/")]
        if head == "Cardiomegaly" and "borderline" not in mods:
            out.add("Cardiomegaly")
        if head == "Cardiac Shadow" and "enlarged" in mods and "borderline" not in mods:
            out.add("Cardiomegaly")
        if head == "Pleural Effusion":
            out.add("Pleural Effusion")
        if head == "Pulmonary Atelectasis":
            out.add("Atelectasis")
        if head == "Fractures, Bone":
            out.add("Fracture")
        if head in ("Nodule", "Mass", "Cavitation") and lung_or_unlocated(mods):
            out.add("Lung Lesion")
        if head in ("Opacity", "Density") and lung_or_unlocated(mods):
            out.add("Lung Opacity")
        if head in ("Airspace Disease", "Infiltrate", "Consolidation"):
            out.add("Lung Opacity")
    return out


def main(report_dir: Path, proc: Path) -> None:
    with open(proc / "frontal.csv") as fh:
        usable = [r["study"] for r in csv.DictReader(fh)]
    codes = {}
    for p in report_dir.glob("*.xml"):
        root = ET.parse(p).getroot()
        codes[root.find("uId").get("id")] = [(m.text or "") for m in root.iter("major")]

    kept = [s for s in usable if not any(c.split("/")[0].strip() == "No Indexing" for c in codes[s])]
    gt = {s: positives(codes[s]) for s in kept}
    with open(proc / "ground_truth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study"] + FINDINGS)
        w.writerows([s] + [int(f in gt[s]) for f in FINDINGS] for s in kept)

    with open(proc / "chexbert_labels.csv") as fh:
        cxb = {r["study"]: r for r in csv.DictReader(fh)}
    print(f"Usable studies: {len(usable)}; 'No Indexing' excluded: {len(usable) - len(kept)}; "
          f"ground-truth studies: {len(kept)}")
    print(f"Studies with none of the six findings: {sum(not gt[s] for s in kept)}")
    print(f"\n{'finding':18s} {'GT pos':>6s} {'%':>5s} | CheXbert vs GT: {'sens':>5s} {'spec':>5s} {'kappa':>5s}")
    n = len(kept)
    for f in FINDINGS:
        y = [f in gt[s] for s in kept]
        x = [cxb[s][f] == "1" for s in kept]
        tp = sum(a and b for a, b in zip(x, y)); fp = sum(a and not b for a, b in zip(x, y))
        fn = sum(b and not a for a, b in zip(x, y)); tn = n - tp - fp - fn
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
        print(f"{f:18s} {sum(y):6d} {100 * sum(y) / n:4.1f}% | {'':15s} {tp / (tp + fn):5.2f} "
              f"{tn / (tn + fp):5.2f} {(po - pe) / (1 - pe):5.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
