"""Week-1 patient-ID check on the IU X-Ray (Open-i) reports.

Answers: does the release contain any patient identifier, and is there evidence of repeat
patients (prior exams referenced, or image-name parts shared across studies)?

Usage: python -I scripts/data_prep/check_patient_ids.py data/raw/iu_xray/reports/ecgen-radiology
"""
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

PRIOR_WORDS = re.compile(r"\b(prior|previous|unchanged|stable|interval|again|redemonstrat\w*|compared)\b", re.I)


def main(report_dir: Path) -> None:
    files = sorted(report_dir.glob("*.xml"), key=lambda p: int(p.stem))
    tags = Counter()
    id_attrs = Counter()
    comparison = Counter()
    prior_mentions = 0
    images_per_report = Counter()
    im_series_to_reports = defaultdict(set)
    reports_without_images = 0

    for path in files:
        root = ET.parse(path).getroot()
        for el in root.iter():
            tags[el.tag] += 1
            for attr in el.attrib:
                if "id" in attr.lower():
                    id_attrs[f"{el.tag}@{attr}"] += 1

        uid = root.find("uId").get("id")
        sections = {a.get("Label"): (a.text or "").strip() for a in root.iter("AbstractText")}
        comp = sections.get("COMPARISON", "")
        comparison["none/empty" if comp.lower().rstrip(".") in {"", "none"} else "mentions a prior"] += 1
        if PRIOR_WORDS.search(sections.get("FINDINGS", "") + " " + sections.get("IMPRESSION", "")):
            prior_mentions += 1

        images = [img.get("id") for img in root.iter("parentImage")]
        images_per_report[len(images)] += 1
        if not images:
            reports_without_images += 1
        for img in images:
            # e.g. CXR1_1_IM-0001-3001 -> series part "IM-0001"
            m = re.search(r"_(IM-\d+)-", img)
            if m:
                im_series_to_reports[m.group(1)].add(uid)

    print(f"Reports: {len(files)}")
    print("\nAll element tags (count):")
    for tag, n in sorted(tags.items()):
        print(f"  {tag}: {n}")
    print("\nID-like attributes:")
    for k, n in sorted(id_attrs.items()):
        print(f"  {k}: {n}")
    print(f"\nCOMPARISON field: {dict(comparison)}")
    print(f"Findings/impression wording suggesting a prior exam: {prior_mentions}")
    print(f"Images per report: {dict(sorted(images_per_report.items()))}")
    print(f"Reports without images: {reports_without_images}")
    shared = {k: v for k, v in im_series_to_reports.items() if len(v) > 1}
    print(f"IM-series values: {len(im_series_to_reports)}; shared across >1 report: {len(shared)}")
    for k, v in list(shared.items())[:10]:
        print(f"  {k}: {sorted(v)[:6]}{' ...' if len(v) > 6 else ''} ({len(v)} reports)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
