"""Zero-shot frontal vs. lateral view classification of IU X-Ray images with BiomedCLIP.

Writes data/processed/views.csv (image, study, p_frontal) and a contact sheet of a random sample
for checking by eye.

Usage: python -I scripts/data_prep/classify_views.py <reports_dir> <images_dir> <out_dir>
"""
import csv
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageDraw

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
PROMPTS = {
    "frontal": ["a frontal chest x-ray", "a posteroanterior (PA) chest radiograph",
                "an anteroposterior (AP) chest radiograph"],
    "lateral": ["a lateral chest x-ray", "a lateral view chest radiograph",
                "a side view chest radiograph"],
}
SAMPLE_SIZE = 60
SEED = 0


def png_to_study(report_dir: Path) -> dict[str, str]:
    mapping = {}
    for path in report_dir.glob("*.xml"):
        root = ET.parse(path).getroot()
        uid = root.find("uId").get("id")
        for img in root.iter("parentImage"):
            full = img.get("id")
            mapping[full] = uid
            mapping[re.sub(r"^(CXR\d+)_\d+_", r"\1_", full)] = uid
    return mapping


@torch.no_grad()
def main(report_dir: Path, image_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = open_clip.create_model_from_pretrained(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model = model.to(device).eval()

    text_feats = []
    for view in ("frontal", "lateral"):
        t = model.encode_text(tokenizer(PROMPTS[view]).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
        text_feats.append(t.mean(0))
    text = torch.stack(text_feats)
    text = text / text.norm(dim=-1, keepdim=True)

    study_of = png_to_study(report_dir)
    paths = sorted(image_dir.glob("*.png"))
    rows = []
    for i in range(0, len(paths), 64):
        batch = paths[i:i + 64]
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        probs = (model.logit_scale.exp() * f @ text.T).softmax(-1)[:, 0].tolist()
        rows += [(p.stem, study_of[p.stem], round(pf, 4)) for p, pf in zip(batch, probs)]

    with open(out_dir / "views.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "study", "p_frontal"])
        w.writerows(rows)

    n_frontal = sum(r[2] >= 0.5 for r in rows)
    print(f"Images: {len(rows)}; frontal: {n_frontal}; lateral: {len(rows) - n_frontal}")
    uncertain = sum(0.2 < r[2] < 0.8 for r in rows)
    print(f"Uncertain (0.2 < p_frontal < 0.8): {uncertain}")

    by_study = {}
    for img, study, pf in rows:
        by_study.setdefault(study, []).append(pf)
    counts = {}
    for pfs in by_study.values():
        key = (sum(p >= 0.5 for p in pfs), len(pfs))
        counts[key] = counts.get(key, 0) + 1
    print("Studies by (frontal images, total images):", dict(sorted(counts.items())))

    random.seed(SEED)
    sample = random.sample(rows, SAMPLE_SIZE)
    thumb, cols = 160, 10
    sheet = Image.new("RGB", (cols * thumb, (SAMPLE_SIZE // cols) * (thumb + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (img, _, pf) in enumerate(sample):
        im = Image.open(image_dir / f"{img}.png").convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = (k % cols) * thumb, (k // cols) * (thumb + 20)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + thumb + 2), f"{k}: {'F' if pf >= 0.5 else 'L'} {pf:.2f}", fill="black")
    sheet.save(out_dir / "views_sample_sheet.png")
    with open(out_dir / "views_sample.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "image", "p_frontal"])
        w.writerows((k, img, pf) for k, (img, _, pf) in enumerate(sample))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
