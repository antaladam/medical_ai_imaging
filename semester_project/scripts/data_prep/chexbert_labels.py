"""Label IU X-Ray reports with CheXbert (14 CheXpert observations).

Model definition follows CheXbert (Smit et al. 2020, github.com/stanfordmlgroup/CheXbert) as
packaged in f1chexbert 0.0.2 (J.-B. Delbrouck, MIT licence). Weights: `chexbert.pth` from the
Hugging Face repo StanfordAIMI/RRG_scorers, loaded with weights_only=True and strict=True.

Head outputs (CheXbert classification mode): 0 blank, 1 positive, 2 negative, 3 uncertain;
"No Finding" head: 0 blank, 1 positive. Binary label used in this project (plan §5): positive
only (1 -> 1; blank, negative, uncertain -> 0).

Text labelled: FINDINGS + IMPRESSION, concatenated. With --drop-masks, the de-identification
masks ("XXXX") are removed first.

Usage: python -I scripts/data_prep/chexbert_labels.py <reports_dir> <out_csv> [--drop-masks]
"""
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel, BertTokenizer

CONDITIONS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
              "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture", "Support Devices", "No Finding"]


class BertLabeler(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained("bert-base-uncased"))
        hidden = self.bert.pooler.dense.in_features
        self.dropout = nn.Dropout(0.1)
        self.linear_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(13)] + [nn.Linear(hidden, 2)])

    def forward(self, ids, mask):
        cls = self.dropout(self.bert(ids, attention_mask=mask)[0][:, 0, :])
        return [head(cls) for head in self.linear_heads]


def report_text(root, drop_masks: bool) -> tuple[str, str]:
    sec = {a.get("Label"): (a.text or "").strip() for a in root.iter("AbstractText")}
    text = " ".join(t for t in (sec.get("FINDINGS", ""), sec.get("IMPRESSION", "")) if t)
    if drop_masks:  # remove de-identification masks ("XXXX"), then tidy spaces
        text = re.sub(r"\s+", " ", re.sub(r"\bX{4}\b", " ", text)).strip()
    return root.find("uId").get("id"), text


def load_model(device: str):
    """CheXbert with the checked weights, in eval mode, and its tokenizer (also used by
    label_outputs.py for the systems' free text)."""
    ckpt = hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    model = BertLabeler()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # position_ids buffers differ between transformers versions; anything else is an error
    missing = [k for k in missing if not k.endswith("position_ids")]
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    assert not missing and not unexpected, (missing, unexpected)
    return model.to(device).eval(), BertTokenizer.from_pretrained("bert-base-uncased")


@torch.no_grad()
def main(report_dir: Path, out_csv: Path, drop_masks: bool) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(device)

    reports = [report_text(ET.parse(p).getroot(), drop_masks) for p in report_dir.glob("*.xml")]
    reports.sort(key=lambda r: int(r[0][3:]))
    rows = []
    for i in range(0, len(reports), 32):
        batch = reports[i:i + 32]
        enc = tok([t if t else "" for _, t in batch], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        outs = [o.argmax(1).tolist() for o in model(enc["input_ids"], enc["attention_mask"])]
        for j, (uid, text) in enumerate(batch):
            raw = [outs[c][j] for c in range(14)]
            rows.append([uid, int(not text)] + raw + [int(r == 1) for r in raw])

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "empty_text"] + [f"raw_{c}" for c in CONDITIONS] + CONDITIONS)
        w.writerows(rows)
    print(f"Labelled {len(rows)} reports ({sum(r[1] for r in rows)} with empty findings+impression)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), drop_masks=len(sys.argv) > 3 and sys.argv[3] == "--drop-masks")
