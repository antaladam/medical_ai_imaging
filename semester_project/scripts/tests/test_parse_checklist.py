"""Offline test of scripts/evaluation/parse_checklist.py.

Usage: python -I scripts/tests/test_parse_checklist.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from parse_checklist import parse  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


BLOCK = """Lung Opacity: absent
Cardiomegaly: present
Atelectasis: uncertain
Lung Lesion: absent
Pleural Effusion: absent
Fracture: absent

### 1. Image Type
Cardiomegaly is visible."""
v, start = parse(BLOCK)
check("plain block parsed, at start", start and v == {"Lung Opacity": "absent", "Cardiomegaly": "present",
      "Atelectasis": "uncertain", "Lung Lesion": "absent", "Pleural Effusion": "absent", "Fracture": "absent"})

v, start = parse(BLOCK.replace("Cardiomegaly: present", "*   **Cardiomegaly:** Present (mild)")
                 .replace("Fracture: absent", "- fracture – ABSENT."))
check("markdown, bullets, case, dash and trailing text", start and v["Cardiomegaly"] == "present" and v["Fracture"] == "absent")

v, start = parse("Intro sentence first.\n" + BLOCK)
check("block not at start is still parsed but flagged", not start and v["Cardiomegaly"] == "present")

v, _ = parse(BLOCK + "\nCardiomegaly: absent")
check("disagreeing duplicate -> conflict", v["Cardiomegaly"] == "conflict")
v, _ = parse(BLOCK + "\nCardiomegaly: present")
check("agreeing duplicate -> that answer", v["Cardiomegaly"] == "present")

v, start = parse(BLOCK.replace("Atelectasis: uncertain\n", ""))
check("missing finding -> missing, block incomplete", v["Atelectasis"] == "missing" and not start)

v, _ = parse("Cardiomegaly: Not present.\nThe lungs show no Lung Opacity: present elsewhere")
check("other wording or mid-sentence names are not checklist lines",
      v["Cardiomegaly"] == "missing" and v["Lung Opacity"] == "missing")

v, _ = parse("Cardiomegaly: presently unclear")
check("answer word must be a whole word", v["Cardiomegaly"] == "missing")

v, start = parse(None)
check("empty answer -> all missing", set(v.values()) == {"missing"} and not start)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
