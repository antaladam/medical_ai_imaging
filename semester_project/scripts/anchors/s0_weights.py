"""Download and safely load TorchXRayVision weights for S0.

Download: the library's own downloader writes straight to the final file name and crashes on a
Windows cp1252 console (its progress bar prints U+2588), leaving a truncated file that the
library then treats as already downloaded. Here the file goes to `<name>.part`, its size is
checked against Content-Length, and only then is it renamed. SHA-256 is printed for the record.

Loading: the library loads every weights file with torch.load(weights_only=False), which can
run code stored in the file. Here: the ResNet file is a plain state dict, loaded with
weights_only=True. The DenseNet files are legacy (pre-zip) whole-module pickles, which
weights_only=True cannot read at all, so they go through torch's legacy loader with a
restricted unpickler that allows only the 14 globals the files were found to use. The weights
are then copied into a freshly built model with a strict key check.

Usage (download only): python -I scripts/anchors/s0_weights.py <weights name> [...]
"""
import hashlib
import pickle
import sys
import types
import urllib.request
from pathlib import Path

import torch
import torchvision
import torchxrayvision as xrv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py
from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
CACHE = PROJECT / ".cache" / "torchxrayvision"


def weights_path(name: str) -> Path:
    return CACHE / Path(xrv.models.model_urls[name]["weights_url"]).name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str) -> Path:
    path = weights_path(name)
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with urllib.request.urlopen(xrv.models.model_urls[name]["weights_url"], timeout=60) as r:
        expected = int(r.headers["Content-Length"])
        with open(part, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    got = part.stat().st_size
    if got != expected:
        part.unlink()
        raise RuntimeError(f"{name}: got {got} bytes, expected {expected}")
    part.rename(path)
    return path


# Exactly the globals the three DenseNet pickles reference (all GLOBAL opcodes scanned with
# pickletools; no STACK_GLOBAL / INST / OBJ present; diary 2026-10-17). torch resolves the
# *Storage names itself before asking the unpickler.
ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_tensor_v2"),
    ("torch.nn.modules.activation", "ReLU"), ("torch.nn.modules.batchnorm", "BatchNorm2d"),
    ("torch.nn.modules.container", "Sequential"), ("torch.nn.modules.conv", "Conv2d"),
    ("torch.nn.modules.linear", "Linear"), ("torch.nn.modules.pooling", "AvgPool2d"),
    ("torch.nn.modules.pooling", "MaxPool2d"),
    ("torchxrayvision.models", "DenseNet"), ("torchxrayvision.models", "_DenseBlock"),
    ("torchxrayvision.models", "_DenseLayer"), ("torchxrayvision.models", "_Transition"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global not allowed: {module}.{name}")
        return super().find_class(module, name)


# A stand-in pickle module for torch's legacy loader (which these pre-zip files need, and which
# weights_only=True refuses: they store ('module', class, source) records).
_restricted_pickle = types.ModuleType("restricted_pickle")
_restricted_pickle.Unpickler = _RestrictedUnpickler
_restricted_pickle.load = lambda f, **kw: _RestrictedUnpickler(f, **kw).load()


def load(name: str) -> torch.nn.Module:
    """A library model with the named weights, in eval mode, outputs = raw logits."""
    path = download(name)
    if name.startswith("resnet50"):
        # The library's ResNet without op_threshs is exactly this torchvision model
        # (1-channel stem), and its forward() is self.model(x).
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = torchvision.models.resnet50(num_classes=len(xrv.models.model_urls[name]["labels"]),
                                            weights=None)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.load_state_dict(state, strict=True)
        model.targets = xrv.models.model_urls[name]["labels"]
    else:
        saved = torch.load(path, map_location="cpu", pickle_module=_restricted_pickle,
                           weights_only=False)  # restricted by ALLOWED_GLOBALS instead
        for mod in saved.modules():  # same patch as the library: pickles predate this attribute
            if not hasattr(mod, "_non_persistent_buffers_set"):
                mod._non_persistent_buffers_set = set()
        model = xrv.models.DenseNet(weights=None)  # op_threshs=None, apply_sigmoid=False -> logits
        model.load_state_dict(saved.state_dict(), strict=True)
        model.targets = model.pathologies = xrv.models.model_urls[name]["labels"]
    return model.eval()


if __name__ == "__main__":
    for n in sys.argv[1:]:
        p = download(n)
        print(f"{n}: {p.name} {p.stat().st_size} bytes sha256 {sha256(p)}")
