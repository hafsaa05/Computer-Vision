# check_submission.py
# tests our zip locally before uploading, like codabench would
# (unzip in temp folder, block internet, import run.py, check outputs)
#   python check_submission.py runs/resnet50/resnet50_tuned.zip

import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
import zipfile

import numpy as np
import torch
from PIL import Image

import config
from dataset import get_transforms
from model import load_trained_model

zip_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.RUN_DIR, "resnet50_tuned.zip")
images = [os.path.join(config.SAMPLE_DIR, f) for f in ["ped1.png", "ped2.png", "ped3.png"]]
failed = []


def check(name, ok, extra=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({extra})" if extra else ""))
    if not ok:
        failed.append(name)


# 1. zip structure
with zipfile.ZipFile(zip_path) as z:
    files = z.namelist()
    tmp_dir = tempfile.mkdtemp()
    z.extractall(tmp_dir)
needed = {"run.py", "metadata.yaml", "assets/model.pt", "assets/meta.json"}
check("zip has run.py, metadata.yaml, assets/ at root", needed <= set(files), str(files))
check("zip size", True, f"{os.path.getsize(zip_path) / 1e6:.1f} MB")


# 2. block internet, run.py must work offline
def no_internet(*a, **k):
    raise RuntimeError("tried to use internet!")


socket.socket.connect = no_internet
socket.create_connection = no_internet

spec = importlib.util.spec_from_file_location("run", os.path.join(tmp_dir, "run.py"))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
check("run.py has load_model / predict_image / predict_batch",
      all(hasattr(run, f) for f in ["load_model", "predict_image", "predict_batch"]))
run.load_model()
check("load_model() works offline", True)

meta = json.load(open(os.path.join(tmp_dir, "assets", "meta.json")))
attrs = meta["attributes"]
imgs = [Image.open(p).convert("RGB") for p in images]

# 3. output format
out = run.predict_image({"image": imgs[0], "attribute_names": attrs})
check("predict_image gives 40 floats", isinstance(out, list) and len(out) == 40 and all(isinstance(v, float) for v in out))
check("all values between 0 and 1", 0 <= min(out) and max(out) <= 1, f"min {min(out):.3f}, max {max(out):.3f}")
batch_out = run.predict_batch([{"image": im, "attribute_names": attrs} for im in imgs])
check("predict_batch gives one row per image", len(batch_out) == len(imgs))
check("predict_batch same as predict_image", np.allclose(batch_out[0], out, atol=1e-5))

# 4. attribute order
order = list(np.random.RandomState(0).permutation(40))
shuffled = run.predict_image({"image": imgs[0], "attribute_names": [attrs[i] for i in order]})
check("shuffled attribute order is followed", np.allclose(np.array(out)[order], shuffled, atol=1e-5))

# 5. different input types should give the same answer
buf = io.BytesIO()
imgs[0].save(buf, "PNG")
inputs = {
    "PIL image": {"image": imgs[0]},
    "numpy array": {"image": np.array(imgs[0])},
    "tensor CHW": {"image": torch.from_numpy(np.array(imgs[0])).permute(2, 0, 1).float() / 255},
    "bytes": {"image": buf.getvalue()},
    "file path": {"image_path": images[0]},
}
for name, s in inputs.items():
    s["attribute_names"] = attrs
    check(f"input type: {name}", np.allclose(run.predict_image(s), out, atol=2e-3))

# 6. compare with our own model to make sure packing didnt break anything
tmp_model = os.path.join(tmp_dir, "assets", "model.pt")
model = load_trained_model(tmp_model, backbone=meta.get("backbone", config.BACKBONE))
tf = get_transforms(train=False)
x = torch.stack([tf(im) for im in imgs])
with torch.no_grad():
    ref = ((torch.sigmoid(model(x)) + torch.sigmoid(model(torch.flip(x, dims=[3])))) / 2).numpy()
thr = np.array(meta["thr"])
ref_decision = ref >= thr[None]
sub_decision = np.array(batch_out) > 0.5
flips = int((ref_decision != sub_decision).sum())
check("same yes/no decisions as our model", flips == 0, f"{flips} different")

print()
print("ALL TESTS PASSED" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
