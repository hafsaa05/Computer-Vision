# make_submission.py
# packs our model + run.py into the zip codabench wants (files at the root of the zip)
#   python make_submission.py               -> runs/resnet50/resnet50_tuned.zip
#   python make_submission.py --no_tuning   -> all thresholds 0.5

import argparse
import json
import os
import zipfile

import config

parser = argparse.ArgumentParser()
parser.add_argument("--no_tuning", action="store_true")
parser.add_argument("--out", default=None)
parser.add_argument("--model", default=config.MODEL_PATH)
parser.add_argument("--backbone", default=config.BACKBONE)
parser.add_argument("--thr_json", default=config.THRESHOLDS_JSON)
args = parser.parse_args()

attrs = json.load(open(config.PRIOR_JSON))["attribute_names"]
if args.no_tuning:
    thr = [0.5] * len(attrs)
    name = "resnet50_thr05.zip"
else:
    thr = json.load(open(args.thr_json))["thr"]
    name = "resnet50_tuned.zip"
assert len(thr) == len(attrs) == config.NUM_ATTRIBUTES

out_path = args.out or os.path.join(config.RUN_DIR, name)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

meta = {"attributes": attrs, "thr": [float(t) for t in thr], "h": config.IMG_H, "w": config.IMG_W, "backbone": args.backbone}
code_dir = os.path.join(config.BASE_DIR, "submission_code")

with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(code_dir, "run.py"), "run.py")
    z.write(os.path.join(code_dir, "metadata.yaml"), "metadata.yaml")  # official file, dont change
    z.write(args.model, "assets/model.pt")  # already fp16
    z.writestr("assets/meta.json", json.dumps(meta, indent=1))

print("saved", out_path, f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
