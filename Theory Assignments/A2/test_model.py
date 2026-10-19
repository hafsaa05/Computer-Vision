# test_model.py
# quick test of our model on a few pedestrian images
#   python test_model.py                  (uses sample_images/)
#   python test_model.py img1.jpg img2.jpg

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

import config
from dataset import get_transforms
from model import get_head, load_trained_model

MODEL = config.MODEL_PATH
BACKBONE = config.BACKBONE
THR = config.THRESHOLDS_JSON

images = sys.argv[1:]
if not images:
    images = [os.path.join(config.SAMPLE_DIR, f) for f in ["ped1.png", "ped2.png", "ped3.png"]]

attrs = json.load(open(config.PRIOR_JSON))["attribute_names"]
thr = np.array(json.load(open(THR))["thr"])

model = load_trained_model(MODEL, BACKBONE)
n_params = sum(p.numel() for p in model.parameters())
print(f"model loaded, params: {n_params / 1e6:.1f}M")

# check the head is trained, random init weights stay inside +-1/sqrt(in_features)
head = get_head(model)
bound = 1 / head.in_features ** 0.5
print(f"head max |w| = {head.weight.abs().max().item():.3f}  (random init would be <= {bound:.3f})")

tf = get_transforms(train=False)
batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in images])
with torch.no_grad():
    probs = torch.sigmoid(model(batch)).numpy()

confident = ((probs < 0.05) | (probs > 0.95)).mean() * 100
print(f"confident outputs: {confident:.0f}%\n")

for path, p in zip(images, probs):
    present = [f"{a} ({p[i]:.2f})" for i, a in enumerate(attrs) if p[i] >= thr[i]]
    print(os.path.basename(path))
    for item in present:
        print("   ", item)
    print()
