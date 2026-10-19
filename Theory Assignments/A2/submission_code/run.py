# run.py - our code submission for UPAR 2027 Track 1
# codabench imports this file and calls load_model(), predict_image(sample), predict_batch(samples)
# sample["attribute_names"] gives the attribute order, present means output > 0.5
# we shift the logits so prob >= tuned thr gives output > 0.5
# only torch, torchvision, numpy, PIL (no internet)

import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BATCH_SIZE = 128

model = None
device = "cpu"
info = {}
logit_shift = None
pool = ThreadPoolExecutor(8)
printed_keys = False
image_root = None  # found automatically

# we dont know the exact key name for the image, so we try a few
IMAGE_KEYS = ["image", "img", "pil_image", "image_array", "array", "pixels", "data",
              "image_path", "path", "file", "filename", "image_name", "name"]
ROOT_KEYS = ["root", "data_root", "image_root", "img_root", "data_dir", "image_dir", "images_dir", "dataset_dir", "base_dir"]


def load_model():
    global model, device, info, logit_shift
    info = json.load(open(os.path.join(HERE, "assets", "meta.json")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_attr = len(info["attributes"])
    backbone = info.get("backbone", "convnext_small")
    net = getattr(torchvision.models, backbone)(weights=None)  # weights=None so no internet needed
    if backbone.startswith("resnet"):
        net.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(net.fc.in_features, n_attr))
    else:
        net.classifier[-1] = nn.Sequential(nn.Dropout(0.0), nn.Linear(net.classifier[-1].in_features, n_attr))
    state = torch.load(os.path.join(HERE, "assets", "model.pt"), map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() for k, v in state.items()})
    model = net.to(device).eval()

    thr = np.clip(np.array(info["thr"], dtype=np.float32), 1e-3, 1 - 1e-3)
    logit_shift = torch.tensor(np.log(thr / (1 - thr)), device=device)


def find_file(rel_path, sample):
    # paths like "MEVID/images/0.jpg" are relative to some data folder
    global image_root
    rel_path = str(rel_path)
    if os.path.isfile(rel_path):
        return rel_path
    if image_root is not None and os.path.isfile(os.path.join(image_root, rel_path)):
        return os.path.join(image_root, rel_path)

    # 1) folders given inside the sample itself
    roots = []
    if isinstance(sample, dict):
        for k in ROOT_KEYS:
            if isinstance(sample.get(k), str):
                roots.append(sample[k])
        for k, v in sample.items():
            if isinstance(v, str) and os.path.isdir(v):
                roots.append(v)
    # 2) environment variables and common places
    for k, v in os.environ.items():
        if isinstance(v, str) and ("DIR" in k or "DATA" in k or "INPUT" in k or "ROOT" in k) and os.path.isdir(v):
            roots.append(v)
    roots += [os.getcwd(), "/app/input_data", "/app/input", "/app/data", "/app", os.path.dirname(HERE), "/data", "/input", "/"]
    for r in roots:
        if os.path.isfile(os.path.join(r, rel_path)):
            image_root = r
            print(f"[run.py] image root = {r}", file=sys.stderr)
            return os.path.join(r, rel_path)

    # 3) last try: search for it once and remember the folder
    first_part = rel_path.replace("\\", "/").split("/")[0]
    for start in ["/app", os.getcwd(), "/data", "/input", "/mnt", "/tmp", "/home"]:
        if not os.path.isdir(start):
            continue
        for dirpath, dirnames, _ in os.walk(start):
            if first_part in dirnames and os.path.isfile(os.path.join(dirpath, rel_path)):
                image_root = dirpath
                print(f"[run.py] found image root by search = {dirpath}", file=sys.stderr)
                return os.path.join(dirpath, rel_path)
            if dirpath.count(os.sep) - start.count(os.sep) >= 6:
                dirnames[:] = []  # dont go too deep
    raise FileNotFoundError(f"could not find {rel_path}, sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")


def to_pil(x, sample=None):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, (str, os.PathLike)):
        return Image.open(find_file(x, sample)).convert("RGB")
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = x.transpose(1, 2, 0)  # CHW -> HWC
    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = x * 255
        x = x.clip(0, 255).astype(np.uint8)
    return Image.fromarray(x).convert("RGB")


def is_pixels(v):
    return isinstance(v, (Image.Image, bytes, bytearray, torch.Tensor, np.ndarray))


def get_image(sample):
    if not isinstance(sample, dict):
        return to_pil(sample, sample)
    # first try real image data under any key
    for k in IMAGE_KEYS:
        if is_pixels(sample.get(k)):
            return to_pil(sample[k], sample)
    for k, v in sample.items():
        if k != "attribute_names" and is_pixels(v):
            return to_pil(v, sample)
    # then a file path
    for k in IMAGE_KEYS:
        if isinstance(sample.get(k), (str, os.PathLike)):
            return to_pil(sample[k], sample)
    raise KeyError(f"could not find image in sample, keys: {list(sample.keys())}")


def prepare(sample):
    h, w = info["h"], info["w"]
    return np.asarray(get_image(sample).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def predict_batch(samples):
    global printed_keys
    if model is None:
        load_model()
    if not printed_keys and samples:
        s = samples[0]
        if isinstance(s, dict):
            print("[run.py] sample keys: " + str({k: type(v).__name__ for k, v in s.items() if k != "attribute_names"}), file=sys.stderr)
        printed_keys = True

    imgs = list(pool.map(prepare, samples))  # load images in parallel
    outputs = []
    with torch.no_grad():
        for i in range(0, len(imgs), BATCH_SIZE):
            x = torch.from_numpy(np.stack(imgs[i:i + BATCH_SIZE])).to(device).permute(0, 3, 1, 2).float() / 255
            x = (x - MEAN.to(device)) / STD.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                # flip tta
                p = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, dims=[3])).float())) / 2
            p = p.float().clamp(1e-6, 1 - 1e-6)
            p = torch.sigmoid(torch.log(p / (1 - p)) - logit_shift)  # apply tuned thresholds
            outputs.append(p.cpu())
    probs = torch.cat(outputs).numpy()

    # return attributes in the order the ingestion program asks for
    our_names = info["attributes"]
    if samples and isinstance(samples[0], dict) and samples[0].get("attribute_names"):
        cols = []
        for name in samples[0]["attribute_names"]:
            if name in our_names:
                cols.append(probs[:, our_names.index(name)])
            else:
                cols.append(np.zeros(len(probs), dtype=np.float32))
        probs = np.stack(cols, axis=1)

    probs = np.nan_to_num(probs, nan=0.0).clip(0.0, 1.0)
    return [[float(v) for v in row] for row in probs]


def predict_image(sample):
    return predict_batch([sample])[0]
