# dataset.py
# UPAR csv loading + dataset class + augmentations

import os
import random

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

import config


def read_annotations(csv_path):
    # first column is the image path, rest are the 40 attributes
    df = pd.read_csv(csv_path)
    img_col = df.columns[0]
    attr_names = list(df.columns[1:])
    return df, img_col, attr_names


def clean_labels(labels):
    # missing labels (-1 or 2) are masked so they dont count
    labels = np.asarray(labels, dtype=np.float32)
    mask = ((labels == 0) | (labels == 1)).astype(np.float32)
    y = np.where(mask > 0, labels, 0).astype(np.float32)
    return y, mask


def get_dataset_name(img_path):
    # "Market1501/query/xxx.jpg" -> "Market1501"
    return str(img_path).replace("\\", "/").split("/")[0]


class RandomLowRes:
    # cctv images are often low res, so randomly downscale some
    def __call__(self, img):
        if random.random() < 0.3:
            s = random.uniform(0.3, 0.7)
            new_w = max(8, int(img.width * s))
            new_h = max(16, int(img.height * s))
            img = img.resize((new_w, new_h))
        return img


def get_transforms(train=True, light=False):
    normalize = T.Normalize(config.MEAN, config.STD)
    if not train:
        return T.Compose([
            T.Resize((config.IMG_H, config.IMG_W)),
            T.ToTensor(),
            normalize,
        ])

    if light:
        # light aug for colab, only 2 cpu cores so heavy aug slows it down
        return T.Compose([
            T.Resize((config.IMG_H, config.IMG_W)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.2),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])

    return T.Compose([
        RandomLowRes(),
        T.Resize((config.IMG_H, config.IMG_W)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.8),
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.4, scale=(0.02, 0.2)),  # helps with occlusion
    ])


class PedestrianDataset(Dataset):
    def __init__(self, img_paths, labels, masks, img_root, transform):
        self.img_paths = list(img_paths)
        self.labels = labels
        self.masks = masks
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.img_root, str(self.img_paths[idx]))
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx], self.masks[idx]
