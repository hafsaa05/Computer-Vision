# config.py
# all settings and paths in one place
# CV Assignment 2 - Hafsa Rashid (23K-0064), Syeda Sara Ali (23K-0070)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# paths
DATA_DIR = os.path.join(BASE_DIR, "data")
# our colab run folder (best.pt is not in the repo, copy it here)
RUN_DIR = os.path.join(BASE_DIR, "runs", "resnet50")
MODEL_PATH = os.path.join(RUN_DIR, "best.pt")
THRESHOLDS_JSON = os.path.join(RUN_DIR, "thresholds.json")
VAL_PROBS = os.path.join(RUN_DIR, "val_probs.npy")
VAL_GT_CSV = os.path.join(DATA_DIR, "val_gt.csv")
PRIOR_JSON = os.path.join(DATA_DIR, "attribute_prior.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")

# image size (height x width), pedestrians are tall so 256x128
IMG_H = 256
IMG_W = 128

# imagenet mean/std because backbone is imagenet pretrained
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NUM_ATTRIBUTES = 40

# training settings
# our colab run used --epochs 3 --bs 64 --lr 3e-4 --light_aug (see colab_train.ipynb)
BACKBONE = "resnet50"
EPOCHS = 12
BATCH_SIZE = 48
LR = 1e-4
WEIGHT_DECAY = 0.05
EMA_DECAY = 0.999
DROPOUT = 0.2
LABEL_SMOOTH = 0.02
NUM_WORKERS = 4
SEED = 0
