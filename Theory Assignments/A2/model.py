# model.py
# resnet50 (or convnext) with a new 40 output head + weighted bce loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import config


def build_model(backbone=config.BACKBONE, num_attrs=config.NUM_ATTRIBUTES, pretrained=True, dropout=config.DROPOUT):
    weights = "DEFAULT" if pretrained else None

    if backbone == "resnet50":
        model = torchvision.models.resnet50(weights=weights)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_attrs))

    elif backbone in ["convnext_tiny", "convnext_small"]:
        model = getattr(torchvision.models, backbone)(weights=weights, stochastic_depth_prob=0.1)
        # new head with 40 outputs
        in_features = model.classifier[-1].in_features  # 768
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_attrs))

    else:
        raise ValueError("backbone not supported: " + backbone)
    return model


def get_head(model):
    # last linear layer, used in test_model.py
    if hasattr(model, "fc"):
        return model.fc[1]
    return model.classifier[-1][1]


def load_trained_model(path=config.MODEL_PATH, backbone=config.BACKBONE, device="cpu"):
    # best.pt is a fp16 state_dict to keep it small
    model = build_model(backbone, pretrained=False, dropout=0.0)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v.float() for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_pos_weights(y, mask):
    # rare attributes (like sunglasses ~2%) get higher weight so the model doesnt ignore them
    # sqrt to make it less extreme, and clip between 1 and 8
    pos = (y * mask).sum(0)
    total = mask.sum(0).clip(min=1)
    freq = (pos / total).clip(1e-3, 1 - 1e-3)
    w = ((1 - freq) / freq) ** 0.5
    return w.clip(1.0, 8.0)


def weighted_bce_loss(logits, y, mask, pos_weight, smooth=config.LABEL_SMOOTH):
    y = y * (1 - smooth) + 0.5 * smooth  # small label smoothing
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = torch.where(y > 0.5, pos_weight[None, :], torch.ones_like(pos_weight)[None, :])
    loss = loss * w * mask  # masked labels dont count
    return loss.sum() / mask.sum().clamp(min=1)
