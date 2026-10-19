# train.py
# fine-tune resnet50 on the UPAR train set (track 1), we ran it on colab T4
#   python train.py --train_csv <train gt.csv> --val_csv <val gt.csv> --img_root <data folder> --backbone resnet50 --epochs 3 --bs 64 --lr 3e-4 --light_aug --out runs/resnet50
# if colab disconnects run the same command with --resume

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import PedestrianDataset, clean_labels, get_dataset_name, get_transforms, read_annotations
from metrics import dataset_wise_metrics, find_best_thresholds
from model import build_model, get_pos_weights, weighted_bce_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EMA:
    # ema of weights, usually a bit better on val
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.model.state_dict()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                ema_state[k].copy_(v)


@torch.no_grad()
def predict(model, loader, device, flip_tta=False):
    model.eval()
    all_probs = []
    for imgs, _, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(imgs).float())
            if flip_tta:
                p_flip = torch.sigmoid(model(torch.flip(imgs, dims=[3])).float())
                p = (p + p_flip) / 2
        all_probs.append(p.cpu())
    return torch.cat(all_probs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--bs", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--out", default="runs/exp1")
    parser.add_argument("--backbone", default="resnet50", help="resnet50 / convnext_tiny / convnext_small")
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--light_aug", action="store_true", help="lighter augmentation (faster on colab)")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in --out")
    parser.add_argument("--no_pretrained", action="store_true", help="only for testing without internet")
    parser.add_argument("--limit", type=int, default=0, help="use only N images (for quick testing)")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    train_df, img_col, attrs = read_annotations(args.train_csv)
    val_df, _, _ = read_annotations(args.val_csv)
    if args.limit:
        train_df = train_df.iloc[:args.limit]
        val_df = val_df.iloc[:args.limit]

    y_train, m_train = clean_labels(train_df[attrs].values)
    y_val, m_val = clean_labels(val_df[attrs].values)
    val_groups = np.array([get_dataset_name(p) for p in val_df[img_col]])
    print(f"train: {len(train_df)}  val: {len(val_df)}  attributes: {len(attrs)}  device: {device}")

    pos_weight = torch.tensor(get_pos_weights(y_train, m_train), dtype=torch.float32, device=device)

    train_set = PedestrianDataset(train_df[img_col], y_train, m_train, args.img_root, get_transforms(True, light=args.light_aug))
    val_set = PedestrianDataset(val_df[img_col], y_val, m_val, args.img_root, get_transforms(False))
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers)

    model = build_model(args.backbone, len(attrs), pretrained=not args.no_pretrained).to(device)
    ema = EMA(model, config.EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=args.epochs * len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -1
    history = []
    start_epoch = 0
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scheduler.total_steps = args.epochs * len(train_loader)  # in case --epochs was changed
        start_epoch, best_score, history = ckpt["epoch"], ckpt["best_score"], ckpt["history"]
        print(f"resuming from epoch {start_epoch}, best score so far {best_score:.2f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.time()
        total_loss = 0
        for imgs, y, m in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            imgs, y, m = imgs.to(device), y.to(device), m.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(imgs)
            loss = weighted_bce_loss(logits.float(), y, m, pos_weight)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            total_loss += loss.item()

        probs = predict(ema.model, val_loader, device)
        res = dataset_wise_metrics(probs, y_val, m_val, np.full(len(attrs), 0.5), val_groups)
        res["epoch"] = epoch + 1
        res["loss"] = total_loss / len(train_loader)
        res["time_sec"] = round(time.time() - start)
        history.append(res)
        print(res)

        if res["score"] > best_score:
            best_score = res["score"]
            # save best model in fp16 (half the size)
            state = {k: v.half() if v.is_floating_point() else v for k, v in ema.model.state_dict().items()}
            torch.save(state, os.path.join(args.out, "best.pt"))
        json.dump(history, open(os.path.join(args.out, "log.json"), "w"), indent=1)
        # full checkpoint for --resume
        torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_score": best_score,
                    "history": history}, last_path)

    # best model + flip tta, then tune thresholds on val
    state = torch.load(os.path.join(args.out, "best.pt"), map_location=device)
    ema.model.load_state_dict({k: v.float() for k, v in state.items()})
    probs = predict(ema.model, val_loader, device, flip_tta=True)
    np.save(os.path.join(args.out, "val_probs.npy"), probs)
    base = dataset_wise_metrics(probs, y_val, m_val, np.full(len(attrs), 0.5), val_groups)
    thr = find_best_thresholds(probs, y_val, m_val, val_groups)
    tuned = dataset_wise_metrics(probs, y_val, m_val, thr, val_groups)
    print("val @0.5 :", base)
    print("val tuned:", tuned)
    json.dump({"thr": thr.tolist()}, open(os.path.join(args.out, "thresholds.json"), "w"))
    json.dump({"backbone": args.backbone, "epochs": args.epochs, "batch_size": args.bs, "lr": args.lr,
               "light_aug": args.light_aug, "base": base, "tuned": tuned},
              open(os.path.join(args.out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
