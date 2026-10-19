# evaluate.py
# scores our val probabilities against the official val labels
# makes the tables and graphs in results/ for the report
#   python evaluate.py   (uses runs/resnet50/val_probs.npy and thresholds.json)

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from dataset import clean_labels, get_dataset_name, read_annotations
from metrics import compute_metrics, dataset_wise_metrics, find_best_thresholds

parser = argparse.ArgumentParser()
parser.add_argument("--probs", default=config.VAL_PROBS)
parser.add_argument("--thr", default=config.THRESHOLDS_JSON)
parser.add_argument("--out", default=config.RESULTS_DIR)
args = parser.parse_args()

out = args.out
os.makedirs(out, exist_ok=True)

df, img_col, attrs = read_annotations(config.VAL_GT_CSV)
y, mask = clean_labels(df[attrs].values)
groups = np.array([get_dataset_name(p) for p in df[img_col]])
datasets = list(np.unique(groups))
probs = np.load(args.probs)
n_attr = len(attrs)
print(f"val images: {len(df)}, attributes: {n_attr}")
print("images per dataset:", {str(d): int((groups == d).sum()) for d in datasets})

tuned_thr = np.array(json.load(open(args.thr))["thr"], dtype=np.float32)
half_thr = np.full(n_attr, 0.5, dtype=np.float32)

# baseline: predict the positive rate of each attribute for every image
prior = json.load(open(config.PRIOR_JSON))
prior_map = dict(zip(prior["attribute_names"], prior["positive_rate"]))
prior_probs = np.tile(np.array([prior_map[a] for a in attrs], dtype=np.float32), (len(df), 1))

# 1) score per dataset
methods = {
    "Prior baseline": (prior_probs, half_thr),
    "Our model @0.5": (probs, half_thr),
    "Our model + tuned thr": (probs, tuned_thr),
}
rows = []
for name, (p, t) in methods.items():
    for d in datasets:
        idx = groups == d
        r = compute_metrics(p[idx], y[idx], mask[idx], t)
        rows.append({"method": name, "dataset": d, "mA": round(r["mA"], 2), "F1": round(r["F1"], 2), "score": round(r["score"], 2)})
    r = dataset_wise_metrics(p, y, mask, t, groups)
    rows.append({"method": name, "dataset": "MEAN", "mA": round(r["mA"], 2), "F1": round(r["F1"], 2), "score": round(r["score"], 2)})
per_dataset = pd.DataFrame(rows)
per_dataset.to_csv(os.path.join(out, "per_dataset.csv"), index=False)
print("\n", per_dataset.to_string(index=False))

# 2) per attribute results, tuned thresholds
pred = (probs >= tuned_thr[None]).astype(np.float32)
tp = (pred * y).sum(0)
fp = (pred * (1 - y)).sum(0)
fn = ((1 - pred) * y).sum(0)
tn = ((1 - pred) * (1 - y)).sum(0)
pos = y.sum(0)
neg = (1 - y).sum(0)
tpr = tp / np.maximum(pos, 1)
tnr = tn / np.maximum(neg, 1)
f1_attr = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
pos_rate = 100 * pos / len(y)

per_attr = pd.DataFrame({
    "attribute": attrs,
    "positive_rate": pos_rate,
    "TPR": 100 * tpr,
    "TNR": 100 * tnr,
    "mA": 50 * (tpr + tnr),
    "F1": 100 * f1_attr,
    "threshold": tuned_thr,
}).round(2)


def attr_group(name):
    # "UpperBody-Color-Red" -> "UpperBody-Color", "Age-Young" -> "Age"
    parts = name.split("-")
    if parts[0] in ["Age", "Gender", "Hair", "Accessory"]:
        return parts[0]
    return parts[0] + "-" + parts[1]


per_attr["group"] = [attr_group(a) for a in attrs]
per_attr.sort_values("mA").to_csv(os.path.join(out, "per_attribute.csv"), index=False)
per_group = per_attr.groupby("group").agg(n_attr=("attribute", "count"), mA=("mA", "mean"), F1=("F1", "mean")).astype(float).round(1).sort_values("mA")
per_group.to_csv(os.path.join(out, "per_group.csv"))
print("\n", per_group.to_string())

rare = int((pos_rate < 5).sum())
print(f"\nattributes with < 5% positives: {rare} / {n_attr}")

# 3) check threshold tuning: tune on one half, test on the other
rng = np.random.RandomState(0)
half_a = np.zeros(len(df), dtype=bool)
for d in datasets:
    idx = np.where(groups == d)[0]
    half_a[rng.permutation(idx)[: len(idx) // 2]] = True

split_rows = []
for tune_idx, test_idx, name in [(half_a, ~half_a, "tune A -> test B"), (~half_a, half_a, "tune B -> test A")]:
    t = find_best_thresholds(probs[tune_idx], y[tune_idx], mask[tune_idx], groups[tune_idx])
    s_half = dataset_wise_metrics(probs[test_idx], y[test_idx], mask[test_idx], half_thr, groups[test_idx])["score"]
    s_tuned = dataset_wise_metrics(probs[test_idx], y[test_idx], mask[test_idx], t, groups[test_idx])["score"]
    split_rows.append({"split": name, "score_at_0.5": round(s_half, 2), "score_tuned": round(s_tuned, 2), "gain": round(s_tuned - s_half, 2)})
split_df = pd.DataFrame(split_rows)
split_df.to_csv(os.path.join(out, "threshold_check.csv"), index=False)
print("\n", split_df.to_string(index=False))

summary = {
    "n_images": len(df),
    "n_attributes": n_attr,
    "score_at_0.5": per_dataset.query("method == 'Our model @0.5' and dataset == 'MEAN'")["score"].item(),
    "score_tuned": per_dataset.query("method == 'Our model + tuned thr' and dataset == 'MEAN'")["score"].item(),
    "prior_baseline": per_dataset.query("method == 'Prior baseline' and dataset == 'MEAN'")["score"].item(),
    "rare_attributes_below_5pct": rare,
    "macro_F1_per_attribute": round(float(100 * f1_attr.mean()), 2),
    "threshold_check": split_rows,
}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)

# graphs
# red = rare attribute
s = per_attr.sort_values("mA")
fig, ax = plt.subplots(figsize=(9, 9))
colors = ["#c0392b" if r < 5 else "#2c7fb8" for r in s["positive_rate"]]
ax.barh(s["attribute"], s["mA"], color=colors)
ax.set_xlim(50, 100)
ax.set_xlabel("mA (%)   (red = less than 5% positive images)")
ax.set_title("mA for each attribute (tuned thresholds)")
ax.tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(os.path.join(out, "fig_per_attribute_mA.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(pos_rate, 50 * (tpr + tnr), color="#2c7fb8")
ax.set_xscale("log")
ax.set_xlabel("positive rate in val set (%, log scale)")
ax.set_ylabel("mA (%)")
for i in np.argsort(50 * (tpr + tnr))[:4]:
    ax.annotate(attrs[i], (pos_rate[i], 50 * (tpr[i] + tnr[i])), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_title("Class imbalance vs mA")
fig.tight_layout()
fig.savefig(os.path.join(out, "fig_imbalance_vs_mA.png"), dpi=130)
plt.close(fig)

plot_df = per_dataset[per_dataset["dataset"] != "MEAN"].pivot(index="dataset", columns="method", values="score")
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_df.plot.bar(ax=ax, rot=0)
ax.set_ylabel("challenge score (%)")
ax.set_ylim(40, 95)
ax.set_title("Score on each dataset")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "fig_score_per_dataset.png"), dpi=130)
plt.close(fig)

print("\nsaved everything in", out)
