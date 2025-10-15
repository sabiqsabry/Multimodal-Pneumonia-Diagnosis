import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

FIG_DIR = "outputs/figures"
THRESH_FILE = os.path.join(FIG_DIR, "cxr_real_emr_weighted_threshold_comparison.json")

LOG_FILES = {
    "CXR-only": "outputs/logs/cxr_only_train_log.csv",
    "EMR-only": "outputs/logs/real_emr_train_log.csv",
    "Fusion (Unweighted)": "outputs/logs/cxr_real_emr_train_log.csv",
    "Fusion (Weighted)": "outputs/logs/cxr_real_emr_fusion_weighted_train_log.csv",
}

os.makedirs(FIG_DIR, exist_ok=True)

# Figure 4.5: Sensitivity vs Specificity across thresholds (line)
if os.path.exists(THRESH_FILE):
    with open(THRESH_FILE, "r") as f:
        data = json.load(f)
    ths = sorted([float(k) for k in data.keys()])
    sens = [data[str(t)]["sensitivity"] for t in ths]
    spec = [data[str(t)]["specificity"] for t in ths]

    plt.figure(figsize=(6, 4))
    plt.plot(ths, sens, marker="o", label="Sensitivity")
    plt.plot(ths, spec, marker="s", label="Specificity")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Sensitivity vs Specificity across thresholds (Weighted Fusion)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "sensitivity_specificity_thresholds.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

    # Bar chart version (grouped bars)
    x = np.arange(len(ths))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, sens, width, label="Sensitivity")
    plt.bar(x + width/2, spec, width, label="Specificity")
    plt.xticks(x, [str(t) for t in ths])
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Sensitivity & Specificity by Threshold (Weighted Fusion)")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    bar_out = os.path.join(FIG_DIR, "sensitivity_specificity_thresholds_bar.png")
    plt.savefig(bar_out, dpi=300)
    print(f"Saved: {bar_out}")
else:
    print(f"Missing threshold file: {THRESH_FILE}")

# Combined Training Curves: Loss & Accuracy for all models in one image
# Layout: 2 rows x 4 columns -> [Loss panels for 4 models] on row 1, [Accuracy panels for 4 models] on row 2
fig = plt.figure(figsize=(16, 7))
ax_grid = GridSpec(2, 4, figure=fig, wspace=0.25, hspace=0.35)
model_names = list(LOG_FILES.keys())

for col, name in enumerate(model_names):
    csv_path = LOG_FILES[name]
    if not os.path.exists(csv_path):
        print(f"Missing log file: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    df = df[df["split"].isin(["train", "val"])].copy()

    # Loss subplot (row 0)
    ax_loss = fig.add_subplot(ax_grid[0, col])
    for sp in ["train", "val"]:
        sub = df[df["split"] == sp]
        ax_loss.plot(sub["epoch"], sub["loss"], marker="o", label=f"{sp}")
    ax_loss.set_title(f"{name} — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    if col == 0:
        ax_loss.legend()

    # Accuracy subplot (row 1)
    ax_acc = fig.add_subplot(ax_grid[1, col])
    for sp in ["train", "val"]:
        sub = df[df["split"] == sp]
        ax_acc.plot(sub["epoch"], sub["accuracy"], marker="o", label=f"{sp}")
    ax_acc.set_title(f"{name} — Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)
    if col == 0:
        ax_acc.legend()

combined_out = os.path.join(FIG_DIR, "training_curves_combined.png")
plt.tight_layout()
fig.savefig(combined_out, dpi=300)
print(f"Saved: {combined_out}")
