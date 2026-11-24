"""
train_resnet34.py - train a plain ResNet34 on the pre/post crops from gen_sample_data.py
and report how well it does.

Input:  pre + post RGB stacked into 6 channels, 256x256
Output: one class per crop = the worst damage in it
        0 empty   1 none   2 minor   3 major   4 destroyed

Usage (from the project root, x-view-images/):
    python train_resnet34.py
    python train_resnet34.py --epochs 30 --batch 64 --lr 1e-4 --name longer
    python train_resnet34.py --eval runs/resnet34/best.pt     # metrics only, no training
    python train_resnet34.py --limit 200 --epochs 1 --name smoke     # quick check

Everything is saved in runs/<name>/:
    best.pt           weights of the epoch with the best val macro F1
    history.csv       one row per epoch
    metrics.json      every number in the printed report
    predictions.csv   one row per val crop, most confident mistakes first
    report.png        plots

The first run downloads ImageNet weights (about 83 MB); --no-pretrained skips that.

Val picks the best epoch AND is what the report scores, so expect the numbers to be
slightly optimistic. Use a held-out test set for a final, unbiased score.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34

CLASSES = ["empty", "none", "minor", "major", "destroyed"]
N = len(CLASSES)

# Yes/no questions the 5 class probabilities can answer: positive if label >= the number
BINARY = {"has_building": 1, "any_damage": 2, "severe_damage": 3, "destroyed": 4}
TARGET_RECALL = 0.90

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------------------------------- #
# Data and model
# --------------------------------------------------------------------------- #
class Crops(Dataset):
    def __init__(self, folder, augment=False, limit=None, seed=0):
        self.folder = Path(folder)
        with open(self.folder / "labels.csv", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if limit is not None and limit < len(self.rows):
            self.rows = random.Random(seed).sample(self.rows, limit)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def load(self, name):
        with Image.open(self.folder / name) as image:
            array = np.array(image.convert("RGB"))
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255

    def __getitem__(self, i):
        row = self.rows[i]
        pre, post = self.load(row["pre"]), self.load(row["post"])
        if self.augment:
            # same flip and rotation for both, separate lighting for each
            pre, post = jitter(pre), jitter(post)
            pair = torch.cat([pre, post])
            if random.random() < 0.5:
                pair = pair.flip(2)
            pair = torch.rot90(pair, random.randrange(4), (1, 2))
            pre, post = pair[:3], pair[3:]
        x = torch.cat([(pre - MEAN) / STD, (post - MEAN) / STD])
        return x, int(row["label"])


def jitter(image, amount=0.2):
    contrast = 1 + random.uniform(-amount, amount)
    brightness = 1 + random.uniform(-amount, amount)
    mean = image.mean()
    return (((image - mean) * contrast + mean) * brightness).clamp(0, 1)


def seed_worker(_):
    random.seed(torch.initial_seed() % 2**32)


def build_model(pretrained=True):
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    # 6 input channels: reuse the RGB filters for pre and post, halved so the sum
    # reaching the next layer stays the same size
    old = model.conv1
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(torch.cat([old.weight, old.weight], dim=1) / 2)
    model.fc = nn.Linear(model.fc.in_features, N)
    return model


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    """One pass. Returns mean loss, labels and probabilities."""
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if training else None
    total, labels, probs = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        total += loss.item() * len(y)
        labels.append(y.cpu())
        probs.append(logits.detach().float().softmax(dim=1).cpu())
    labels, probs = torch.cat(labels).numpy(), torch.cat(probs).numpy()
    return total / len(labels), labels, probs


# --------------------------------------------------------------------------- #
# Metrics (plain numpy). nan means "not defined", e.g. no such class in the data
# --------------------------------------------------------------------------- #
def divide(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)


def mean_defined(values):
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else float("nan")


def confusion(labels, preds):
    """Rows are the true class, columns the predicted class."""
    return np.bincount(labels * N + preds, minlength=N * N).reshape(N, N)


def class_scores(cm):
    """Per-class precision, recall, F1. Classes absent from the labels get nan recall/F1."""
    hit = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    f1 = divide(2 * hit, support + predicted)
    f1[support == 0] = np.nan
    return divide(hit, predicted), divide(hit, support), f1


def macro_f1(cm):
    return mean_defined(class_scores(cm)[2])


def balanced_accuracy(cm):
    return mean_defined(class_scores(cm)[1])


def quadratic_kappa(cm):
    total = cm.sum()
    weights = (np.subtract.outer(np.arange(N), np.arange(N)) / (N - 1)) ** 2
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / total
    denominator = (weights * expected).sum()
    return float(1 - (weights * cm).sum() / denominator) if denominator > 0 else float("nan")


def auroc(positive, score):
    """Chance a random positive scores above a random negative (ties count half)."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]     # average rank for ties
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_curve(positive, score):
    """Precision, recall and threshold at every distinct score, highest score first."""
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(positive[order])
    last_of_tie = np.r_[np.diff(score[order]) != 0, True]
    flagged = np.arange(1, len(score) + 1)
    return (hits[last_of_tie] / flagged[last_of_tie],
            divide(hits[last_of_tie], positive.sum()),
            score[order][last_of_tie])


def average_precision(positive, score):
    if positive.sum() == 0:
        return float("nan")
    precision, recall, _ = pr_curve(positive, score)
    return float((np.diff(np.r_[0, recall]) * precision).sum())


def calibration(labels, probs, bins=10):
    """Expected calibration error, plus (mean confidence, accuracy, share) per bin."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    which = np.minimum((confidence * bins).astype(int), bins - 1)
    table = [(float(confidence[which == b].mean()), float(correct[which == b].mean()),
              float((which == b).mean())) for b in range(bins) if (which == b).any()]
    return float(sum(share * abs(acc - conf) for conf, acc, share in table)), table


def binary_report(positive, score):
    flagged = score >= 0.5
    hit = int((flagged & positive).sum())
    out = {
        "positives": int(positive.sum()),
        "prevalence": float(positive.mean()),
        "auroc": auroc(positive, score),
        "ap": average_precision(positive, score),
        "precision": float(divide(hit, flagged.sum())),
        "recall": float(divide(hit, positive.sum())),
        "f1": float(divide(2 * hit, flagged.sum() + positive.sum())),
    }
    # Cost of catching TARGET_RECALL of the positives
    if positive.any():
        precision, recall, threshold = pr_curve(positive, score)
        i = int(np.argmax(recall >= TARGET_RECALL - 1e-12))
        out["at_target_recall"] = {"threshold": float(threshold[i]), "precision": float(precision[i]),
                                   "recall": float(recall[i]),
                                   "share_flagged": float((score >= threshold[i]).mean())}
    return out


def scene_interval(cms, score, rounds=1000, seed=0):
    """95% interval of a confusion-matrix score, resampling whole scenes."""
    rng = np.random.default_rng(seed)
    cms = np.asarray(cms)
    values = [score(cms[rng.integers(0, len(cms), len(cms))].sum(axis=0)) for _ in range(rounds)]
    return [float(v) for v in np.nanpercentile(values, [2.5, 97.5])]


def recall_of(c):
    def score(cm):
        return class_scores(cm)[1][c]
    return score


def disaster_of(row):
    return row["scene"].rsplit("_", 1)[0]


def evaluate(labels, probs, rows):
    preds = probs.argmax(axis=1)
    cm = confusion(labels, preds)
    precision, recall, f1 = class_scores(cm)
    support = cm.sum(axis=1)
    ece, calibration_table = calibration(labels, probs)
    off_by = np.abs(preds - labels)

    scenes = {}
    for row, label, pred in zip(rows, labels, preds):
        scenes.setdefault(row["scene"], []).append((label, pred))
    scene_cms = [confusion(*np.array(v).T) for v in scenes.values()]

    per_class = {}
    for c, name in enumerate(CLASSES):
        per_class[name] = {
            "support": int(support[c]), "predicted": int(cm[:, c].sum()),
            "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c]),
            "recall_95": scene_interval(scene_cms, recall_of(c)) if support[c] else [float("nan")] * 2,
            "auroc": auroc(labels == c, probs[:, c]),
            "ap": average_precision(labels == c, probs[:, c]),
        }

    disasters = np.array([disaster_of(r) for r in rows])
    per_disaster = {}
    for name in sorted(set(disasters)):
        m = disasters == name
        dcm = confusion(labels[m], preds[m])
        damaged = labels[m] >= 2
        per_disaster[name] = {
            "n": int(m.sum()), "damaged": int(damaged.sum()),
            "accuracy": float((preds[m] == labels[m]).mean()),
            "macro_f1": macro_f1(dcm), "balanced_accuracy": balanced_accuracy(dcm),
            "any_damage_auroc": auroc(damaged, probs[m, 2:].sum(axis=1)),
            "destroyed": int((labels[m] == 4).sum()),
            "destroyed_recall": float(class_scores(dcm)[1][4]),
        }

    majority = np.zeros_like(cm)
    majority[:, support.argmax()] = support

    return {
        "n": len(labels),
        "overall": {
            "accuracy": float((preds == labels).mean()),
            "balanced_accuracy": balanced_accuracy(cm),
            "macro_f1": macro_f1(cm),
            "macro_f1_95": scene_interval(scene_cms, macro_f1),
            "weighted_f1": float(np.nansum(f1 * support) / support.sum()),
            "quadratic_kappa": quadratic_kappa(cm),
            "mean_class_error": float(off_by.mean()),
            "off_by_2_or_more": float((off_by >= 2).mean()),
            "log_loss": float(-np.log(probs[np.arange(len(labels)), labels].clip(1e-12)).mean()),
            "ece": ece,
            "mean_confidence": float(probs.max(axis=1).mean()),
        },
        "baseline_always_" + CLASSES[support.argmax()]: {
            "accuracy": float(support.max() / support.sum()),
            "macro_f1": macro_f1(majority), "balanced_accuracy": balanced_accuracy(majority),
        },
        "per_class": per_class,
        "binary": {k: binary_report(labels >= c, probs[:, c:].sum(axis=1)) for k, c in BINARY.items()},
        "per_disaster": per_disaster,
        "confusion": cm.tolist(),
        "calibration": calibration_table,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def num(v, width=9):
    return f"{'-' if v is None or np.isnan(v) else f'{v:.3f}':>{width}}"


def print_report(m):
    o = m["overall"]
    base_name = next(k for k in m if k.startswith("baseline_"))
    base = m[base_name]
    lo, hi = o["macro_f1_95"]
    print(f"\n=== Overall ({m['n']} val crops) ===")
    print(f"{'':24}{'model':>9}{base_name[9:].replace('_', ' '):>16}")
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        print(f"{key:24}{num(o[key])}{num(base[key], 16)}")
    print(f"macro F1 95% interval, resampling scenes: {lo:.3f} to {hi:.3f}")
    for key, note in (("weighted_f1", ""), ("quadratic_kappa", "1 = perfect, 0 = chance; far-off mistakes cost more"),
                      ("mean_class_error", "average number of classes off"),
                      ("off_by_2_or_more", "share of crops"), ("log_loss", "lower is better"),
                      ("ece", "calibration error, 0 = confidence matches accuracy"),
                      ("mean_confidence", "compare with accuracy")):
        print(f"{key:24}{num(o[key])}  {note}".rstrip())
    if "train_check" in m:
        t = m["train_check"]
        print("\n=== Overfitting check (balanced accuracy, no augmentation) ===")
        print(f"train {t['train_balanced_accuracy']:.3f}   val {o['balanced_accuracy']:.3f}   "
              f"gap {t['gap']:+.3f}   ({t['n']} train crops)")
        print("Train crops are sampled differently from val, so read a large gap as a warning, not proof.")

    print("\n=== Per class (AP for a random guess = share) ===")
    print(f"{'':11}{'support':>9}{'share':>9}{'predicted':>11}{'precision':>11}{'recall':>9}"
          f"{'(95% interval)':>17}{'f1':>9}{'auroc':>9}{'ap':>9}")
    for name, c in m["per_class"].items():
        span = f"{c['recall_95'][0]:.2f} to {c['recall_95'][1]:.2f}" if c["support"] else "-"
        print(f"{name:11}{c['support']:9d}{c['support'] / m['n']:9.3f}{c['predicted']:11d}"
              f"{num(c['precision'], 11)}{num(c['recall'])}{span:>17}{num(c['f1'])}{num(c['auroc'])}{num(c['ap'])}")

    print("\n=== Confusion matrix (rows = true, columns = predicted) ===")
    print(f"{'':11}" + "".join(f"{n:>11}" for n in CLASSES))
    for name, row in zip(CLASSES, m["confusion"]):
        print(f"{name:11}" + "".join(f"{v:11d}" for v in row))

    print(f"\n=== Yes/no questions (precision/recall/F1 at probability 0.5; last columns: "
          f"threshold that catches {TARGET_RECALL:.0%}) ===")
    print(f"{'':15}{'share':>8}{'auroc':>8}{'ap':>8}{'prec':>8}{'recall':>8}{'f1':>8}"
          f"{'thresh':>10}{'prec':>8}{'flagged':>9}")
    for name, b in m["binary"].items():
        t = b.get("at_target_recall", {})
        print(f"{name:15}" + "".join(num(b[k], 8) for k in ("prevalence", "auroc", "ap", "precision", "recall", "f1"))
              + num(t.get("threshold"), 10) + num(t.get("precision"), 8) + num(t.get("share_flagged"), 9))

    print("\n=== Per disaster ===")
    print(f"{'':20}{'n':>6}{'damaged':>9}{'acc':>8}{'bal acc':>9}{'macro F1':>10}{'dmg AUROC':>11}"
          f"{'destroyed':>11}{'recall':>8}")
    for name, d in m["per_disaster"].items():
        print(f"{name:20}{d['n']:6d}{d['damaged']:9d}{num(d['accuracy'], 8)}{num(d['balanced_accuracy'])}"
              f"{num(d['macro_f1'], 10)}{num(d['any_damage_auroc'], 11)}{d['destroyed']:11d}"
              f"{num(d['destroyed_recall'], 8)}")


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f3f7fd", "#2a78d6", "#0d366b"])


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def unit_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_plots(path, m, labels, probs, history):
    fig, axes = plt.subplots(3, 4, figsize=(17, 12), facecolor=SURFACE, layout="constrained")

    epochs = [h["epoch"] for h in history]
    if epochs:
        ax = axes[0, 0]
        ax.plot(epochs, [h["train_loss"] for h in history], color=BLUE, lw=2, marker="o", ms=4, label="train")
        ax.plot(epochs, [h["val_loss"] for h in history], color=ORANGE, lw=2, marker="o", ms=4, label="val")
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
        style(ax, "Loss (val rising = overfitting)", "epoch")
        ax = axes[0, 1]
        ax.plot(epochs, [h["val_macro_f1"] for h in history], color=BLUE, lw=2, marker="o", ms=4, label="macro F1")
        ax.plot(epochs, [h["val_f1_destroyed"] for h in history], color=ORANGE, lw=2, marker="o", ms=4,
                label="destroyed F1")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
        style(ax, "Val F1", "epoch")
        for ax in axes[0, :2]:
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.xaxis.get_major_locator().set_params(integer=True)
    else:
        for ax in axes[0, :2]:
            ax.axis("off")
            ax.text(0.5, 0.5, "no training history (--eval)", ha="center", color=MUTED)

    cm = np.array(m["confusion"])
    share = np.nan_to_num(divide(cm, cm.sum(axis=1, keepdims=True)))
    ax = axes[0, 2]
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{share[i, j]:.0%}\n{cm[i, j]}", ha="center", va="center", fontsize=7,
                    color="white" if share[i, j] > 0.4 else INK)
    ax.set_xticks(range(N), CLASSES)
    ax.set_yticks(range(N), CLASSES)
    style(ax, "Confusion (share of each true class)", "predicted", "true")

    ax = axes[0, 3]
    conf, acc, _ = np.array(m["calibration"]).T
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.plot(conf, acc, color=BLUE, lw=2, marker="o", ms=5)
    unit_axes(ax)
    style(ax, f"Calibration, ECE {m['overall']['ece']:.3f}",
          "confidence (below dashes = overconfident)", "accuracy")

    for ax, (name, c) in zip(axes[1], BINARY.items()):
        positive, score = labels >= c, probs[:, c:].sum(axis=1)
        b = m["binary"][name]
        if positive.any():
            precision, recall, _ = pr_curve(positive, score)
            ax.plot(recall, precision, color=BLUE, lw=2)
            ax.axhline(b["prevalence"], color=MUTED, lw=1, ls="--")
        unit_axes(ax)
        style(ax, f"{name}: AP {b['ap']:.3f} (guess {b['prevalence']:.3f})", "recall", "precision")

    y = np.arange(N)
    for ax, key, title in ((axes[2, 0], "f1", "F1 per class"), (axes[2, 1], "recall", "Recall per class")):
        values = [m["per_class"][n][key] for n in CLASSES]
        ax.barh(y, np.nan_to_num(values), color=BLUE, height=0.55)
        for i, v in enumerate(values):
            ax.text((0 if np.isnan(v) else v) + 0.02, i, "n/a" if np.isnan(v) else f"{v:.2f}",
                    va="center", fontsize=8, color=INK)
        if key == "recall":
            for i, n in enumerate(CLASSES):
                ax.plot(m["per_class"][n]["recall_95"], [i + 0.38] * 2, color=MUTED, lw=1.5)
            title += " (gray line = 95% interval)"
        ax.set_yticks(y, CLASSES)
        ax.set_xlim(0, 1.12)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.invert_yaxis()
        style(ax, title)

    names = list(m["per_disaster"])
    for ax, key, title in ((axes[2, 2], "macro_f1", "Macro F1 per disaster"),
                           (axes[2, 3], "destroyed_recall", "Destroyed recall per disaster (n)")):
        values = [m["per_disaster"][n][key] for n in names]
        ax.barh(range(len(names)), np.nan_to_num(values), color=BLUE, height=0.55)
        for i, (n, v) in enumerate(zip(names, values)):
            text = "n/a" if np.isnan(v) else f"{v:.2f}"
            if key == "destroyed_recall":
                text += f"  ({m['per_disaster'][n]['destroyed']})"
            ax.text((0 if np.isnan(v) else v) + 0.02, i, text, va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlim(0, 1.2)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.invert_yaxis()
        style(ax, title)

    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def save_predictions(path, rows, labels, probs):
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    order = sorted(range(len(rows)), key=lambda i: (preds[i] == labels[i], -confidence[i]))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["correct", "true", "predicted", "confidence"]
                        + [f"p_{c}" for c in CLASSES] + ["scene", "pre", "post"])
        for i in order:
            writer.writerow([int(preds[i] == labels[i]), CLASSES[labels[i]], CLASSES[preds[i]],
                             f"{confidence[i]:.4f}"] + [f"{p:.4f}" for p in probs[i]]
                            + [rows[i]["scene"], rows[i]["pre"], rows[i]["post"]])


def jsonable(v):
    """nan -> null, so metrics.json is valid JSON."""
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return None if isinstance(v, float) and np.isnan(v) else v


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Train and score ResNet34 on pre/post damage crops")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--name", default="resnet34", help="output folder under runs/")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu'; default auto")
    ap.add_argument("--no-pretrained", action="store_true", help="start from random weights")
    ap.add_argument("--eval", type=Path, default=None, metavar="WEIGHTS", help="score these weights, no training")
    ap.add_argument("--limit", type=int, default=None, help="use only this many crops per split")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    out = Path("runs") / (args.name if args.eval is None else f"{args.name}_eval")
    out.mkdir(parents=True, exist_ok=True)

    def loader(dataset, shuffle=False):
        return DataLoader(dataset, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                          pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                          worker_init_fn=seed_worker)

    val_set = Crops(args.data / "val", limit=args.limit)
    val_loader = loader(val_set)
    history = []

    if args.eval is None:
        train_set = Crops(args.data / "train", augment=True, limit=args.limit)
        train_loader = loader(train_set, shuffle=True)
        print(f"{len(train_set)} train crops, {len(val_set)} val crops")
        model = build_model(pretrained=not args.no_pretrained).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)

        best = -1.0
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, _, _ = run_epoch(model, train_loader, device, optimizer, scheduler)
            val_loss, labels, probs = run_epoch(model, val_loader, device)
            cm = confusion(labels, probs.argmax(axis=1))
            f1 = class_scores(cm)[2]
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_accuracy": float((probs.argmax(axis=1) == labels).mean()),
                   "val_balanced_accuracy": balanced_accuracy(cm), "val_macro_f1": macro_f1(cm),
                   **{f"val_f1_{c}": float(f1[i]) for i, c in enumerate(CLASSES)},
                   "seconds": time.time() - start}
            history.append(row)
            star = ""
            if row["val_macro_f1"] > best:
                best, star = row["val_macro_f1"], "  *best"
                torch.save(model.state_dict(), out / "best.pt")
            print(f"epoch {epoch:3d}/{args.epochs}  train loss {train_loss:.3f}  val loss {val_loss:.3f}  "
                  f"acc {row['val_accuracy']:.3f}  macro F1 {row['val_macro_f1']:.3f}  "
                  f"destroyed F1 {row['val_f1_destroyed']:.3f}  ({row['seconds']:.0f}s){star}", flush=True)
            with open(out / "history.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerows(history)

    weights = args.eval or out / "best.pt"
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    _, labels, probs = run_epoch(model, val_loader, device)
    metrics = evaluate(labels, probs, val_set.rows)
    metrics["weights"] = str(weights)

    if (args.data / "train" / "labels.csv").exists():
        check_set = Crops(args.data / "train", limit=min(2000, args.limit or 2000), seed=args.seed)
        _, train_labels, train_probs = run_epoch(model, loader(check_set), device)
        score = balanced_accuracy(confusion(train_labels, train_probs.argmax(axis=1)))
        metrics["train_check"] = {"n": len(check_set), "train_balanced_accuracy": score,
                                  "gap": score - metrics["overall"]["balanced_accuracy"]}

    print_report(metrics)
    (out / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2))
    save_predictions(out / "predictions.csv", val_set.rows, labels, probs)
    save_plots(out / "report.png", metrics, labels, probs, history)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
