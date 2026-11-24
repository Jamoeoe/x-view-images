"""
evaluate.py - standalone, repeatable scoring for xView2 damage-detection models.

Does NOT touch training. Point it at any YOLO weights file and it will:

  1. Standard detection metrics (Ultralytics val): mAP50 / mAP50-95 per class,
     plus "damage mAP50" = mean mAP50 over minor/major/destroyed (the headline
     number to compare runs on).
  2. Grid-cell triage metric: every full image is cut into a GxG grid *at scoring
     time* (no pre-split images needed). Each cell is labelled positive if it has
     >= K buildings of the target classes, and scored by the model's K-th highest
     confidence for those classes in that cell. Reports AUROC and average
     precision (threshold-free), plus precision/recall/F1 at a fixed confidence.

Always evaluates the ORIGINAL full-resolution images, so scores stay comparable
no matter how a model was trained (full images, tiles, different imgsz...).

Each run appends one row to a CSV so you can compare models side by side.

Usage (from the project root, x-view-images/):
    python evaluate.py --weights X_View/datasets/data/Xview_mods/Trial/weights/best.pt
    python evaluate.py --weights path/to/best.pt --name yolov8s_1024_50ep --grid 4
    python evaluate.py --weights path/to/best.pt --skip-map      # grid metric only (faster)
"""

import argparse
import csv
import math
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Class names in data.yaml -> which ones count for each grid target.
# Edit here if your class list changes.
GRID_TARGETS = {
    "critical": ["major_damage", "destroyed"],
    "any_damage": ["minor_damage", "major_damage", "destroyed"],
}
DAMAGE_CLASSES = ["minor_damage", "major_damage", "destroyed"]


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def load_dataset(data_yaml: Path, split: str):
    """Return (class_names, image_dir, resolved_yaml_dict) with absolute paths.

    The repo's data.yaml uses relative paths (path: ./X_View/), which Ultralytics
    resolves differently depending on its settings. We resolve them ourselves and
    hand Ultralytics a temp yaml with absolute paths so it can't get lost.
    """
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    base = cfg.get("path", ".")
    candidates = [Path.cwd() / base, data_yaml.parent / base, data_yaml.parent]
    split_rel = cfg.get(split)
    if split_rel is None:
        sys.exit(f"[error] split '{split}' not defined in {data_yaml}")

    for root in candidates:
        if (root / split_rel).is_dir():
            root = root.resolve()
            break
    else:
        sys.exit(f"[error] could not find '{split_rel}' relative to any of: "
                 + ", ".join(str(c) for c in candidates))

    resolved = {
        "path": str(root),
        # train isn't used for scoring; a list (e.g. data_xbd.yaml) is fine too
        "train": str(root / cfg["train"]) if isinstance(cfg.get("train"), str) else str(root / split_rel),
        "val": str(root / split_rel),   # ultralytics val reads the 'val' key
        "nc": len(names),
        "names": names,
    }
    return names, root / split_rel, resolved


def label_path_for(img_path: Path) -> Path:
    """YOLO convention: .../images/x.png -> .../labels/x.txt"""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_gt(lbl_path: Path):
    """Return (cls[int], xc[float], yc[float]) normalized, or empty arrays."""
    if not lbl_path.is_file():
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    rows = [ln.split() for ln in lbl_path.read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    arr = np.array([[float(v) for v in r[:5]] for r in rows])
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2]


# --------------------------------------------------------------------------- #
# Grid metric
# --------------------------------------------------------------------------- #
def cell_index(xc, yc, g):
    gx = np.clip((np.asarray(xc) * g).astype(int), 0, g - 1)
    gy = np.clip((np.asarray(yc) * g).astype(int), 0, g - 1)
    return gy * g + gx


def grid_cells(gt_cls, gt_xc, gt_yc, pr_cls, pr_xc, pr_yc, pr_conf,
               target_ids, g, min_count):
    """Per-cell (label, score) for one image."""
    labels = np.zeros(g * g, dtype=bool)
    scores = np.zeros(g * g, dtype=float)

    m = np.isin(gt_cls, target_ids)
    if m.any():
        counts = np.bincount(cell_index(gt_xc[m], gt_yc[m], g), minlength=g * g)
        labels = counts >= min_count

    m = np.isin(pr_cls, target_ids)
    if m.any():
        idx = cell_index(pr_xc[m], pr_yc[m], g)
        conf = pr_conf[m]
        for c in np.unique(idx):
            cc = np.sort(conf[idx == c])[::-1]
            if len(cc) >= min_count:
                scores[c] = cc[min_count - 1]   # K-th highest confidence
    return labels, scores


def auroc(y, s):
    """Rank-based AUROC (Mann-Whitney U), handles ties. NaN if one class only."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):                       # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, s):
    """Area under the precision-recall curve (step-wise, like sklearn)."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    # only evaluate at the last index of each distinct score (ties grouped)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    prec = tp[last] / (tp[last] + fp[last])
    rec = tp[last] / y.sum()
    rec_prev = np.r_[0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def prf_at(y, s, thr):
    y = np.asarray(y, bool)
    p = np.asarray(s) >= thr
    tp = (p & y).sum()
    fp = (p & ~y).sum()
    fn = (~p & y).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return float(prec), float(rec), float(f1)


def best_f1(y, s):
    best = (0.0, 0.0)
    for thr in np.unique(s[s > 0]):
        f1 = prf_at(y, s, thr)[2]
        if f1 > best[0]:
            best = (f1, float(thr))
    return best


# --------------------------------------------------------------------------- #
# Training-run info (read from the Ultralytics run folder next to the weights)
# --------------------------------------------------------------------------- #
def training_info(weights: Path):
    """Pull model, epochs and wall-clock training time from <run>/results.csv + args.yaml.

    Ultralytics' results.csv has a 'time' column = cumulative seconds at the end of
    each epoch. Returns blanks if the files aren't there (e.g. downloaded weights).
    """
    run_dir = weights.parent.parent
    info = {"train_model": "", "train_epochs": "", "train_time_min": "",
            "train_sec_per_epoch": "", "train_batch": ""}

    args_file = run_dir / "args.yaml"
    if args_file.is_file():
        try:
            a = yaml.safe_load(args_file.read_text()) or {}
            info["train_model"] = Path(str(a.get("model", ""))).name
            info["train_batch"] = a.get("batch", "")
        except Exception:
            pass

    res_file = run_dir / "results.csv"
    if res_file.is_file():
        try:
            with open(res_file, newline="") as f:
                rows = [{k.strip(): v.strip() for k, v in r.items() if k}
                        for r in csv.DictReader(f)]
            times = [float(r["time"]) for r in rows if r.get("time")]
            if rows:
                info["train_epochs"] = len(rows)
            if times:
                info["train_time_min"] = round(times[-1] / 60, 1)
                # steady-state speed: skip epoch 1, which includes startup/warm-up
                diffs = np.diff(times)
                per_epoch = float(np.median(diffs)) if len(diffs) else times[0]
                info["train_sec_per_epoch"] = round(per_epoch, 1)
        except Exception:
            pass
    return info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="path to best.pt / last.pt")
    ap.add_argument("--data", default="X_View/datasets/data/data.yaml")
    ap.add_argument("--split", default="val", help="split key in data.yaml (val/test)")
    ap.add_argument("--name", default=None, help="label for this run in the CSV")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference size; default = the size the model was trained at")
    ap.add_argument("--nms-iou", type=float, default=0.7,
                    help="NMS IoU for both mAP and grid passes (Ultralytics val default 0.7)")
    ap.add_argument("--max-det", type=int, default=1000,
                    help="max boxes per image (some scenes have 300+ buildings)")
    ap.add_argument("--grid", type=int, default=4, help="GxG cells per image (4 -> 256px cells)")
    ap.add_argument("--min-count", type=int, default=1,
                    help="cell is positive if it has >= this many target buildings")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold for the reported grid precision/recall/F1")
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu'; default auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-map", action="store_true", help="skip the Ultralytics mAP pass")
    ap.add_argument("--out", default="eval_results.csv", help="CSV to append results to")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    weights = Path(args.weights)
    if not weights.is_file():
        sys.exit(f"[error] weights not found: {weights}")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    names, img_dir, resolved = load_dataset(Path(args.data), args.split)
    name_to_id = {n: i for i, n in enumerate(names)}

    model = YOLO(str(weights))
    imgsz = args.imgsz or int(model.overrides.get("imgsz", 1024) or 1024)
    run_name = args.name or f"{weights.parent.parent.name}/{weights.stem}"

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"\nModel   : {weights}")
    print(f"Images  : {len(images)} in {img_dir}")
    print(f"imgsz={imgsz}  nms_iou={args.nms_iou}  device={device}  grid={args.grid}x{args.grid}"
          f"  min_count={args.min_count}\n")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": run_name,
        "weights": str(weights),
        "split": args.split,
        "n_images": len(images),
        "imgsz": imgsz,
        "nms_iou": args.nms_iou,
    }
    tinfo = training_info(weights)
    row.update(tinfo)
    if tinfo["train_time_min"] != "":
        print(f"Training: {tinfo['train_model']}  {tinfo['train_epochs']} epochs  "
              f"{tinfo['train_time_min']} min  (~{tinfo['train_sec_per_epoch']} s/epoch)\n")

    # ---------------- 1. Standard detection metrics ---------------- #
    if not args.skip_map:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_yaml = Path(tmp) / "data_abs.yaml"
            tmp_yaml.write_text(yaml.safe_dump(resolved))
            t0 = time.time()
            m = model.val(data=str(tmp_yaml), split="val", imgsz=imgsz, batch=args.batch,
                          conf=0.001, iou=args.nms_iou, max_det=args.max_det,
                          device=device, plots=False, save_json=False,
                          project=tmp, name="val", verbose=False)
            print(f"[mAP pass done in {time.time() - t0:.0f}s]")

        box = m.box
        ap50 = {i: float("nan") for i in range(len(names))}
        ap5095 = dict(ap50)
        for k, ci in enumerate(box.ap_class_index):
            ap50[int(ci)] = float(box.ap50[k])
            ap5095[int(ci)] = float(box.ap[k])

        dmg_ids = [name_to_id[n] for n in DAMAGE_CLASSES if n in name_to_id]
        dmg_map50 = float(np.nanmean([ap50[i] for i in dmg_ids]))
        row.update({
            "damage_mAP50": round(dmg_map50, 4),
            "mAP50_all": round(float(box.map50), 4),
            "mAP50-95_all": round(float(box.map), 4),
        })
        for i, n in enumerate(names):
            row[f"AP50_{n}"] = round(ap50[i], 4)

        print(f"{'class':<14}{'AP50':>8}{'AP50-95':>10}")
        for i, n in enumerate(names):
            print(f"{n:<14}{ap50[i]:>8.3f}{ap5095[i]:>10.3f}")
        print(f"{'-' * 32}\n{'ALL':<14}{box.map50:>8.3f}{box.map:>10.3f}")
        print(f"damage mAP50 (minor/major/destroyed): {dmg_map50:.4f}\n")

    # ---------------- 2. Grid-cell triage metric ---------------- #
    targets = {t: [name_to_id[c] for c in cls if c in name_to_id]
               for t, cls in GRID_TARGETS.items()}
    labels = {t: [] for t in targets}
    scores = {t: [] for t in targets}
    skipped = []

    t0 = time.time()
    for k, img in enumerate(images, 1):
        try:
            res = model.predict(source=str(img), imgsz=imgsz, conf=0.001,
                                iou=args.nms_iou, max_det=args.max_det,
                                device=device, verbose=False)[0]
        except Exception as e:                       # corrupt / truncated image
            skipped.append(f"{img.name}: {e}")
            continue
        b = res.boxes
        pr_cls = b.cls.cpu().numpy().astype(int)
        pr_conf = b.conf.cpu().numpy()
        xywhn = b.xywhn.cpu().numpy()
        pr_xc, pr_yc = (xywhn[:, 0], xywhn[:, 1]) if len(xywhn) else (np.zeros(0), np.zeros(0))

        gt_cls, gt_xc, gt_yc = read_gt(label_path_for(img))
        for t, ids in targets.items():
            l, s = grid_cells(gt_cls, gt_xc, gt_yc, pr_cls, pr_xc, pr_yc, pr_conf,
                              ids, args.grid, args.min_count)
            labels[t].append(l)
            scores[t].append(s)
        if k % 50 == 0:
            print(f"  grid pass {k}/{len(images)}")

    print(f"[grid pass done in {time.time() - t0:.0f}s]")
    if skipped:
        print(f"[warn] skipped {len(skipped)} unreadable image(s):")
        for s in skipped:
            print("   ", s)

    print(f"\nGrid {args.grid}x{args.grid}, positive = >= {args.min_count} target building(s) in cell")
    print(f"{'target':<12}{'cells':>7}{'pos%':>7}{'AUROC':>8}{'AP':>8}"
          f"{'P@' + str(args.conf):>9}{'R@' + str(args.conf):>9}{'F1@' + str(args.conf):>9}"
          f"{'bestF1':>8}{'@conf':>7}")
    for t in targets:
        y = np.concatenate(labels[t]) if labels[t] else np.zeros(0, bool)
        s = np.concatenate(scores[t]) if scores[t] else np.zeros(0)
        a, apr = auroc(y, s), average_precision(y, s)
        p, r, f1 = prf_at(y, s, args.conf)
        bf1, bthr = best_f1(y, s)
        print(f"{t:<12}{len(y):>7}{100 * y.mean() if len(y) else 0:>6.1f}%{a:>8.3f}{apr:>8.3f}"
              f"{p:>9.3f}{r:>9.3f}{f1:>9.3f}{bf1:>8.3f}{bthr:>7.2f}")
        row.update({
            f"{t}_AUROC": round(a, 4), f"{t}_AP": round(apr, 4),
            f"{t}_F1": round(f1, 4), f"{t}_P": round(p, 4), f"{t}_R": round(r, 4),
        })
    row.update({"grid": args.grid, "min_count": args.min_count, "conf": args.conf,
                "n_skipped": len(skipped)})
    print("\n(bestF1 picks its threshold on this same data, so treat it as optimistic.)")

    # ---------------- 3. Append to CSV ---------------- #
    out = Path(args.out)
    existing_rows, fields = [], list(row.keys())
    if out.is_file():
        with open(out, newline="") as f:
            rd = csv.DictReader(f)
            existing_rows = list(rd)
            # keep this script's column order; keep any extra old columns at the end
            fields = list(row.keys()) + [k for k in (rd.fieldnames or []) if k not in row]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in existing_rows + [row]:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nAppended results to {out.resolve()}")


if __name__ == "__main__":
    main()
