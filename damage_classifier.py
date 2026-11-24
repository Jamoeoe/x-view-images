"""
damage_classifier.py - per-building damage classification with KNOWN building locations.

Instead of asking YOLO to find AND rate every building in one post-storm image, this:
  1. Uses the xBD building outlines (pixel "xy" polygons) to crop every building from
     the post-storm image AND the matching pre-storm image (xBD pairs are aligned),
     plus a mask channel marking which building in the crop is the target.
  2. Trains an ImageNet-pretrained ResNet classifier on those crops:
        variant "post"     -> post RGB + mask        (4 channels)
        variant "prepost"  -> post RGB + pre RGB + mask (7 channels)
     Comparing the two shows how much the before image adds (esp. for destroyed).
  3. Scores on held-out IMAGES (default 20% of the xBD hurricane images, split by image
     so no building's image appears in both train and val):
        per-class precision/recall/F1, macro F1, xView2-style harmonic F1,
        building-level "critical" (major+destroyed), and the same 4x4 grid
        critical-area AUROC/AP used for the YOLO models.
     NOTE: these are "oracle location" results (true outlines given), on a different
     validation set than the YOLO runs -> compare the variants to each other; compare
     to YOLO only loosely until the course val images' outlines are available.

Outputs:
  crops cache            ..\\xbd_crops\\           (outside the repo, reused between runs)
  per-run folder         X_View\\datasets\\data\\clf_runs\\<name>\\  (best.pt, metrics.json)
  results table          classifier_results.csv  (one row per run)
  log                    classifier_log.txt

Usage (project root, venv active):
    python damage_classifier.py --smoke                 # ~2-5 min self-test, run this first
    python damage_classifier.py                         # full: post_r34, prepost_r34, prepost_r50
    python damage_classifier.py --variants prepost_r34  # just one
"""

import argparse
import csv
import json
import os
import random
import re
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["no_damage", "minor_damage", "major_damage", "destroyed"]
SUBTYPE_TO_ID = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
SEVERE = (2, 3)
DEFAULT_DISASTERS = ["hurricane-florence", "hurricane-harvey",
                     "hurricane-matthew", "hurricane-michael"]
# memmap channel layout: 0-2 post RGB, 3-5 pre RGB, 6 mask
VARIANT_CHANNELS = {"post": [0, 1, 2, 6], "prepost": [0, 1, 2, 3, 4, 5, 6]}
LOG = Path("classifier_log.txt")
RESULTS = Path("classifier_results.csv")
RUNS = Path("X_View/datasets/data/clf_runs")


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Crop dataset
# --------------------------------------------------------------------------- #
RING_RE = re.compile(r"\(([^()]+)\)")


def parse_buildings(json_path):
    """[(class_id, [ring arrays Nx2]), ...] for damage-labelled buildings."""
    d = json.loads(Path(json_path).read_text())
    out = []
    for f in d["features"]["xy"]:
        p = f.get("properties", {})
        if p.get("feature_type", "building") != "building":
            continue
        cid = SUBTYPE_TO_ID.get(p.get("subtype"))
        if cid is None:                       # un-classified / missing -> skip
            continue
        rings = []
        for r in RING_RE.findall(f.get("wkt", "")):
            pts = [tuple(map(float, xy.split()[:2])) for xy in r.split(",") if xy.strip()]
            if len(pts) >= 3:
                rings.append(np.array(pts, dtype=np.float32))
        if not rings:
            continue
        allp = np.concatenate(rings)
        w = allp[:, 0].max() - allp[:, 0].min()
        h = allp[:, 1].max() - allp[:, 1].min()
        if w < 2 or h < 2:
            continue
        out.append((cid, rings))
    return out


def crop_padded(arr, x0, y0, side):
    """Square crop [y0:y0+side, x0:x0+side] with zero padding outside the image."""
    H, W = arr.shape[:2]
    out = np.zeros((side, side) + arr.shape[2:], dtype=arr.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + side), min(H, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def make_crops(post, pre, rings, size, margin, min_side):
    allp = np.concatenate(rings)
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(max(min_side, max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    ox, oy = int(round(cx - side / 2)), int(round(cy - side / 2))

    def rs(a, resample):
        return np.asarray(Image.fromarray(a).resize((size, size), resample))

    post_c = rs(crop_padded(post, ox, oy, side), Image.BILINEAR)
    pre_c = rs(crop_padded(pre, ox, oy, side), Image.BILINEAR)
    m = Image.new("L", (side, side), 0)
    dr = ImageDraw.Draw(m)
    for r in rings:
        dr.polygon([(float(x - ox), float(y - oy)) for x, y in r], fill=255)
    mask_c = np.asarray(m.resize((size, size), Image.NEAREST))
    return np.concatenate([post_c, pre_c, mask_c[..., None]], axis=2), cx, cy


def build_dataset(args):
    """Create (or reuse) the crop cache. Returns dict of arrays + memmap path."""
    xbd = Path(args.xbd)
    lbl_dir, img_dir = xbd / "labels", xbd / "images"
    if not lbl_dir.is_dir() or not img_dir.is_dir():
        sys.exit(f"[error] expected {lbl_dir} and {img_dir}")

    disasters = [d.strip() for d in args.disasters.split(",") if d.strip()]
    jsons = sorted(p for p in lbl_dir.glob("*_post_disaster.json")
                   if any(p.name.startswith(d + "_") for d in disasters))
    if args.smoke:
        jsons = jsons[::max(1, len(jsons) // 30)][:30]

    # split by image, per disaster, fixed seed
    rng = random.Random(args.seed)
    val_stems = set()
    for d in disasters:
        stems = [p.stem for p in jsons if p.name.startswith(d + "_")]
        rng.shuffle(stems)
        val_stems |= set(stems[:int(round(len(stems) * args.val_frac))])

    cache = Path(args.cache) / ("smoke" if args.smoke else "full")
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "meta.json"
    x_path = cache / f"crops_{args.size}.npy"
    params = {"disasters": disasters, "size": args.size, "margin": args.margin,
              "min_side": args.min_side, "seed": args.seed, "val_frac": args.val_frac,
              "n_json": len(jsons)}
    if meta_path.is_file() and x_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("params") == params:
            log(f"Reusing crop cache {x_path} ({meta['n']} buildings)")
            arrs = np.load(cache / "index.npz")
            return {**{k: arrs[k] for k in arrs.files}, "x_path": x_path}

    # pass 1: parse + count
    t0 = time.time()
    items, n = [], 0
    for jp in jsons:
        stem = jp.stem
        post_p = img_dir / f"{stem}.png"
        pre_p = img_dir / f"{stem.replace('_post_disaster', '_pre_disaster')}.png"
        if not post_p.is_file() or not pre_p.is_file():
            continue
        b = parse_buildings(jp)
        if b:
            items.append((stem, post_p, pre_p, b))
            n += len(b)
    gb = n * args.size * args.size * 7 / 1e9
    log(f"{len(items)} image pairs, {n} buildings -> crop cache {gb:.1f} GB at {x_path}")

    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8,
                                  shape=(n, args.size, args.size, 7))
    y = np.zeros(n, np.int64)
    img_id = np.zeros(n, np.int64)
    cxy = np.zeros((n, 2), np.float32)
    is_val = np.zeros(n, bool)
    i = 0
    skipped_imgs = 0
    for k, (stem, post_p, pre_p, blds) in enumerate(items):
        try:
            post = np.asarray(Image.open(post_p).convert("RGB"))
            pre = np.asarray(Image.open(pre_p).convert("RGB"))
        except Exception as e:                    # corrupt / truncated image
            skipped_imgs += 1
            log(f"  [skip] {stem}: {e}")
            continue
        H, W = post.shape[:2]
        for cid, rings in blds:
            X[i], cx, cy = make_crops(post, pre, rings, args.size, args.margin, args.min_side)
            y[i], img_id[i], cxy[i], is_val[i] = cid, k, (cx / W, cy / H), stem in val_stems
            i += 1
        if (k + 1) % 100 == 0:
            log(f"  cropped {k + 1}/{len(items)} images ({time.time() - t0:.0f}s)")
    X.flush()
    del X
    n = i  # if images were skipped, only the first i rows are valid
    np.savez(cache / "index.npz", y=y[:n], img_id=img_id[:n], cxy=cxy[:n], is_val=is_val[:n])
    meta_path.write_text(json.dumps({"params": params, "n": n}))
    log(f"Crops done: {n} buildings in {time.time() - t0:.0f}s ({skipped_imgs} images skipped)")
    arrs = np.load(cache / "index.npz")
    return {**{k: arrs[k] for k in arrs.files}, "x_path": x_path}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def class_metrics(y, pred):
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, pred), 1)
    out = {"confusion": cm.tolist(), "accuracy": float((y == pred).mean())}
    f1s = []
    for c, name in enumerate(CLASSES):
        tp = cm[c, c]
        p = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        r = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[f"P_{name}"], out[f"R_{name}"], out[f"F1_{name}"] = float(p), float(r), float(f1)
        f1s.append(f1)
    out["macro_F1"] = float(np.mean(f1s))
    out["harmonic_F1"] = float(len(f1s) / sum(1 / max(f, 1e-6) for f in f1s))
    return out


def severe_metrics(y, probs, img_id, cxy, grid):
    from evaluate import auroc, average_precision, prf_at   # reuse the YOLO scoring code
    sev_y = np.isin(y, SEVERE)
    sev_s = probs[:, list(SEVERE)].sum(1)
    out = {"bld_critical_AUROC": auroc(sev_y, sev_s),
           "bld_critical_AP": average_precision(sev_y, sev_s)}
    p, r, f1 = prf_at(sev_y, sev_s, 0.5)
    out.update({"bld_critical_P": p, "bld_critical_R": r, "bld_critical_F1": f1})
    # 4x4 grid: cell positive if any severe building centred in it; score = max severe prob
    cell = (np.clip((cxy[:, 1] * grid).astype(int), 0, grid - 1) * grid
            + np.clip((cxy[:, 0] * grid).astype(int), 0, grid - 1))
    key = img_id * grid * grid + cell
    uniq, inv = np.unique(key, return_inverse=True)
    lab = np.zeros(len(uniq), bool)
    sc = np.zeros(len(uniq))
    np.logical_or.at(lab, inv, sev_y)
    np.maximum.at(sc, inv, sev_s)
    out["grid_critical_AUROC"] = auroc(lab, sc)
    out["grid_critical_AP"] = average_precision(lab, sc)
    p, r, f1 = prf_at(lab, sc, 0.5)
    out.update({"grid_critical_P": p, "grid_critical_R": r, "grid_critical_F1": f1,
                "grid_cells_with_buildings": int(len(uniq))})
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_model(arch, in_ch, n_cls, log_fn):
    import torch
    import torchvision
    ssl._create_default_https_context = ssl._create_unverified_context
    ctor = getattr(torchvision.models, arch)
    try:
        model = ctor(weights="DEFAULT")
        pretrained = True
    except Exception as e:
        log_fn(f"  [warn] could not download ImageNet weights for {arch} ({e}); training from scratch")
        model = ctor(weights=None)
        pretrained = False
    old = model.conv1
    new = torch.nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride,
                          old.padding, bias=False)
    with torch.no_grad():
        w = old.weight                                   # (64, 3, k, k)
        n_img = (in_ch - 1) // 3                         # number of RGB images
        for j in range(n_img):
            new.weight[:, 3 * j:3 * j + 3] = w / n_img
        new.weight[:, in_ch - 1:] = w.mean(1, keepdim=True)   # mask channel
    model.conv1 = new
    model.fc = torch.nn.Linear(model.fc.in_features, n_cls)
    return model, pretrained


def run_variant(name, data, args):
    import torch
    import torch.nn.functional as F

    kind, arch_tag = name.split("_")
    arch = {"r18": "resnet18", "r34": "resnet34", "r50": "resnet50"}[arch_tag]
    chans = VARIANT_CHANNELS[kind]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X = np.load(data["x_path"], mmap_mode="r")
    y_all = data["y"]
    tr = np.where(~data["is_val"])[0]
    va = np.where(data["is_val"])[0]
    log(f"=== {name}: {arch}, channels {chans}, {len(tr)} train / {len(va)} val buildings, device {device}")
    counts = np.bincount(y_all[tr], minlength=4)
    log(f"  train class counts {dict(zip(CLASSES, counts.tolist()))}")

    model, pretrained = build_model(arch, len(chans), 4, log)
    model = model.to(device)
    w = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32, device=device)
    epochs = 1 if args.smoke else args.epochs
    steps_per_epoch = max(1, len(tr) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=epochs * steps_per_epoch, pct_start=0.15)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    def to_tensor(idx):
        idx = np.sort(idx)
        a = np.asarray(X[idx][..., chans])                # (B, S, S, C) uint8
        t = torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float() / 255.0
        n_img = (len(chans) - 1) // 3
        for j in range(n_img):
            t[:, 3 * j:3 * j + 3] = (t[:, 3 * j:3 * j + 3] - mean[:, None, None]) / std[:, None, None]
        t[:, -1] = (t[:, -1] - 0.5) / 0.5
        return t, torch.from_numpy(y_all[idx]).to(device)

    def augment(t):
        B = t.shape[0]
        def pick():
            return (torch.rand(B, device=device) < 0.5)[:, None, None, None]
        t = torch.where(pick(), t.flip(3), t)
        t = torch.where(pick(), t.flip(2), t)
        t = torch.where(pick(), t.transpose(2, 3), t)      # with flips -> all 8 rotations/flips
        n_img = (t.shape[1] - 1) // 3
        gain = 1 + 0.1 * torch.randn(B, 1, 1, 1, device=device)
        t[:, :3 * n_img] = t[:, :3 * n_img] * gain
        return t

    @torch.no_grad()
    def predict(idx_set):
        model.eval()
        probs = []
        for s in range(0, len(idx_set), args.batch * 2):
            chunk = idx_set[s:s + args.batch * 2]          # already sorted
            t, _ = to_tensor(chunk)
            probs.append(F.softmax(model(t).float(), 1).cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros((0, 4))

    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    va_sorted = np.sort(va)
    best, best_metrics, t_start = -1.0, None, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(tr)
        tot, nb = 0.0, 0
        for s in range(0, steps_per_epoch * args.batch, args.batch):
            t, yt = to_tensor(perm[s:s + args.batch])
            t = augment(t)
            loss = F.cross_entropy(model(t), yt, weight=w, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            if args.smoke and nb >= 20:
                break
        probs = predict(va_sorted)
        yv = y_all[va_sorted]
        m = class_metrics(yv, probs.argmax(1))
        log(f"  ep {ep:>2}/{epochs} loss {tot / max(nb, 1):.3f} | acc {m['accuracy']:.3f} "
            f"macroF1 {m['macro_F1']:.3f} | F1 no/minor/major/destroyed "
            f"{m['F1_no_damage']:.2f}/{m['F1_minor_damage']:.2f}/"
            f"{m['F1_major_damage']:.2f}/{m['F1_destroyed']:.2f}")
        if m["macro_F1"] > best:
            best = m["macro_F1"]
            sev = severe_metrics(yv, probs, data["img_id"][va_sorted], data["cxy"][va_sorted], args.grid)
            best_metrics = {**m, **sev, "best_epoch": ep}
            torch.save({k: v.half() if v.is_floating_point() else v
                        for k, v in model.state_dict().items()}, out_dir / "best.pt")
    minutes = (time.time() - t_start) / 60
    best_metrics.update({"train_minutes": round(minutes, 1), "pretrained": pretrained})
    (out_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2))

    cm = np.array(best_metrics["confusion"])
    log(f"  BEST {name} (epoch {best_metrics['best_epoch']}): macroF1 {best_metrics['macro_F1']:.3f}, "
        f"harmonicF1 {best_metrics['harmonic_F1']:.3f}, destroyed F1 {best_metrics['F1_destroyed']:.3f}, "
        f"grid critical AUROC {best_metrics['grid_critical_AUROC']:.3f} "
        f"P/R {best_metrics['grid_critical_P']:.2f}/{best_metrics['grid_critical_R']:.2f}  ({minutes:.0f} min)")
    log("  confusion (rows=true no/minor/major/destroyed, cols=pred):\n" +
        "\n".join("      " + " ".join(f"{v:>6}" for v in row) for row in cm))

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "name": name,
           "arch": arch, "channels": "+".join({0: "postRGB", 3: "preRGB", 6: "mask"}[c] for c in chans if c in (0, 3, 6)),
           "n_train": len(tr), "n_val": len(va), "epochs": epochs, "smoke": args.smoke,
           **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in best_metrics.items()
              if k != "confusion"}}
    rows, fields = [], list(row)
    if RESULTS.is_file():
        with open(RESULTS, newline="") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            fields += [k for k in (rd.fieldnames or []) if k not in fields]
    with open(RESULTS, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows + [row]:
            wr.writerow({k: r.get(k, "") for k in fields})
    del X
    return best_metrics


def wait_for_autopilot(max_hours):
    p = Path("autopilot_log.txt")
    if not p.is_file():
        return
    t0 = time.time()
    while True:
        tail = p.read_text(encoding="utf-8", errors="ignore").splitlines()[-3:]
        if any("autopilot finished" in l or "Not enough time" in l for l in tail):
            log("Autopilot is finished - GPU is free.")
            return
        if time.time() - t0 > max_hours * 3600:
            log("Waited long enough for autopilot - starting anyway.")
            return
        if int(time.time() - t0) % 600 < 60:
            log("Waiting for autopilot to finish (checking every minute)...")
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xbd", default="../train", help="extracted xBD folder with images/ and labels/")
    ap.add_argument("--cache", default="../xbd_crops", help="where to store the crop cache")
    ap.add_argument("--disasters", default=",".join(DEFAULT_DISASTERS))
    ap.add_argument("--variants", default="post_r34,prepost_r34,prepost_r50",
                    help="comma list of {post,prepost}_{r18,r34,r50}")
    ap.add_argument("--size", type=int, default=96, help="crop size in pixels")
    ap.add_argument("--margin", type=float, default=0.3, help="context added on each side (fraction)")
    ap.add_argument("--min-side", type=int, default=40, help="minimum crop side in source pixels")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny quick run to check everything works")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for autopilot to finish")
    args = ap.parse_args()

    log(f"##### damage_classifier start {'(SMOKE TEST)' if args.smoke else ''} variants={args.variants}")
    data = build_dataset(args)
    if not args.smoke and not args.no_wait:
        wait_for_autopilot(max_hours=3)
    variants = ["post_r18"] if args.smoke else [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        try:
            run_variant(v, data, args)
        except Exception as e:
            import traceback
            log(f"!!! {v} failed: {e}\n{traceback.format_exc()}")
    log("##### damage_classifier finished")


if __name__ == "__main__":
    main()
