"""
rasterize_masks.py - turn the xBD/xView2 polygon labels into per-pixel masks.

Reads <data>/labels/*.json and writes one single-channel uint8 PNG per label to
<data>/masks/, named like the image it belongs to (so images/x.png -> masks/x.png):

    0 background   1 no damage   2 minor damage   3 major damage   4 destroyed

Pre-disaster labels carry no damage subtype, so every building in them is 1.
Un-classified buildings are 0 by default, as in the shipped targets/. Pass
--unclassified 255 to mark them instead, for use as a loss ignore_index.

Usage (from the project root, x-view-images/):
    python rasterize_masks.py
    python rasterize_masks.py --unclassified 255
    python rasterize_masks.py --verify      # also compare with targets/
"""

# ***** GENERATED COURTESY OF BESTY ********
# ***** NOT CURRENTLY USED, would be needed for U-NET stuff which does per pixel classification with resnet

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

DAMAGE_CLASSES = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
UNCLASSIFIED = "un-classified"

RING_RE = re.compile(r"\(([^()]+)\)")


def parse_polygon(wkt):
    """'POLYGON ((x y, x y, ...))' -> list of (N, 2) float arrays, one per ring."""
    if not wkt.startswith("POLYGON"):
        raise ValueError(f"unsupported geometry: {wkt[:40]!r}")
    rings = [
        np.array([pt.split()[:2] for pt in ring.split(",")], dtype=np.float64)
        for ring in RING_RE.findall(wkt)
    ]
    if not rings:
        raise ValueError(f"no coordinates in: {wkt[:40]!r}")
    return rings


def class_value(subtype, unclassified=0):
    if subtype is None:  # pre-disaster: building footprint only
        return 1
    if subtype == UNCLASSIFIED:
        return unclassified
    return DAMAGE_CLASSES[subtype]


def rasterize(label, unclassified=0):
    """Parsed label JSON -> (height, width) uint8 mask.

    Later buildings overwrite earlier ones where they overlap. Vertices are
    rounded (half up) to whole pixels and edge pixels are filled, matching targets/.
    """
    meta = label["metadata"]
    mask = np.zeros((meta["height"], meta["width"]), dtype=np.uint8)
    for feature in label["features"]["xy"]:
        value = class_value(feature["properties"].get("subtype"), unclassified)
        if value == 0:
            continue
        # One fillPoly call per building: extra rings become holes (even-odd rule)
        rings = [np.floor(r + 0.5).astype(np.int32) for r in parse_polygon(feature["wkt"])]
        cv2.fillPoly(mask, rings, value)
    return mask


def process(label_path, out_dir, unclassified, target_dir=None):
    """Write one mask. Returns (differing pixels, pixels) vs its target, or None."""
    mask = rasterize(json.loads(label_path.read_text()), unclassified)
    out_path = out_dir / f"{label_path.stem}.png"
    if not cv2.imwrite(str(out_path), mask):
        raise OSError(f"could not write {out_path}")
    if target_dir is None:
        return None

    target_path = target_dir / f"{label_path.stem}_target.png"
    target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
    if target is None:
        raise FileNotFoundError(target_path)
    if target.shape != mask.shape:
        raise ValueError(f"{target_path}: shape {target.shape} != mask {mask.shape}")
    written = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
    return int((written != target).sum()), mask.size


def parse_args():
    ap = argparse.ArgumentParser(description="Rasterize xView2 polygon labels to masks")
    ap.add_argument("--data", type=Path, default=Path("train_images_labels_targets/train"),
                    help="folder containing labels/ (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/masks")
    ap.add_argument("--unclassified", type=int, default=0, choices=range(256),
                    metavar="0-255", help="value for un-classified buildings (default: 0)")
    ap.add_argument("--workers", type=int, default=None, help="default: one per CPU")
    ap.add_argument("--verify", action="store_true",
                    help="compare with <data>/targets (needs --unclassified 0)")
    args = ap.parse_args()
    if args.verify and args.unclassified != 0:
        ap.error("--verify needs --unclassified 0, which is what targets/ uses")
    return args


def main():
    args = parse_args()
    labels = sorted((args.data / "labels").glob("*.json"))
    if not labels:
        sys.exit(f"[error] no label files in {args.data / 'labels'}")

    out_dir = args.out or args.data / "masks"
    target_dir = args.data / "targets"
    if out_dir.resolve() == target_dir.resolve():
        sys.exit("[error] refusing to write into targets/")
    out_dir.mkdir(parents=True, exist_ok=True)

    work = partial(process, out_dir=out_dir, unclassified=args.unclassified,
                   target_dir=target_dir if args.verify else None)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(work, labels, chunksize=32))
    print(f"Wrote {len(results)} masks to {out_dir}")

    if args.verify:
        bad = [(d, p.name) for (d, _), p in zip(results, labels) if d]
        n_diff = sum(d for d, _ in results)
        n_total = sum(n for _, n in results)
        print(f"Identical to targets/: {len(results) - len(bad)}/{len(results)}")
        print(f"Differing pixels: {n_diff}/{n_total}")
        for d, name in sorted(bad, reverse=True)[:10]:
            print(f"  {d:7d}  {name}")


if __name__ == "__main__":
    main()
