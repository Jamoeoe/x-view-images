"""
convert_xbd.py - add extra xBD training images to the YOLO dataset.

The xBD "Challenge training set" (train_images_labels_targets.tar.gz) stores each
building as a polygon in a JSON file. YOLO needs one .txt per image with
"class x_center y_center width height" (normalized 0-1). This script:

  1. Picks the post-disaster images for the chosen disasters (default: the four
     hurricanes already in the project).
  2. Converts every building polygon to a YOLO box, using the SAME class numbers
     as the existing data.yaml (0 unclassified, 1 no_damage, 2 minor, 3 major,
     4 destroyed).
  3. Skips any image whose name already exists in the project's train/ or val/
     folders, so validation stays clean and scores stay comparable.
  4. Writes images + labels to X_View/datasets/data/train/xbd/{images,labels}
     (inside train/, which .gitignore already excludes).
  5. Writes X_View/datasets/data/data_xbd.yaml = original train + xBD train,
     same val set.

Usage (from the project root, x-view-images/):
    python convert_xbd.py                       # xBD extracted at ..\\train
    python convert_xbd.py --xbd "D:\\data\\xBD\\train"
Then train with:
    python train.py --data X_View/datasets/data/data_xbd.yaml --model yolov8s.pt --epochs 100 --name v8s_xbd
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

DEFAULT_DISASTERS = ["hurricane-florence", "hurricane-harvey",
                     "hurricane-matthew", "hurricane-michael"]

# xBD subtype -> class id in the project's data.yaml
SUBTYPE_TO_CLASS = {
    "un-classified": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def polygon_bbox(wkt: str):
    """Min/max x,y over every coordinate in a WKT POLYGON/MULTIPOLYGON."""
    nums = [float(n) for n in NUM_RE.findall(wkt)]
    if len(nums) < 4:
        return None
    xs, ys = nums[0::2], nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def convert_label(json_path: Path, min_px: float):
    """Return (list of YOLO lines, Counter of classes, n_skipped_buildings)."""
    d = json.loads(json_path.read_text())
    w = float(d["metadata"].get("width", 1024))
    h = float(d["metadata"].get("height", 1024))
    lines, counts, skipped = [], Counter(), 0
    for feat in d["features"]["xy"]:
        props = feat.get("properties", {})
        if props.get("feature_type", "building") != "building":
            skipped += 1
            continue
        cls = SUBTYPE_TO_CLASS.get(props.get("subtype"))
        bb = polygon_bbox(feat.get("wkt", ""))
        if cls is None or bb is None:
            skipped += 1
            continue
        x0, y0, x1, y1 = bb
        x0, x1 = max(0.0, x0), min(w, x1)          # clip to image
        y0, y1 = max(0.0, y0), min(h, y1)
        if x1 - x0 < min_px or y1 - y0 < min_px:   # degenerate / sliver
            skipped += 1
            continue
        xc, yc = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        counts[cls] += 1
    return lines, counts, skipped


def place_file(src: Path, dst: Path):
    """Hard-link if possible (no extra disk space), otherwise copy."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_ok(p: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.load()          # full decode catches truncated files
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xbd", default="../train",
                    help="extracted xBD train folder (contains images/ and labels/)")
    ap.add_argument("--data", default="X_View/datasets/data/data.yaml",
                    help="project's existing data.yaml")
    ap.add_argument("--out", default="X_View/datasets/data/train/xbd",
                    help="where to write images/ and labels/")
    ap.add_argument("--yaml-out", default="X_View/datasets/data/data_xbd.yaml")
    ap.add_argument("--disasters", default=",".join(DEFAULT_DISASTERS),
                    help="comma-separated disaster prefixes to include")
    ap.add_argument("--min-px", type=float, default=2.0,
                    help="drop boxes thinner than this many pixels")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip decoding each image to catch corrupt files (faster)")
    args = ap.parse_args()

    xbd = Path(args.xbd)
    xbd_img, xbd_lbl = xbd / "images", xbd / "labels"
    if not xbd_img.is_dir() or not xbd_lbl.is_dir():
        sys.exit(f"[error] expected {xbd_img} and {xbd_lbl} to exist")

    data_yaml = Path(args.data)
    data_dir = data_yaml.parent
    cfg = yaml.safe_load(data_yaml.read_text())

    # Existing project images (by name) -> never reuse them
    existing = set()
    for split in ("train", "val"):
        d = data_dir / split / "images"
        if d.is_dir():
            existing |= {p.stem for p in d.iterdir()}
    print(f"Existing project images (train+val): {len(existing)}")

    disasters = [s.strip() for s in args.disasters.split(",") if s.strip()]
    jsons = sorted(p for p in xbd_lbl.glob("*_post_disaster.json")
                   if any(p.name.startswith(dz + "_") for dz in disasters))
    print(f"xBD post-disaster label files for {disasters}: {len(jsons)}")

    out_img, out_lbl = Path(args.out) / "images", Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    per_disaster = Counter()
    class_counts = Counter()
    n_overlap = n_missing = n_corrupt = n_bld_skipped = n_written = 0

    for i, jp in enumerate(jsons, 1):
        stem = jp.stem
        if stem in existing:
            n_overlap += 1
            continue
        src_img = xbd_img / f"{stem}.png"
        if not src_img.is_file():
            n_missing += 1
            continue
        if not args.no_verify and not image_ok(src_img):
            n_corrupt += 1
            print(f"  [skip] corrupt image: {src_img.name}")
            continue

        lines, counts, skipped = convert_label(jp, args.min_px)
        (out_lbl / f"{stem}.txt").write_text("".join(lines))
        place_file(src_img, out_img / src_img.name)

        class_counts.update(counts)
        n_bld_skipped += skipped
        per_disaster[stem.split("_")[0]] += 1
        n_written += 1
        if i % 200 == 0:
            print(f"  {i}/{len(jsons)}")

    names = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][k] for k in sorted(cfg["names"])]
    print(f"\nWrote {n_written} images + labels to {Path(args.out).resolve()}")
    for dz, n in sorted(per_disaster.items()):
        print(f"  {dz:<22}{n}")
    print(f"Skipped: {n_overlap} already in project train/val, {n_missing} missing image, "
          f"{n_corrupt} corrupt, {n_bld_skipped} buildings (unknown class / degenerate)")
    print("Boxes per class (new data):")
    for c in range(len(names)):
        print(f"  {c} {names[c]:<14}{class_counts[c]}")

    # data_xbd.yaml: same as data.yaml, but train = original + xbd
    rel_root = Path(cfg.get("path", "."))
    out_rel = os.path.relpath(Path(args.out).resolve(), (Path.cwd() / rel_root).resolve())
    new_cfg = {
        "path": cfg.get("path", "."),
        "train": [cfg["train"], "./" + Path(out_rel, "images").as_posix()],
        "val": cfg["val"],
        "nc": cfg.get("nc", len(names)),
        "names": names,
    }
    Path(args.yaml_out).write_text(yaml.safe_dump(new_cfg, sort_keys=False))
    print(f"\nWrote {args.yaml_out}:\n" + yaml.safe_dump(new_cfg, sort_keys=False))


if __name__ == "__main__":
    main()
