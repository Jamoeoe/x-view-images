"""
convert_xbd_footprints.py - build a YOLO *segmentation* dataset for finding building
outlines in PRE-disaster satellite images (one class: "building").

Uses every xBD pre-disaster image in the extracted training set (all disaster types -
more variety helps it generalise to new places like Long Island). Each building polygon
("xy" pixel outline in the *_pre_disaster.json labels) becomes one YOLO-seg line:
    0 x1 y1 x2 y2 ... xn yn        (normalized 0-1)

Split is by image (default 10% held out for validation, fixed seed).
Images are hard-linked (no extra disk space), output lives OUTSIDE the repo.

Usage (project root, venv active):
    python convert_xbd_footprints.py
    python convert_xbd_footprints.py --xbd "..\\train" --out "..\\xbd_footprints"
Then train:
    python train.py --data ..\\xbd_footprints\\footprints.yaml --model yolo11s-seg.pt --epochs 100 --patience 25 --name footprints_y11s_seg
"""

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

RING_RE = re.compile(r"\(([^()]+)\)")


def outer_ring(wkt):
    rings = RING_RE.findall(wkt or "")
    if not rings:
        return None
    pts = [tuple(map(float, p.split()[:2])) for p in rings[0].split(",") if p.strip()]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]                       # YOLO doesn't need the closing point
    return pts if len(pts) >= 3 else None


def place(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xbd", default="../train", help="extracted xBD train folder")
    ap.add_argument("--out", default="../xbd_footprints")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-px", type=float, default=3.0, help="drop outlines smaller than this")
    args = ap.parse_args()

    xbd, out = Path(args.xbd), Path(args.out)
    lbl_dir, img_dir = xbd / "labels", xbd / "images"
    jsons = sorted(lbl_dir.glob("*_pre_disaster.json"))
    print(f"{len(jsons)} pre-disaster label files in {lbl_dir}")

    # split by image, stratified by disaster
    by_dis = {}
    for j in jsons:
        by_dis.setdefault(j.name.rsplit("_", 3)[0], []).append(j)
    rng = random.Random(args.seed)
    val = set()
    for dis, files in by_dis.items():
        files = files[:]
        rng.shuffle(files)
        val |= {f.stem for f in files[:max(1, round(len(files) * args.val_frac))]}

    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = Counter()
    per_dis = Counter()
    for i, jp in enumerate(jsons, 1):
        img = img_dir / f"{jp.stem}.png"
        if not img.is_file():
            stats["missing_image"] += 1
            continue
        d = json.loads(jp.read_text())
        W = float(d["metadata"].get("width", 1024))
        H = float(d["metadata"].get("height", 1024))
        lines = []
        for f in d["features"]["xy"]:
            if f.get("properties", {}).get("feature_type", "building") != "building":
                continue
            pts = outer_ring(f.get("wkt"))
            if not pts:
                stats["bad_polygon"] += 1
                continue
            xs = [min(max(x, 0), W) for x, _ in pts]
            ys = [min(max(y, 0), H) for _, y in pts]
            if max(xs) - min(xs) < args.min_px or max(ys) - min(ys) < args.min_px:
                stats["tiny"] += 1
                continue
            lines.append("0 " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in zip(xs, ys)) + "\n")
        split = "val" if jp.stem in val else "train"
        (out / split / "labels" / f"{jp.stem}.txt").write_text("".join(lines))
        place(img, out / split / "images" / img.name)
        stats[f"{split}_images"] += 1
        stats[f"{split}_buildings"] += len(lines)
        stats["empty_images"] += (len(lines) == 0)
        per_dis[jp.name.rsplit("_", 3)[0]] += 1
        if i % 500 == 0:
            print(f"  {i}/{len(jsons)}")

    cfg = {"path": str(out.resolve()), "train": "train/images", "val": "val/images",
           "nc": 1, "names": ["building"]}
    (out / "footprints.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("\nImages per disaster:")
    for k, v in sorted(per_dis.items()):
        print(f"  {k:<28}{v}")
    print("\n" + "\n".join(f"{k}: {v}" for k, v in sorted(stats.items())))
    print(f"\nWrote {out.resolve() / 'footprints.yaml'}")


if __name__ == "__main__":
    main()
