"""
extract_labels.py - pull pixel and lat/long coordinates out of the xBD/xView2 labels.

From Python:
    from extract_labels import load_features
    for f in load_features("hurricane-florence_00000032", "post"):
        f.rect      # (left, upper, right, lower) ints, PIL-style: right/lower exclusive
        f.pixels    # (N, 2) array of x, y
        f.lat_lngs  # (N, 2) array of lat, lng; row i is the same vertex as pixels[i]

As CSV, one row per building (from the project root, x-view-images/):
    python extract_labels.py hurricane-florence_00000032
    python extract_labels.py hurricane-florence_00000032 --phase pre
    python extract_labels.py --out features.csv        # every scene
"""

# ***** GENERATED COURTESY OF BESTY ********

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rasterize_masks import parse_polygon

LABEL_DIR = Path("train_images_labels_targets/train/labels")

COLUMNS = ["name", "phase", "uid", "subtype",
           "left", "upper", "right", "lower",
           "lat_min", "lng_min", "lat_max", "lng_max",
           "pixel_wkt", "lng_lat_wkt"]


@dataclass
class Feature:
    uid: str
    subtype: str | None     # None in pre-disaster labels
    rect: tuple[int, int, int, int]
    pixels: np.ndarray
    lat_lngs: np.ndarray
    pixel_wkt: str
    lng_lat_wkt: str


def pixel_rect(pixels, width, height):
    """Smallest PIL-style box holding every mask pixel of the polygon.

    Uses the same half-up rounding as rasterize_masks.py. Edge pixels are filled
    there, hence the +1 on the exclusive right/lower sides.
    """
    left, upper = np.floor(pixels.min(axis=0) + 0.5).astype(int)
    right, lower = np.floor(pixels.max(axis=0) + 0.5).astype(int) + 1
    return (int(max(left, 0)), int(max(upper, 0)),
            int(min(right, width)), int(min(lower, height)))


def load_features(name, phase="post", label_dir=LABEL_DIR):
    """All buildings in one scene, e.g. name='hurricane-florence_00000032'."""
    path = Path(label_dir) / f"{name}_{phase}_disaster.json"
    label = json.loads(path.read_text())
    width, height = label["metadata"]["width"], label["metadata"]["height"]
    geo_by_uid = {g["properties"]["uid"]: g for g in label["features"]["lng_lat"]}

    features = []
    for item in label["features"]["xy"]:
        uid = item["properties"]["uid"]
        geo_wkt = geo_by_uid[uid]["wkt"]
        pixels = parse_polygon(item["wkt"])[0]          # outer ring
        lat_lngs = parse_polygon(geo_wkt)[0][:, ::-1]   # WKT order is lng lat
        if len(pixels) != len(lat_lngs):
            raise ValueError(f"{path.name}: vertex counts differ for {uid}")
        features.append(Feature(uid, item["properties"].get("subtype"),
                                pixel_rect(pixels, width, height),
                                pixels, lat_lngs, item["wkt"], geo_wkt))
    return features


def to_row(name, phase, f):
    return [name, phase, f.uid, f.subtype or "", *f.rect,
            *f.lat_lngs.min(axis=0), *f.lat_lngs.max(axis=0),
            f.pixel_wkt, f.lng_lat_wkt]


def write_csv(stream, names, phase):
    writer = csv.writer(stream)
    writer.writerow(COLUMNS)
    count = 0
    for name in names:
        for feature in load_features(name, phase):
            writer.writerow(to_row(name, phase, feature))
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="Extract pixel and lat/long coordinates from labels")
    ap.add_argument("names", nargs="*", help="scene names; omit for every scene")
    ap.add_argument("--phase", choices=["pre", "post"], default="post")
    ap.add_argument("--out", type=Path, default=None, help="CSV path (default: print)")
    args = ap.parse_args()

    suffix = f"_{args.phase}_disaster"
    names = args.names or sorted(p.stem.removesuffix(suffix)
                                 for p in LABEL_DIR.glob(f"*{suffix}.json"))
    if not names:
        sys.exit(f"[error] no {args.phase} labels in {LABEL_DIR}")

    if args.out is None:
        write_csv(sys.stdout, names, args.phase)
    else:
        with open(args.out, "w", newline="") as f:
            count = write_csv(f, names, args.phase)
        print(f"Wrote {count} buildings from {len(names)} scenes to {args.out}")


if __name__ == "__main__":
    main()


# .venv/bin/python extract_labels.py hurricane-florence_00000032 --phase post --out features.csv