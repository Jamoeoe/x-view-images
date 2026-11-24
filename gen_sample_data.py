import csv
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from extract_labels import LABEL_DIR, load_features
from rasterize_masks import DAMAGE_CLASSES, rasterize

SEED = "bazinga"

IN_PATH = "train_images_labels_targets/train"
OUT_PATH = "data"

EXTENSIONS = ["_pre_disaster.png", "_post_disaster.png"]

IMAGE_SIZE = 1024
SAMPLE_SIZE = 512

# 5 equal groups: random crops, then crops around a building of each damage class
SAMPLES_PER_GROUP = 2000
GROUPS = ["random", "destroyed", "major-damage", "minor-damage", "no-damage"]

VAL_FRACTION = 0.2          # of scenes, within each disaster
MAX_USES = 4                # most training crops any one building may be fully inside
MAX_CROP_IOU = 0.5          # training crops of one scene may not overlap more than this
MAX_FAILS = 10              # give up on a building after this many rejected crops
MIN_LABEL_PIXELS = 32       # a class needs this many pixels in the crop to set its label
SAME_PLACE_OVERLAP = 0.1    # scenes sharing more ground than this stay in the same split

UNCLASSIFIED = 255


# random crop position that contains feature_rect, or anywhere if there is none
def pick_crop(feature_rect=None, size=SAMPLE_SIZE):
    if feature_rect:
        if feature_rect[2] - feature_rect[0] > size or feature_rect[3] - feature_rect[1] > size:
            return None

        left = random.randint(
            # max f_right - size, 0 -> smallest left coord allowed
            max(feature_rect[2] - size, 0),

            # min f_left, 1024 - size -> largest left coord allowed
            min(feature_rect[0], IMAGE_SIZE - size)
        )

        upper = random.randint(
            # max f_lower - size, 0 -> smallest upper coord allowed
            max(feature_rect[3] - size, 0),

            # min f_upper, 1024 - size -> largest upper coord allowed
            min(feature_rect[1], IMAGE_SIZE - size)
        )

    else:
        left = random.randint(0, IMAGE_SIZE - size)
        upper = random.randint(0, IMAGE_SIZE - size)

    return (left, upper, left + size, upper + size)


# cut the same crops out of a scene's pre and post images
def save_scene(name, crops, out_dir):
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    for extension in EXTENSIONS:
        with Image.open(f"{IN_PATH}/images/{name}{extension}") as image:
            for crop in crops:
                image.crop(crop).save(f"{out_dir}/images/{sample_id(name, crop)}{extension}")


def sample_id(name, crop):
    return f"{name}_{crop[0]}_{crop[1]}"


# makes smallest rect that contains both rects
def union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


class Scene:
    def __init__(self, name):
        self.name = name
        post = load_features(name, "post")
        pre = {f.uid: f for f in load_features(name, "pre")}

        self.subtypes = [f.subtype for f in post]
        self.rects = np.array([union(f.rect, pre[f.uid].rect) for f in post]).reshape(-1, 4)
        self.uses = np.zeros(len(post), dtype=int)
        self.crops = []
        self.bounds = ground_bounds(post)

        # Un-classified buildings are 255 here, so they are not mistaken for empty ground
        with open(LABEL_DIR / f"{name}_post_disaster.json") as f:
            self.mask = rasterize(json.load(f), UNCLASSIFIED)

    # worst damage in the crop: 0 empty, 1 none, 2 minor, 3 major, 4 destroyed
    # None if an un-classified building leaves it unknown
    def label(self, crop):
        left, upper, right, lower = crop
        counts = np.bincount(self.mask[upper:lower, left:right].ravel(), minlength=256)
        worst = max((c for c in (1, 2, 3, 4) if counts[c] >= MIN_LABEL_PIXELS), default=0)
        if counts[UNCLASSIFIED] and worst < 4:
            return None
        return worst

    def inside(self, crop):
        return ((self.rects[:, :2] >= crop[:2]) & (self.rects[:, 2:] <= crop[2:])).all(axis=1)

    # keep the crop if it respects the repeat limits
    def try_add(self, crop):
        inside = self.inside(crop)
        if (self.uses[inside] >= MAX_USES).any():
            return False
        if any(iou(crop, other) > MAX_CROP_IOU for other in self.crops):
            return False
        self.uses[inside] += 1
        self.crops.append(crop)
        return True


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return w * h / (area - w * h)


# (lat_min, lng_min, lat_max, lng_max) of the whole image, None if there are no buildings
def ground_bounds(features):
    if not features:
        return None
    pixels = np.vstack([f.pixels for f in features])
    ground = np.vstack([f.lat_lngs for f in features])
    mean, std = pixels.mean(axis=0), pixels.std(axis=0)

    def design(p):
        return np.c_[(p - mean) / std, np.ones(len(p))]

    coef, _, rank, _ = np.linalg.lstsq(design(pixels), ground, rcond=None)
    if rank < 3:
        return None
    s = IMAGE_SIZE
    corners = design(np.array([[0, 0], [s, 0], [0, s], [s, s]])) @ coef
    return np.concatenate([corners.min(axis=0), corners.max(axis=0)])


# some scenes show the same ground, so group them to keep train and val separate
def group_overlapping(scenes):
    parent = {s.name: s.name for s in scenes}

    def find(n):
        while parent[n] != n:
            n = parent[n]
        return n

    placed = [s for s in scenes if s.bounds is not None]
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            w = np.minimum(a.bounds[2:], b.bounds[2:]) - np.maximum(a.bounds[:2], b.bounds[:2])
            if (w <= 0).any():
                continue
            smaller = min(np.prod(a.bounds[2:] - a.bounds[:2]), np.prod(b.bounds[2:] - b.bounds[:2]))
            if np.prod(w) / smaller > SAME_PLACE_OVERLAP:
                parent[find(a.name)] = find(b.name)

    groups = defaultdict(list)
    for s in scenes:
        groups[find(s.name)].append(s)
    return list(groups.values())


# split whole scenes, so each disaster is in both train and val
def split_scenes(scenes):
    by_disaster = defaultdict(list)
    for s in scenes:
        by_disaster[s.name.rsplit("_", 1)[0]].append(s)

    train, val = [], []
    for disaster_scenes in by_disaster.values():
        groups = group_overlapping(disaster_scenes)
        random.shuffle(groups)
        n_val = 0
        for group in groups:
            if n_val < VAL_FRACTION * len(disaster_scenes):
                val += group
                n_val += len(group)
            else:
                train += group
    return train, val


# natural mix: scene first, then anywhere in it
def sample_random(scenes):
    rows = []
    for _ in range(SAMPLES_PER_GROUP * 50):
        if len(rows) == SAMPLES_PER_GROUP:
            break
        scene = random.choice(scenes)
        crop = pick_crop()
        label = scene.label(crop)
        if label is not None and scene.try_add(crop):
            rows.append((scene, crop, label, "random"))
    return rows


# scene first, then a building of this class in it, so a few dense scenes don't dominate
def sample_class(scenes, subtype):
    wanted = DAMAGE_CLASSES[subtype]
    left_to_try = {}
    for scene in scenes:
        buildings = [i for i, s in enumerate(scene.subtypes) if s == subtype]
        if buildings:
            left_to_try[scene] = buildings

    rows = []
    fails = Counter()
    while len(rows) < SAMPLES_PER_GROUP and left_to_try:
        scene = random.choice(list(left_to_try))
        building = random.choice(left_to_try[scene])
        crop = pick_crop(tuple(int(v) for v in scene.rects[building]))

        # the worst damage in the crop has to be the class we are sampling for
        if crop and scene.label(crop) == wanted and scene.try_add(crop):
            rows.append((scene, crop, wanted, subtype))
        else:
            fails[scene, building] += 1

        if (crop is None or scene.uses[building] >= MAX_USES
                or fails[scene, building] >= MAX_FAILS):
            left_to_try[scene].remove(building)
            if not left_to_try[scene]:
                del left_to_try[scene]
    return rows


# every tile of every scene, so validation keeps the natural class balance
def tile_scenes(scenes):
    rows = []
    for scene in scenes:
        for upper in range(0, IMAGE_SIZE, SAMPLE_SIZE):
            for left in range(0, IMAGE_SIZE, SAMPLE_SIZE):
                crop = (left, upper, left + SAMPLE_SIZE, upper + SAMPLE_SIZE)
                label = scene.label(crop)
                if label is not None:
                    rows.append((scene, crop, label, "tile"))
    return rows


def write_split(rows, out_dir):
    crops = defaultdict(list)
    for scene, crop, _, _ in rows:
        crops[scene.name].append(crop)
    with ProcessPoolExecutor() as pool:
        list(pool.map(save_scene, crops.keys(), crops.values(), [out_dir] * len(crops), chunksize=8))

    with open(f"{out_dir}/labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pre", "post", "label", "group", "scene", "left", "upper"])
        for scene, crop, label, group in rows:
            sample = sample_id(scene.name, crop)
            writer.writerow([f"images/{sample}_pre_disaster.png", f"images/{sample}_post_disaster.png",
                             label, group, scene.name, crop[0], crop[1]])


def main(out_path=OUT_PATH):
    random.seed(SEED)

    suffix = "_post_disaster"
    names = sorted(p.stem.removesuffix(suffix) for p in LABEL_DIR.glob(f"*{suffix}.json"))
    scenes = [Scene(name) for name in names]
    train, val = split_scenes(scenes)
    print(f"{len(train)} train scenes, {len(val)} val scenes")

    train_rows = []
    for group in GROUPS:
        rows = sample_random(train) if group == "random" else sample_class(train, group)
        train_rows += rows
        short = "" if len(rows) == SAMPLES_PER_GROUP else f"  (wanted {SAMPLES_PER_GROUP})"
        print(f"  {group}: {len(rows)}{short}")
    random.shuffle(train_rows)
    val_rows = tile_scenes(val)

    for split, rows in (("train", train_rows), ("val", val_rows)):
        write_split(rows, f"{out_path}/{split}")
        counts = Counter(label for _, _, label, _ in rows)
        print(f"{split}: {len(rows)} samples, labels {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
