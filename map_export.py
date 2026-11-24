"""
map_export.py - prepare data for the damage map (damage_map.py).

Run once after damage_classifier.py has finished:
  1. Re-reads the xBD hurricane labels in the same order damage_classifier.py used
     for its crop cache, and pairs every cropped building with its real-world
     outline (lng_lat polygons) and its source image.
  2. Runs the best classifier (highest macro F1 in classifier_results.csv, or --model)
     on every building -> predicted class + confidence.
  3. Works out each image tile's map corners (fits pixel -> lon/lat from the outlines),
     so the map can overlay the real before/after satellite image.
  4. Writes mock tweets (tweets/mock_tweets.csv) unless real tweets are already there.

Output: map_data/  (buildings.json, images.json, polygons.json, meta.json)

Usage (project root, venv active):
    python map_export.py
    python map_export.py --model prepost_r34
    python map_export.py --no-model        # ground-truth only (no predictions)
"""

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import damage_classifier as dc

OUT = Path("map_data")
TWEETS = Path("tweets")
PAIR_RE = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_geo(json_path):
    """Same filtering as damage_classifier.parse_buildings, plus uid + lon/lat ring + pixel ring."""
    d = json.loads(Path(json_path).read_text())
    ll_by_uid = {f["properties"].get("uid"): f["wkt"] for f in d["features"].get("lng_lat", [])}
    out = []
    for f in d["features"]["xy"]:
        p = f.get("properties", {})
        if p.get("feature_type", "building") != "building":
            continue
        cid = dc.SUBTYPE_TO_ID.get(p.get("subtype"))
        if cid is None:
            continue
        rings = []
        for r in dc.RING_RE.findall(f.get("wkt", "")):
            pts = [tuple(map(float, xy.split()[:2])) for xy in r.split(",") if xy.strip()]
            if len(pts) >= 3:
                rings.append(np.array(pts, dtype=np.float32))
        if not rings:
            continue
        allp = np.concatenate(rings)
        if allp[:, 0].max() - allp[:, 0].min() < 2 or allp[:, 1].max() - allp[:, 1].min() < 2:
            continue
        ll = [(float(a), float(b)) for a, b in PAIR_RE.findall(ll_by_uid.get(p.get("uid"), ""))]
        out.append({"cid": cid, "uid": p.get("uid"), "xy": allp, "ll": ll})
    return out


def fit_affine(xy, ll):
    """Least-squares pixel(x,y) -> (lon,lat). Returns 3x2 matrix or None."""
    if len(xy) < 3 or len(xy) != len(ll):
        return None
    A = np.c_[np.asarray(xy, float), np.ones(len(xy))]
    M, *_ = np.linalg.lstsq(A, np.asarray(ll, float), rcond=None)
    return M


def pick_model(name):
    if name:
        return name
    best, best_f1 = None, -1
    if dc.RESULTS.is_file():
        for r in csv.DictReader(open(dc.RESULTS, newline="")):
            if r.get("smoke") in ("True", "true", "1"):
                continue
            try:
                f1 = float(r.get("macro_F1", "nan"))
            except ValueError:
                continue
            if f1 > best_f1 and (dc.RUNS / r["name"] / "best.pt").is_file():
                best, best_f1 = r["name"], f1
    return best


def predict(model_name, x_path, n):
    import torch
    import torch.nn.functional as F
    kind, tag = model_name.split("_")
    arch = {"r18": "resnet18", "r34": "resnet34", "r50": "resnet50"}[tag]
    chans = dc.VARIANT_CHANNELS[kind]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = dc.build_model(arch, len(chans), 4, print)
    sd = torch.load(dc.RUNS / model_name / "best.pt", map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    model = model.to(device).eval()
    X = np.load(x_path, mmap_mode="r")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    probs = []
    with torch.no_grad():
        for s in range(0, n, 512):
            a = np.asarray(X[s:min(n, s + 512)][..., chans])
            t = torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float() / 255.0
            for j in range((len(chans) - 1) // 3):
                t[:, 3 * j:3 * j + 3] = (t[:, 3 * j:3 * j + 3] - mean[:, None, None]) / std[:, None, None]
            t[:, -1] = (t[:, -1] - 0.5) / 0.5
            probs.append(F.softmax(model(t).float(), 1).cpu().numpy())
            if (s // 512) % 20 == 0:
                print(f"  predicted {min(n, s + 512)}/{n}")
    return np.concatenate(probs)


# --------------------------------------------------------------------------- #
# Mock tweets
# --------------------------------------------------------------------------- #
LANDFALL = {"hurricane-florence": datetime(2018, 9, 14, 7), "hurricane-harvey": datetime(2017, 8, 25, 22),
            "hurricane-matthew": datetime(2016, 10, 4, 11), "hurricane-michael": datetime(2018, 10, 10, 17)}
CRIT = [
    "Water coming in fast, we're on the second floor with 3 kids. Need boat rescue ASAP",
    "Roof just came off our house, elderly neighbor next door not answering. Please send help",
    "Tree through the living room, my dad is hurt and can't get out. Road is blocked",
    "Power lines down across the street and sparking, house next to it is on fire",
    "Stranded on our roof, water still rising. 2 adults 1 dog. Please share",
    "Whole block flattened near the school. People trapped, we hear them calling",
    "Car swept off the road by floodwater, driver still inside. Need rescue now",
    "Nursing home on our street has no power and water is coming in. Patients still inside",
    "Gas smell everywhere after the house across from us collapsed. Nobody can reach 911",
    "Our apartment building partially collapsed. Several families still unaccounted for",
]
NONCRIT = [
    "Power's out but we're safe. Stay safe everyone",
    "Lots of branches and debris down on our street but everyone here is ok",
    "Shelter at the high school still has room and hot food",
    "Wind finally calming down. Going to check on the neighbors in the morning",
    "Thank you to the line crews already out working tonight",
    "Roads in our neighborhood are passable now, some fences down",
    "Grocery store on the main road reopened with generators",
    "Fence and shingles gone but house is fine. Grateful",
    "Anyone know when power is coming back? Phone at 10%",
    "Prayers for everyone hit by this storm",
]


def make_mock_tweets(lat, lon, hur_names, hur_idx, true_cls, path, seed=1):
    rng = random.Random(seed)
    rows, tid = [], 1
    for h, name in enumerate(hur_names):
        sel = np.where(hur_idx == h)[0]
        if len(sel) == 0:
            continue
        severe = sel[np.isin(true_cls[sel], (2, 3))]
        storm = name.split("-")[-1].title()
        for kind, pool, n in (("Critical", severe if len(severe) else sel, 50), ("NonCritical", sel, 100)):
            for _ in range(n):
                i = rng.choice(list(pool))
                t0 = LANDFALL.get(name, datetime(2018, 1, 1))
                rows.append({
                    "tweet_id": f"mock{tid:05d}",
                    "text": rng.choice(CRIT if kind == "Critical" else NONCRIT) + f" #Hurricane{storm}",
                    "user": f"@mock_user{rng.randint(100, 9999)}",
                    "created_at": (t0 + timedelta(minutes=rng.randint(-600, 2880))).isoformat(timespec="minutes"),
                    "latitude": round(float(lat[i]) + rng.gauss(0, 0.0015), 6),
                    "longitude": round(float(lon[i]) + rng.gauss(0, 0.0015), 6),
                    "label": kind,
                    "is_mock": True,
                })
                tid += 1
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} mock tweets to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xbd", default="../train")
    ap.add_argument("--cache", default="../xbd_crops/full")
    ap.add_argument("--model", default=None, help="clf_runs name; default = best macro F1")
    ap.add_argument("--no-model", action="store_true", help="skip predictions (ground truth only)")
    ap.add_argument("--remake-mock", action="store_true", help="rewrite tweets/mock_tweets.csv")
    args = ap.parse_args()

    cache = Path(args.cache)
    meta = json.loads((cache / "meta.json").read_text())
    params = meta["params"]
    idx = np.load(cache / "index.npz")
    n = int(meta["n"])
    x_path = cache / f"crops_{params['size']}.npy"

    # Rebuild the exact item list damage_classifier used
    lbl_dir, img_dir = Path(args.xbd) / "labels", Path(args.xbd) / "images"
    jsons = sorted(p for p in lbl_dir.glob("*_post_disaster.json")
                   if any(p.name.startswith(d + "_") for d in params["disasters"]))
    if len(jsons) != params["n_json"]:
        sys.exit(f"[error] found {len(jsons)} label files, crop cache was built from {params['n_json']}")
    hur_names = params["disasters"]
    lat, lon, hur, img, polys, images = [], [], [], [], {}, []
    for jp in jsons:
        stem = jp.stem
        pre_name = stem.replace("_post_disaster", "_pre_disaster")
        if not (img_dir / f"{stem}.png").is_file() or not (img_dir / f"{pre_name}.png").is_file():
            continue
        blds = parse_geo(jp)
        if not blds:
            continue
        k = len(images)
        h = next(i for i, d in enumerate(hur_names) if stem.startswith(d + "_"))
        xy_all = np.concatenate([b["xy"] for b in blds if len(b["ll"]) == len(b["xy"])] or [np.zeros((0, 2))])
        ll_all = [p for b in blds if len(b["ll"]) == len(b["xy"]) for p in b["ll"]]
        M = fit_affine(xy_all, ll_all)
        corners = None
        if M is not None:
            c = np.c_[np.array([[0, 1024], [0, 0], [1024, 0], [1024, 1024]], float), np.ones(4)] @ M
            corners = [[round(float(a), 7), round(float(b), 7)] for a, b in c]  # [lon,lat]: BL,TL,TR,BR
        images.append({"name": stem, "pre": pre_name, "h": h, "corners": corners})
        polys[k] = []
        for b in blds:
            if b["ll"]:
                ll = b["ll"]
            elif M is not None:          # no lon/lat in the label -> convert pixels with the fit
                ll = [tuple(v) for v in (np.c_[b["xy"], np.ones(len(b["xy"]))] @ M)]
            else:
                ll = [(0.0, 0.0)]
            ll = np.array(ll, float)
            lon.append(float(ll[:, 0].mean()))
            lat.append(float(ll[:, 1].mean()))
            hur.append(h)
            img.append(k)
            polys[k].append([[round(float(a), 6), round(float(b_), 6)] for a, b_ in ll])
    if len(lat) != n:
        sys.exit(f"[error] matched {len(lat)} buildings but crop cache has {n}; was the cache rebuilt "
                 f"with different settings?")
    true_cls = idx["y"][:n]
    if not np.array_equal(np.array(img), idx["img_id"][:n]):
        sys.exit("[error] building order does not match the crop cache")
    print(f"Matched {n} buildings across {len(images)} images")

    model_name = None if args.no_model else pick_model(args.model)
    if model_name:
        print(f"Predicting with classifier '{model_name}' ...")
        probs = predict(model_name, x_path, n)
        pred, conf = probs.argmax(1), probs.max(1)
        p_sev = probs[:, 2] + probs[:, 3]
    else:
        print("No classifier used - map will show ground truth only.")
        pred, conf, p_sev = true_cls.copy(), np.ones(n), np.isin(true_cls, (2, 3)).astype(float)

    OUT.mkdir(exist_ok=True)
    buildings = {
        "lat": [round(v, 6) for v in lat], "lon": [round(v, 6) for v in lon],
        "h": hur, "img": img, "true": true_cls.tolist(), "pred": pred.tolist(),
        "conf": [round(float(v), 3) for v in conf], "psev": [round(float(v), 3) for v in p_sev],
        "val": idx["is_val"][:n].astype(int).tolist(),
    }
    (OUT / "buildings.json").write_text(json.dumps(buildings, separators=(",", ":")))
    (OUT / "images.json").write_text(json.dumps(images, separators=(",", ":")))
    (OUT / "polygons.json").write_text(json.dumps(polys, separators=(",", ":")))
    (OUT / "meta.json").write_text(json.dumps({
        "model": model_name, "hurricanes": hur_names, "classes": dc.CLASSES,
        "xbd_images": str(img_dir.resolve()), "created": datetime.now().isoformat(timespec="seconds"),
        "n_buildings": n, "n_val": int(idx["is_val"][:n].sum())}, indent=2))
    print(f"Wrote map data to {OUT.resolve()}")

    mock = TWEETS / "mock_tweets.csv"
    real = [p for p in TWEETS.glob("*.csv") if p.name != mock.name] if TWEETS.is_dir() else []
    if args.remake_mock or (not real and not mock.is_file()):
        make_mock_tweets(np.array(lat), np.array(lon), hur_names, np.array(hur), true_cls, mock)
    elif real:
        print(f"Real tweet files found ({', '.join(p.name for p in real)}) - mock tweets not needed.")


if __name__ == "__main__":
    main()
