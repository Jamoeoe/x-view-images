"""
autopilot.py - run a queue of training experiments unattended, evaluating each one
and choosing the next experiment from the results.

Stage 1 (always):
  A. yolov8m on xBD data                 -> does a bigger model help now that there's more data?
  B. yolov8s on xBD data + vertical flips -> does flip augmentation help?
Stage 2 (chosen from Stage 1, compared against the current best, v8s_xbd = 0.2844 damage mAP50):
  - medium helped            -> yolov8l (+ flips if flips helped)
  - medium didn't, flips did -> yolov8m + flips
  - neither helped           -> YOLO11s (newer architecture, same size)
  then, if time is left, one more candidate from the same list.

Every run uses --patience 30 (stops once it plateaus), is evaluated with evaluate.py
(rows land in eval_results.csv), and every step is logged to autopilot_log.txt.
A run still going at the deadline is stopped and its best.pt is evaluated.

Usage (from the project root, in the venv):
    python autopilot.py --hours 19
    python autopilot.py --hours 19 --dry-run      # print the plan only
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

DATA = "X_View/datasets/data/data_xbd.yaml"
RUNS = Path("X_View/datasets/data/Xview_mods")
RESULTS = Path("eval_results.csv")
LOG = Path("autopilot_log.txt")
MIN_GAIN = 0.01          # damage mAP50 improvement that counts as "real" (noise was ~0.001)
EVAL_RESERVE_H = 0.4     # time kept back at the end for the final evaluation


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def unique_name(name):
    n, i = name, 2
    while (RUNS / n).exists():
        n, i = f"{name}_{i}", i + 1
    return n


def score_of(name):
    """damage_mAP50 of the last row in eval_results.csv with this name (or None)."""
    if not RESULTS.is_file():
        return None
    with open(RESULTS, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("name") == name]
    try:
        return float(rows[-1]["damage_mAP50"]) if rows else None
    except (KeyError, ValueError):
        return None


def kill_tree(proc):
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.kill()


def run_experiment(label, train_args, deadline, dry):
    """Train + evaluate one config. Returns (run name, damage_mAP50 or None)."""
    name = unique_name(label)
    cmd = [sys.executable, "train.py", "--data", DATA, "--epochs", "100",
           "--patience", "30", "--name", name] + train_args
    log(f"START {name}: {' '.join(cmd[1:])}")
    if dry:
        return name, None

    stop_at = deadline - timedelta(hours=EVAL_RESERVE_H)
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    stopped = False
    while proc.poll() is None:
        if datetime.now() >= stop_at:
            log(f"Deadline reached - stopping {name} (best.pt so far is kept)")
            kill_tree(proc)
            stopped = True
            break
        time.sleep(30)
    log(f"{name} training {'stopped' if stopped else 'exit code ' + str(proc.returncode)}"
        f" after {(time.time() - t0) / 3600:.1f} h")

    best = RUNS / name / "weights" / "best.pt"
    if not best.is_file():
        log(f"{name}: no best.pt found - skipping evaluation")
        return name, None
    ev = subprocess.run([sys.executable, "evaluate.py", "--weights", str(best), "--name", name])
    s = score_of(name)
    log(f"{name}: evaluate exit code {ev.returncode}, damage_mAP50 = {s}")
    return name, s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=19.0, help="total time budget")
    ap.add_argument("--baseline", default="v8s_xbd", help="row in eval_results.csv to beat")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    deadline = datetime.now() + timedelta(hours=args.hours)
    base = score_of(args.baseline)
    log(f"=== autopilot start, budget {args.hours} h, deadline {deadline:%a %H:%M}, "
        f"baseline {args.baseline} = {base} ===")
    if base is None:
        base = 0.2844
        log(f"Baseline row not found; using {base}")

    def time_left_h():
        return (deadline - datetime.now()).total_seconds() / 3600

    # ---- Stage 1 ----
    _, s_m = run_experiment("v8m_xbd", ["--model", "yolov8m.pt"], deadline, args.dry_run)
    if time_left_h() < 1.5:
        log("Not enough time left for stage 1B. Done.")
        return
    _, s_flip = run_experiment("v8s_xbd_flip", ["--model", "yolov8s.pt", "--flipud", "0.5"],
                               deadline, args.dry_run)

    size_helps = s_m is not None and s_m > base + MIN_GAIN
    flip_helps = s_flip is not None and s_flip > base + MIN_GAIN
    flip = ["--flipud", "0.5"] if flip_helps else []
    log(f"Stage 1 results: medium={s_m}, flips={s_flip}, baseline={base} -> "
        f"size_helps={size_helps}, flip_helps={flip_helps}")

    # ---- Stage 2: ordered candidates ----
    if size_helps:
        candidates = [("v8l_xbd" + ("_flip" if flip_helps else ""),
                       ["--model", "yolov8l.pt", "--batch", "8"] + flip),
                      ("y11m_xbd" + ("_flip" if flip_helps else ""),
                       ["--model", "yolo11m.pt"] + flip)]
    elif flip_helps:
        candidates = [("v8m_xbd_flip", ["--model", "yolov8m.pt"] + flip),
                      ("y11s_xbd_flip", ["--model", "yolo11s.pt"] + flip)]
    else:
        candidates = [("y11s_xbd", ["--model", "yolo11s.pt"]),
                      ("y11m_xbd", ["--model", "yolo11m.pt"])]
    if args.dry_run:
        log("(dry run: stage 2 depends on stage 1 scores; showing the 'neither helped' branch)")

    for label, extra in candidates:
        if time_left_h() < 2.0:
            log(f"Only {time_left_h():.1f} h left - not starting {label}.")
            break
        run_experiment(label, extra, deadline, args.dry_run)

    log("=== autopilot finished ===")


if __name__ == "__main__":
    main()
