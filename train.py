"""
train.py - script version of XView_w_YOLO.ipynb (the notebook is left unchanged).

Trains a YOLO model on the xView2 damage dataset. Defaults match the notebook
(yolov8n.pt, 3 epochs, imgsz 1024), but each can be overridden from the command line.

Usage (from the project root, x-view-images/):
    python train.py
    python train.py --model yolov8s.pt --epochs 50 --imgsz 1024 --name v8s_1024_50ep
    python train.py --predict          # also run the notebook's prediction step afterwards

Score the result with:
    python evaluate.py --weights <save_dir>/weights/best.pt --name <run name>
"""

import argparse
import os
import ssl
from pathlib import Path

import torch
from ultralytics import YOLO

# Paths relative to the project root (same as the notebook)
YAML_LOC = "X_View/datasets/data/data.yaml"
FOLDER_LOC = "X_View/datasets/data/"


def parse_args():
    ap = argparse.ArgumentParser(description="Train YOLO on xView2 damage data")
    ap.add_argument("--data", default=YAML_LOC,
                    help="dataset yaml (e.g. X_View/datasets/data/data_xbd.yaml for the extra xBD images)")
    ap.add_argument("--model", default="yolov8n.pt", help="starting weights / model name")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto-fit to GPU memory")
    ap.add_argument("--name", default="Trial", help="output folder name under Xview_mods/")
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', '0'...; default auto")
    ap.add_argument("--patience", type=int, default=100,
                    help="stop early if val mAP50-95 hasn't improved for this many epochs")
    ap.add_argument("--flipud", type=float, default=0.0,
                    help="probability of vertical flip augmentation (0.5 suits satellite images)")
    ap.add_argument("--predict", action="store_true",
                    help="after training, run the notebook's prediction step on val images")
    ap.add_argument("--conf", type=float, default=0.25, help="prediction confidence (with --predict)")
    ap.add_argument("--iou", type=float, default=0.9, help="prediction NMS IoU (with --predict)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Checking to see if we have a GPU available (ROCm GPUs also show up as 'cuda')
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cwd = os.getcwd()
    print(f"Working directory: {cwd}")
    print(f"Device: {device}")

    # Allow the model download to go through (same as the notebook)
    ssl._create_default_https_context = ssl._create_unverified_context
    model = YOLO(args.model)

    full_path = os.path.join(cwd, args.data)
    print(f"Dataset: {full_path}")
    project_path = os.path.join(cwd, FOLDER_LOC)

    # ---------------- Training ---------------- #
    model.train(
        data=full_path,                     # path to data.yaml
        epochs=args.epochs,
        project=project_path,
        name=f"./Xview_mods/{args.name}",   # name of the output folder
        device=device,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        flipud=args.flipud,
    )

    # Ultralytics adds -2, -3... if the folder already exists, so use the real path
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nTraining finished. Weights: {best}")
    print(f"Score it with:\n  python evaluate.py --weights \"{best}\" --name {args.name}\n")

    # ---------------- Optional: notebook's prediction step ---------------- #
    if args.predict:
        val_dir = Path(cwd, FOLDER_LOC, "val/images")
        if not best.is_file():
            raise FileNotFoundError(f"Model weights not found at: {best}")
        if not val_dir.is_dir():
            raise NotADirectoryError(f"Validation image folder not found at: {val_dir}")

        print(f"Using model: {best}")
        print(f"Running on validation images in: {val_dir}")
        YOLO(str(best)).predict(
            source=str(val_dir),
            imgsz=args.imgsz,       # match training size (the notebook defaulted to 640)
            conf=args.conf,
            iou=args.iou,
            save=True,              # save annotated images
            save_txt=True,          # also save YOLO txt outputs
            project="runs",
            exist_ok=True,
            verbose=True,
        )


# The guard is required on Windows: the data loader starts worker processes
# that re-import this file, and without it training would start again in each one.
if __name__ == "__main__":
    main()
