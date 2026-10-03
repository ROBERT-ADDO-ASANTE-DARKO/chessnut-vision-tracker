"""
chess_yolo11seg_train.py
========================
Fine-tunes YOLO11-seg on the SAM3-labeled chess piece segmentation dataset.

After training, inference replaces the current two-step pipeline
(YOLO detection + SAM2.1 masking) with a single model that outputs
tight segmentation masks directly.

Run after chess_sam3_labeler.py has completed:
    python chess_yolo11seg_train.py
"""

import sys

# ── env — must precede all third-party imports ────────────────────────────────
_CLOUDSPACE_SP = '/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages'
if _CLOUDSPACE_SP not in sys.path:
    sys.path.insert(0, _CLOUDSPACE_SP)

import os
from ultralytics import YOLO
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_YAML  = '/teamspace/studios/this_studio/chess_seg_merged/data.yaml'
BASE_MODEL    = 'yolo11s-seg.pt'          # pretrained COCO checkpoint
PROJECT_DIR   = '/teamspace/studios/this_studio/runs/segment'
RUN_NAME      = 'chess_seg_v2'            # merged dataset run

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS        = 60
IMGSZ         = 640
BATCH         = 16
PATIENCE      = 15      # early stopping
LR0           = 0.01
WORKERS       = 4
DEVICE        = 0       # GPU 0

def main():
    if not Path(DATASET_YAML).exists():
        print(f"ERROR: Dataset not found at {DATASET_YAML}")
        print("Run chess_sam3_labeler.py first to generate the dataset.")
        return

    # Count samples
    train_dir = Path(DATASET_YAML).parent / 'images' / 'train'
    val_dir   = Path(DATASET_YAML).parent / 'images' / 'val'
    n_train   = len(list(train_dir.glob('*.jpg'))) if train_dir.exists() else 0
    n_val     = len(list(val_dir.glob('*.jpg')))   if val_dir.exists()   else 0
    print(f"Dataset: {n_train} train frames / {n_val} val frames")
    print(f"Base model : {BASE_MODEL}")
    print(f"Output     : {PROJECT_DIR}/{RUN_NAME}")
    print()

    model = YOLO(BASE_MODEL)

    results = model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        patience  = PATIENCE,
        lr0       = LR0,
        workers   = WORKERS,
        device    = DEVICE,
        project   = PROJECT_DIR,
        name      = RUN_NAME,
        pretrained= True,
        plots     = True,
        save      = True,
        # Augmentation — moderate; masks are clean so heavy aug risks
        # distorting polygon labels
        hsv_h     = 0.015,
        hsv_s     = 0.4,
        hsv_v     = 0.3,
        fliplr    = 0.5,
        degrees   = 5.0,
        translate = 0.05,
        scale     = 0.3,
        mosaic    = 0.5,   # half-rate mosaic (pieces look different in collage)
        mixup     = 0.0,   # off — mixing chess boards creates nonsensical layouts
    )

    best = Path(PROJECT_DIR) / RUN_NAME / 'weights' / 'best.pt'
    print(f"\nTraining complete.")
    print(f"Best weights : {best}")

    # Quick validation summary
    metrics = model.val()
    print(f"\nValidation results:")
    print(f"  mAP50 (box)  : {metrics.box.map50:.4f}")
    print(f"  mAP50 (mask) : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95 (mask): {metrics.seg.map:.4f}")


if __name__ == '__main__':
    main()
