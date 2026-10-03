"""
chessnut_evo_yolo11seg_finetune.py
===================================
Fine-tunes the existing chess-piece YOLO11-seg model (runs/segment/seg_v2,
trained on 246 hand/SAM3-labeled frames) on the small hand-labeled
Chessnut Evo dataset (chessnut_evo_seg_dataset/, 18 train / 3 val frames).

Warm-starts from seg_v2's weights rather than a COCO checkpoint, and uses a
low LR + frozen backbone + heavy patience since the fine-tuning set is tiny
relative to the base dataset — the goal is to adapt to Chessnut Evo's piece
shapes/lighting without forgetting what seg_v2 already learned.

Run:
    python chessnut_evo_yolo11seg_finetune.py
"""

import sys

_CLOUDSPACE_SP = '/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages'
_SYSCONDA_SP   = '/system/conda/miniconda3/lib/python3.10/site-packages'
sys.path = [p for p in sys.path if _CLOUDSPACE_SP not in p and _SYSCONDA_SP not in p]
sys.path.insert(0, _SYSCONDA_SP)
sys.path.insert(0, _CLOUDSPACE_SP)

from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_YAML = '/teamspace/studios/this_studio/chessnut_evo_seg_dataset/data.yaml'
BASE_MODEL   = '/teamspace/studios/this_studio/runs/segment/seg_v2/weights/best.pt'
PROJECT_DIR  = '/teamspace/studios/this_studio/runs/segment'
RUN_NAME     = 'chessnut_evo_ft_v1'

# ── Hyperparameters — tuned for a tiny (18-image) fine-tuning set ─────────────
EPOCHS    = 300
IMGSZ     = 640
BATCH     = 4
PATIENCE  = 40
LR0       = 0.001     # 10x lower than seg_v2's — avoid destroying base weights
FREEZE    = 10        # freeze backbone layers, only adapt neck/head
WORKERS   = 4
DEVICE    = 0


def main():
    if not Path(DATASET_YAML).exists():
        print(f"ERROR: Dataset not found at {DATASET_YAML}")
        return
    if not Path(BASE_MODEL).exists():
        print(f"ERROR: Base model not found at {BASE_MODEL}")
        return

    train_dir = Path(DATASET_YAML).parent / 'images' / 'train'
    val_dir   = Path(DATASET_YAML).parent / 'images' / 'val'
    n_train   = len(list(train_dir.glob('*.jpg')))
    n_val     = len(list(val_dir.glob('*.jpg')))
    print(f"Dataset: {n_train} train frames / {n_val} val frames")
    print(f"Base model : {BASE_MODEL}")
    print(f"Output     : {PROJECT_DIR}/{RUN_NAME}")
    print()

    model = YOLO(BASE_MODEL)

    model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        patience  = PATIENCE,
        lr0       = LR0,
        freeze    = FREEZE,
        workers   = WORKERS,
        device    = DEVICE,
        project   = PROJECT_DIR,
        name      = RUN_NAME,
        pretrained= True,
        plots     = True,
        save      = True,
        cos_lr    = True,
        # Moderate augmentation — masks are clean, heavy aug risks distorting
        # polygon labels on an already-small dataset
        hsv_h     = 0.015,
        hsv_s     = 0.4,
        hsv_v     = 0.3,
        fliplr    = 0.5,
        degrees   = 5.0,
        translate = 0.05,
        scale     = 0.3,
        mosaic    = 0.5,
        mixup     = 0.0,
    )

    best = Path(PROJECT_DIR) / RUN_NAME / 'weights' / 'best.pt'
    print(f"\nTraining complete.")
    print(f"Best weights : {best}")

    metrics = model.val()
    print(f"\nValidation results:")
    print(f"  mAP50 (box)  : {metrics.box.map50:.4f}")
    print(f"  mAP50 (mask) : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95 (mask): {metrics.seg.map:.4f}")


if __name__ == '__main__':
    main()
