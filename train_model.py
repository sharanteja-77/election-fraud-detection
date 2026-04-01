"""
train_model.py
Standalone training script for the iris recognition CNN.

Usage
-----
    python train_model.py --data_dir data/iris_dataset --epochs 50 --batch 32

Dataset structure expected:
    data/iris_dataset/
        class_0001/   (one folder per voter / iris class)
            img1.jpg
            img2.png
            ...
        class_0002/
            ...

Each subfolder is treated as one voter class. The folder name becomes the class label.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import cv2

from utils.iris_preprocessor import IrisPreprocessor
from models.iris_model import IrisModel, INPUT_SHAPE


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load iris images from the dataset directory.

    Returns
    -------
    X       : (N, 64, 64, 1) float32 array
    y       : (N,) int32 array of class indices
    label_map : { class_index: folder_name }
    """
    preprocessor = IrisPreprocessor()
    data_path    = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Collect class folders
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class sub-folders found in {data_dir}")

    label_map = {i: d.name for i, d in enumerate(class_dirs)}
    print(f"[Train] Found {len(class_dirs)} classes in {data_dir}")

    X_list, y_list = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_idx, class_dir in enumerate(class_dirs):
        images = [p for p in class_dir.iterdir()
                  if p.suffix.lower() in extensions]
        print(f"  Class {class_idx:3d} ({class_dir.name}): {len(images)} images")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            processed = preprocessor.preprocess(img)
            if processed is None:
                continue
            X_list.append(processed)
            y_list.append(class_idx)

    if not X_list:
        raise ValueError("No valid images could be loaded. Check dataset format.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\n[Train] Dataset loaded: {X.shape[0]} samples, {len(class_dirs)} classes")
    return X, y, label_map


def augment_dataset(X: np.ndarray, y: np.ndarray,
                    factor: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment dataset to increase training samples.
    factor=3 means original + 2x augmented copies.
    """
    X_aug = [X]
    y_aug = [y]
    model_helper = IrisModel.__new__(IrisModel)

    for _ in range(factor - 1):
        X_aug.append(IrisModel.augment_batch(X))
        y_aug.append(y)

    X_all = np.concatenate(X_aug, axis=0)
    y_all = np.concatenate(y_aug, axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_all))
    return X_all[idx], y_all[idx]


def main():
    parser = argparse.ArgumentParser(description="Train iris recognition CNN")
    parser.add_argument("--data_dir", default="data/iris_dataset",
                        help="Path to dataset directory (default: data/iris_dataset)")
    parser.add_argument("--epochs",   type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch",    type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--augment",  type=int, default=3,
                        help="Augmentation factor (default: 3)")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split fraction (default: 0.15)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Election Fraud Detection — Iris Model Training")
    print("=" * 60)
    print(f"  Dataset  : {args.data_dir}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Augment  : {args.augment}x")
    print("=" * 60)

    # Load
    X, y, label_map = load_dataset(args.data_dir)

    # Augment
    if args.augment > 1:
        print(f"\n[Train] Augmenting dataset ({args.augment}x)…")
        X, y = augment_dataset(X, y, factor=args.augment)
        print(f"[Train] After augmentation: {len(X)} samples")

    # Train
    model = IrisModel()
    print("\n[Train] Starting training…\n")
    history = model.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch,
        val_split=args.val_split,
    )

    # Summary
    final_acc = history.get("val_accuracy", [0])[-1]
    print(f"\n[Train] ✓ Training complete.")
    print(f"[Train] Final validation accuracy: {final_acc*100:.2f}%")
    print(f"[Train] Model saved to: models/iris_model.h5")

    # Save label map
    import json
    with open("models/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print("[Train] Label map saved to: models/label_map.json")


if __name__ == "__main__":
    main()
