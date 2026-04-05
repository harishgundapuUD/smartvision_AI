import os, random, shutil
from pathlib import Path

def split_detection_dataset(
    images_dir,
    labels_dir,
    output_root,
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)

    images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)

    splits = {
        "train": images[:split_idx],
        "test": images[split_idx:]
    }

    for split, files in splits.items():
        img_out = Path(output_root) / "images" / split
        lbl_out = Path(output_root) / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in files:
            shutil.copy(
                Path(images_dir) / img,
                img_out / img
            )
            shutil.copy(
                Path(labels_dir) / img.replace(".jpg", ".txt"),
                lbl_out / img.replace(".jpg", ".txt")
            )

    return splits
