import os
import random
import shutil

# =========================================================
# PATHS
# =========================================================

IMAGE_DIR = "dataset/detection/images"
LABEL_DIR = "dataset/detection/labels"

OUT_IMG_TRAIN = "dataset/detection/images/train"
OUT_IMG_VAL   = "dataset/detection/images/val"

OUT_LBL_TRAIN = "dataset/detection/labels/train"
OUT_LBL_VAL   = "dataset/detection/labels/val"

# =========================================================
# CREATE FOLDERS
# =========================================================

for path in [
    OUT_IMG_TRAIN, OUT_IMG_VAL,
    OUT_LBL_TRAIN, OUT_LBL_VAL
]:
    os.makedirs(path, exist_ok=True)

# =========================================================
# GET ALL IMAGES
# =========================================================

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".jpg") or f.endswith(".png")
]

random.shuffle(images)

# 80-20 SPLIT
split_idx = int(0.8 * len(images))

train_images = images[:split_idx]
val_images = images[split_idx:]

# =========================================================
# MOVE FILES
# =========================================================

def copy_data(image_list, img_out, lbl_out):

    for img_name in image_list:

        base = os.path.splitext(img_name)[0]

        img_src = os.path.join(IMAGE_DIR, img_name)
        lbl_src = os.path.join(LABEL_DIR, base + ".txt")

        img_dst = os.path.join(img_out, img_name)
        lbl_dst = os.path.join(lbl_out, base + ".txt")

        shutil.copy(img_src, img_dst)

        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)

copy_data(train_images, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
copy_data(val_images, OUT_IMG_VAL, OUT_LBL_VAL)

print("Dataset split completed successfully!")