import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow.tensorflow

# =========================================================
# LOAD MODEL FROM REGISTRY
# =========================================================

with open("models/classification/model_metrics.json", "r") as f:
    registry = json.load(f)

model_info = registry["models"]["classification"]["mobilenetv2"]

run_id = model_info["run_id"]
model_name = "mobilenetv2"

with open("utils/config.json", "r") as f:
    config = json.load(f)
MODEL_DIR = os.path.join(
                            config["dl_model_dirs"]["base_dir"],
                            config["dl_model_dirs"]["classification"]
                        )
mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")

model_uri = f"runs:/{run_id}/{model_name}"

model = mlflow.tensorflow.load_model(model_uri)

# =========================================================
# LOAD CLASS NAMES
# =========================================================

TEST_DIR = "dataset/classification/test"

class_names = sorted(os.listdir(TEST_DIR))

# =========================================================
# IMAGE LOADING FUNCTION
# =========================================================

IMG_SIZE = 224

def load_image(img_path):
    img = tf.keras.utils.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0), img

# =========================================================
# GET SAMPLE IMAGES
# =========================================================

sample_images = []

for class_folder in os.listdir(TEST_DIR):

    class_path = os.path.join(TEST_DIR, class_folder)

    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path)[:2]:  # 2 images per class

        sample_images.append(os.path.join(class_path, img_name))

# =========================================================
# PREDICTION + DISPLAY
# =========================================================

plt.figure(figsize=(15, 10))

for i, img_path in enumerate(sample_images[:12]):

    img_array, img = load_image(img_path)

    preds = model.predict(img_array)[0]

    pred_class_idx = np.argmax(preds)
    pred_class = class_names[pred_class_idx]
    confidence = preds[pred_class_idx]

    # plt.subplot(3, 4, i + 1)

    plt.imshow(img)

    plt.title(f"{pred_class}\n{confidence:.2f}")

    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"sample_predictions_{i}.png")