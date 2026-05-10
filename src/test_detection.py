import os
import json
import mlflow
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# =========================================================
# LOAD REGISTRY
# =========================================================
MODEL_DIR = "models/detection"  # same as training
with open(os.path.join(MODEL_DIR, "yolo_metrics.json"), "r") as f:
    registry = json.load(f)

run_id = registry["yolo"]["run_id"]

with open("utils/config.json", "r") as f:
    config = json.load(f)
# =========================================================
# MLflow SETUP
# =========================================================
mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
mlflow.set_tracking_uri(f"file://{mlruns_path.replace(os.sep, '/')}")

# =========================================================
# LOAD MODEL
# =========================================================
# Fetch the latest best.pt artifact from MLflow run
artifacts_uri = f"runs:/{run_id}/weights/best.pt"
model = YOLO(mlflow.artifacts.download_artifacts(artifacts_uri))

print("Model loaded successfully!")

# =========================================================
# TEST DATASET
# =========================================================
TEST_DIR = config["dataset_dirs"]["detection"]
TEST_DIR = os.path.join(TEST_DIR, "images", "val")  # change if needed
test_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(".jpg")]

# =========================================================
# PREDICTIONS & VISUALIZATION
# =========================================================
CONF_THRESHOLD = 0.5  # filter low confidence predictions

os.makedirs("detection_results", exist_ok=True)

for img_path in test_images:  # test on first 10 images
    results = model.predict(img_path, conf=CONF_THRESHOLD, save=False)
    
    # YOLOv8 returns a list of results per image
    r = results[0]

    # Get boxes, labels, and scores
    boxes = r.boxes.xyxy.cpu().numpy()  # bounding boxes: [x1, y1, x2, y2]
    scores = r.boxes.conf.cpu().numpy()  # confidence scores
    labels = r.boxes.cls.cpu().numpy().astype(int)  # class indices

    # Load image for visualization
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw boxes
    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls}:{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show image
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(os.path.basename(img_path))
    plt.savefig(f"detection_results/{os.path.basename(img_path)}")

# =========================================================
# OPTIONAL: Calculate simple detection accuracy
# =========================================================
# This is an approximate accuracy = (precision * recall) if you want
# You can also compute per-class metrics if you have ground-truth labels.