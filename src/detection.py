import os
import json
import shutil
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

# =========================================================
# LOAD CONFIG (optional if you have)
# =========================================================

def get_latest_dir(parent_dir):
    # List all subdirectories
    subdirs = [f for f in Path(parent_dir).iterdir() if f.is_dir()]
    
    if not subdirs:
        return None  # No subdirectories found
    
    # Sort by modification time, newest first
    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    return str(latest_dir)

with open("utils/config.json", "r") as f:
    config = json.load(f)

MODEL_DIR = os.path.join(
    config["dl_model_dirs"]["base_dir"],
    config["dl_model_dirs"]["detection"]
)

dataset_dir = config["dataset_dirs"]["detection"]

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# MLFLOW SETUP
# =========================================================

mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
os.makedirs(mlruns_path, exist_ok=True)
trash_path = os.path.join(mlruns_path, ".trash")
if os.path.exists(trash_path) and os.path.isdir(trash_path):
    shutil.rmtree(trash_path)
    print(f"Deleted hidden folder: {trash_path}")

mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")

mlflow.set_experiment("yolo_detection")

# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO("yolov8n.pt")  # pretrained COCO model

# =========================================================
# TRAIN WITH MLFLOW
# =========================================================
mlflow.end_run()
with mlflow.start_run(run_name="yolo_v8_detection") as run:

    # -------------------------
    # LOG PARAMS
    # -------------------------

    mlflow.log_param("model", "yolov8n")
    mlflow.log_param("epochs", 50)
    mlflow.log_param("imgsz", 640)
    mlflow.log_param("batch", 16)

    # -------------------------
    # TRAIN
    # -------------------------

    results = model.train(
        data=os.path.join(dataset_dir, "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project=MODEL_DIR,
        name="yolo_run"
    )

    # -------------------------
    # VALIDATION METRICS
    # -------------------------

    metrics = model.val()

    # YOLOv8 metrics are stored in metrics.box.stats or metrics.box.keys()
    # Check available attributes
    print(metrics.box.__dict__)  # or print(dir(metrics.box))

    precision = float(metrics.box.p.mean())      # precision
    recall = float(metrics.box.r.mean())         # recall
    map50 = float(metrics.box.all_ap[0].mean())  # mAP50
    map5095 = float(metrics.box.all_ap[1].mean())  # mAP50-95

    # -------------------------
    # LOG METRICS
    # -------------------------

    mlflow.log_metric("mAP50", map50)
    mlflow.log_metric("mAP50-95", map5095)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # -------------------------
    # SAVE MODEL
    # -------------------------

    best_model_path = "runs/detect/models/detection"
    best_yolo = get_latest_dir(best_model_path)
    best_model_path = os.path.join(best_yolo, "weights/best.pt")

    mlflow.log_artifact(best_model_path)

    # -------------------------
    # RUN ID
    # -------------------------

    run_id = run.info.run_id

# =========================================================
# SAVE REGISTRY JSON
# =========================================================

registry = {
    "yolo": {
        "run_id": run_id,
        "mAP50": map50,
        "mAP50-95": map5095,
        "precision": precision,
        "recall": recall,
        "timestamp": datetime.now().isoformat()
    }
}
mlflow.end_run()
with open(os.path.join(MODEL_DIR, "yolo_metrics.json"), "w") as f:
    json.dump(registry, f, indent=4)

print("YOLO training completed")
print("Run ID:", run_id)