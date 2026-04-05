from ultralytics import YOLO

def evaluate_yolo(best_weights):
    model = YOLO(best_weights)
    metrics = model.val()
    return metrics
