from src.eda.eda_classification import run_eda_classification
from src.eda.eda_detection import run_eda_detection
from src.train_classifier import train_classifiers
from src.evaluate_classifier import evaluate_classifiers
from src.train_yolo import train_yolo
from src.evaluate_yolo import evaluate_yolo
from src.utils.yolo_config import create_yolo_yaml
from src.data.detection_split import split_detection_dataset


DATASET = "dataset"

if __name__ == "__main__":
    CLASS_NAMES = [
                    "person","car","truck","bus","motorcycle","bicycle",
                    "airplane","traffic light","stop sign","bench",
                    "dog","cat","horse","bird","cow","elephant",
                    "bottle","cup","bowl","pizza","cake",
                    "chair","couch","bed","potted plant"
        ]



    # ---- EDA ----
    run_eda_classification(f"{DATASET}/classification/train")
    run_eda_detection(
        f"{DATASET}/detection/images",
        f"{DATASET}/detection/labels"
    )

    # ---- CLASSIFICATION ----
    train_classifiers(
        f"{DATASET}/classification/train",
        f"{DATASET}/classification/val"
    )

    cls_results = evaluate_classifiers(
        f"{DATASET}/classification/test"
    )
    print("Classification Results:", cls_results)

    # ---- YOLO DETECTION ----
    # 3. Split detection data
    split_detection_dataset(
        "dataset/detection/images",
        "dataset/detection/labels",
        "dataset/detection_split"
    )

    # 4. Create YOLO yaml
    yaml_path = create_yolo_yaml(
        "dataset/detection_split",
        CLASS_NAMES
    )

    # 5. Train & evaluate YOLO
    train_yolo(yaml_path)
    evaluate_yolo("runs/detect/train/weights/best.pt")
    # data_yaml = create_yolo_data_yaml(
    #     dataset_root=f"{DATASET}/detection",
    #     class_names=CLASS_NAMES
    # )

    train_yolo(yaml_path)
    yolo_results = evaluate_yolo("runs/detect/train/weights/best.pt")
    print("YOLO Results:", yolo_results)
