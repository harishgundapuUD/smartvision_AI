import yaml
from pathlib import Path

def create_yolo_yaml(dataset_root, class_names):
    data = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/train",
        "test": "images/test",
        "names": class_names
    }

    yaml_path = Path(dataset_root) / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"✅ data.yaml created at: {yaml_path}")
    return str(yaml_path)
    