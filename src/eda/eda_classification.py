import os
import matplotlib.pyplot as plt
from src.utils.artifact_saver import save_json, save_plot

def run_eda_classification(data_dir):
    output_dir = "eda_outputs"
    os.makedirs(output_dir, exist_ok=True)
    class_counts = {
        cls: len(os.listdir(os.path.join(data_dir, cls)))
        for cls in os.listdir(data_dir)
    }

    # plt.figure(figsize=(12, 5))
    # plt.bar(class_counts.keys(), class_counts.values())
    # plt.xticks(rotation=90)
    # plt.title("Classification Dataset Distribution")
    # plt.tight_layout()
    # plt.show()
    fig = plt.figure(figsize=(12, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=90)
    plt.title("Classification Class Distribution")

    save_json(class_counts, f"{output_dir}/classification_counts.json")
    save_plot(fig, f"{output_dir}/classification_distribution.png")
    plt.close(fig)

    return class_counts
