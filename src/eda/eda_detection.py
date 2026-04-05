import os

def run_eda_detection(image_dir, label_dir):
    images = len(os.listdir(image_dir))
    labels = len(os.listdir(label_dir))

    print(f"Detection images: {images}")
    print(f"Detection labels: {labels}")

    return images, labels
