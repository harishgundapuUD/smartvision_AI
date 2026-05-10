import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow.tensorflow
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# LOAD MODEL REGISTRY
# =========================================================

with open("models/classification/model_metrics.json", "r") as f:
    registry = json.load(f)

models_info = registry["models"]["classification"]

# =========================================================
# TEST DATASET
# =========================================================

TEST_DIR = "dataset/classification/test"
IMG_SIZE = 224
BATCH_SIZE = 32

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())

# =========================================================
# STORAGE
# =========================================================

results = {}

best_model = None
best_score = -1

# =========================================================
# EVALUATION LOOP
# =========================================================

for model_name, info in models_info.items():

    print("\n====================================================")
    print(f"Evaluating: {model_name}")
    print("====================================================")

    run_id = info["run_id"]

    model_uri = f"runs:/{run_id}/{model_name}"

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = mlflow.tensorflow.load_model(model_uri)

    # -----------------------------------------------------
    # INFERENCE TIME
    # -----------------------------------------------------

    start_time = time.time()

    y_pred_prob = model.predict(test_generator)

    end_time = time.time()

    inference_time = end_time - start_time

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"Confusion Matrix - {model_name}")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(f"{model_name}_confusion_matrix.png")

    plt.close()

    # -----------------------------------------------------
    # STORE RESULTS
    # -----------------------------------------------------

    results[model_name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "inference_time_sec": float(inference_time)
    }

    # -----------------------------------------------------
    # ACCURACY–SPEED TRADEOFF SCORE
    # -----------------------------------------------------

    score = accuracy - (0.01 * inference_time)

    if score > best_score:
        best_score = score
        best_model = model_name

# =========================================================
# SAVE RESULTS JSON
# =========================================================

with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)

# =========================================================
# VISUAL COMPARISON PLOTS
# =========================================================

models = list(results.keys())

accuracy = [results[m]["accuracy"] for m in models]
precision = [results[m]["precision"] for m in models]
recall = [results[m]["recall"] for m in models]
f1 = [results[m]["f1_score"] for m in models]
time_vals = [results[m]["inference_time_sec"] for m in models]

plt.figure(figsize=(14, 6))

# Accuracy
plt.subplot(1, 3, 1)
plt.bar(models, accuracy)
plt.title("Accuracy")
plt.xticks(rotation=45)

# F1 Score
plt.subplot(1, 3, 2)
plt.bar(models, f1)
plt.title("F1 Score")
plt.xticks(rotation=45)

# Inference Time
plt.subplot(1, 3, 3)
plt.bar(models, time_vals)
plt.title("Inference Time (sec)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("model_comparison_overview.png")

plt.show()

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n====================================================")
print("MODEL COMPARISON COMPLETE")
print("====================================================")

print("\nBest Model (Accuracy-Speed Tradeoff):", best_model)

print("\nDetailed Results:")

for m, r in results.items():

    print(f"""
    {m}
    Accuracy: {r['accuracy']:.4f}
    Precision: {r['precision']:.4f}
    Recall: {r['recall']:.4f}
    F1 Score: {r['f1_score']:.4f}
    Inference Time: {r['inference_time_sec']:.2f}s
    """)

print("====================================================")