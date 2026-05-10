import os
import json
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# =========================================================
# PRETRAINED MODELS
# =========================================================

from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNetV2,
    EfficientNetB0
)

from tensorflow.keras import mixed_precision

# =========================================================
# LOAD CONFIG
# =========================================================

with open("utils/config.json", "r") as f:
    config = json.load(f)

# =========================================================
# PATHS
# =========================================================

DATASET_DIR = config["dataset_dirs"]["classification"]

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(
                            config["dl_model_dirs"]["base_dir"],
                            config["dl_model_dirs"]["classification"]
                        )

MODEL_METRICS_PATH = os.path.join(
                                    MODEL_DIR,
                                    "model_metrics.json"
                                 )

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# GPU SETUP
# =========================================================

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPU is available. Using GPU.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("GPU not available. Using CPU.")

# =========================================================
# MLFLOW SETUP
# =========================================================

mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))

os.makedirs(mlruns_path, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")

mlflow.set_experiment("classification")

# =========================================================
# LOAD EXISTING RESULTS
# =========================================================

results = {
            "models": {
                        "classification": {}
                      }
          }

if os.path.exists(MODEL_METRICS_PATH):
    with open(MODEL_METRICS_PATH, "r") as f:
        existing_results = json.load(f)
    results = results | existing_results

# =========================================================
# MODEL FACTORY
# =========================================================

def get_base_model(architecture, img_size):

    if architecture == "VGG16":
        return VGG16(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(img_size, img_size, 3)
                    )

    elif architecture == "ResNet50":
        return ResNet50(
                            weights='imagenet',
                            include_top=False,
                            input_shape=(img_size, img_size, 3)
                        )

    elif architecture == "MobileNetV2":
        return MobileNetV2(
                                weights='imagenet',
                                include_top=False,
                                input_shape=(img_size, img_size, 3)
                            )

    elif architecture == "EfficientNetB0":
        return EfficientNetB0(
                                weights='imagenet',
                                include_top=False,
                                input_shape=(img_size, img_size, 3)
                             )

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

# =========================================================
# TRAIN ALL ENABLED MODELS
# =========================================================

for model_name, model_cfg in config["models"].items():

    if not model_cfg["enabled"]:
        continue

    print("\n====================================================")
    print(f"Training {model_name}")
    print("====================================================")

    # =====================================================
    # MIXED PRECISION
    # =====================================================

    if model_cfg.get("mixed_precision", False):
        mixed_precision.set_global_policy('mixed_float16')

    # =====================================================
    # PARAMS
    # =====================================================

    IMG_SIZE = model_cfg["img_size"]
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LEARNING_RATE = model_cfg["learning_rate"]
    NUM_CLASSES = model_cfg["num_classes"]

    # =====================================================
    # AUGMENTATION
    # =====================================================

    aug_cfg = model_cfg["augmentation"]

    train_datagen = ImageDataGenerator(
                                            rescale=1./255,
                                            rotation_range=aug_cfg.get("rotation_range", 0),
                                            width_shift_range=aug_cfg.get("width_shift_range", 0),
                                            height_shift_range=aug_cfg.get("height_shift_range", 0),
                                            zoom_range=aug_cfg.get("zoom_range", 0),
                                            horizontal_flip=aug_cfg.get("horizontal_flip", False)
                                        )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # =====================================================
    # DATA GENERATORS
    # =====================================================

    train_generator = train_datagen.flow_from_directory(
                                                            TRAIN_DIR,
                                                            target_size=(IMG_SIZE, IMG_SIZE),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical'
                                                        )

    val_generator = val_test_datagen.flow_from_directory(
                                                            VAL_DIR,
                                                            target_size=(IMG_SIZE, IMG_SIZE),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical'
                                                        )

    test_generator = val_test_datagen.flow_from_directory(
                                                            TEST_DIR,
                                                            target_size=(IMG_SIZE, IMG_SIZE),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical',
                                                            shuffle=False
                                                         )

    # =====================================================
    # BUILD MODEL
    # =====================================================

    base_model = get_base_model(
                                    model_cfg["architecture"],
                                    IMG_SIZE
                                )

    # =====================================================
    # FREEZE / FINETUNE
    # =====================================================

    if model_cfg.get("freeze_all", False):

        for layer in base_model.layers:
            layer.trainable = False

    elif model_cfg.get("fine_tune", False):

        unfreeze_layers = model_cfg.get("unfreeze_last_layers", 20)

        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True

    # =====================================================
    # CUSTOM CLASSIFIER
    # =====================================================

    model = Sequential()
    model.add(base_model)

    # -----------------------------------------------------

    if model_cfg.get("global_average_pooling", False):
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())

    # -----------------------------------------------------

    dense_units = model_cfg.get("dense_units", [])

    for units in dense_units:

        model.add(Dense(units, activation='relu'))

        # BatchNorm
        if model_cfg.get("batch_normalization", False):
            model.add(BatchNormalization())

        # Dropout
        model.add(Dropout(model_cfg.get("dropout_1", 0.3)))

    # -----------------------------------------------------

    model.add(
                Dense(
                        NUM_CLASSES,
                        activation='softmax',
                        dtype='float32'
                     )
            )

    # =====================================================
    # COMPILE
    # =====================================================

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

    model.summary()

    # =====================================================
    # CALLBACKS
    # =====================================================

    early_stop = EarlyStopping(
                                    monitor='val_loss',
                                    patience=5,
                                    restore_best_weights=True
                                )

    reduce_lr = ReduceLROnPlateau(
                                    monitor='val_loss',
                                    factor=0.2,
                                    patience=3,
                                    verbose=1
                                 )

    # =====================================================
    # TRAIN
    # =====================================================

    with mlflow.start_run(run_name=model_name) as run:

        # -------------------------------------------------
        # LOG PARAMS
        # -------------------------------------------------

        mlflow.log_param("architecture", model_cfg["architecture"])
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("num_classes", NUM_CLASSES)

        # -------------------------------------------------
        # FIT
        # -------------------------------------------------

        history = model.fit(
                                train_generator,
                                validation_data=val_generator,
                                epochs=EPOCHS,
                                callbacks=[early_stop, reduce_lr]
                            )

        # -------------------------------------------------
        # EVALUATE
        # -------------------------------------------------

        test_loss, test_accuracy = model.evaluate(test_generator)

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        train_accuracy = max(history.history["accuracy"])
        train_loss = min(history.history["loss"])
        val_accuracy = max(history.history["val_accuracy"])
        val_loss = min(history.history["val_loss"])

        # -------------------------------------------------
        # CUSTOM SCORE
        # -------------------------------------------------

        custom_score = (
                            0.5 * val_accuracy +
                            0.3 * test_accuracy +
                            0.2 * (1 - val_loss)
                        )

        # -------------------------------------------------
        # LOG METRICS
        # -------------------------------------------------

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("custom_score", custom_score)

        # -------------------------------------------------
        # LOG MODEL
        # -------------------------------------------------

        mlflow.tensorflow.log_model(
                                        model=model,
                                        artifact_path=model_name
                                    )

        # -------------------------------------------------
        # RUN ID
        # -------------------------------------------------

        run_id = run.info.run_id

    # =====================================================
    # STORE RESULTS
    # =====================================================

    results["models"]["classification"][model_name] = {
                                                        "run_id": run_id,

                                                        "architecture": model_cfg["architecture"],

                                                        "train_accuracy": float(train_accuracy),
                                                        "train_loss": float(train_loss),

                                                        "val_accuracy": float(val_accuracy),
                                                        "val_loss": float(val_loss),

                                                        "test_accuracy": float(test_accuracy),
                                                        "test_loss": float(test_loss),

                                                        "custom_score": float(custom_score),

                                                        "learning_rate": LEARNING_RATE,
                                                        "epochs": EPOCHS,
                                                        "batch_size": BATCH_SIZE,

                                                        "expected_accuracy": model_cfg.get("expected_accuracy", "N/A"),

                                                        "timestamp": datetime.now().isoformat(),

                                                        # "bestmodel": "yes"
                                                    }

    # =====================================================
    # SAVE JSON
    # =====================================================

    with open(MODEL_METRICS_PATH, "w") as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\nCompleted:", model_name)
    print("Run ID:", run_id)

# =========================================================
# END
# =========================================================

mlflow.end_run()

print("\n====================================================")
print("ALL TRAINING COMPLETED")
print("====================================================")

# =========================================================
# LOAD MODEL EXAMPLE
# =========================================================

"""
import json
import mlflow.tensorflow

with open(
    "models/classification/model_metrics.json",
    "r"
) as f:

    results = json.load(f)

run_id = results["models"]["classification"]["efficientnetb0"]["run_id"]

model_uri = f"runs:/{run_id}/efficientnetb0"

loaded_model = mlflow.tensorflow.load_model(model_uri)

print("Model loaded successfully.")
"""