import os
import json
import yaml
import numpy as np
import streamlit as st
import tensorflow as tf
import mlflow

from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Classification App")

# =========================================================
# LOAD MODEL METRICS
# =========================================================

MODEL_METRICS_PATH = "models/classification/model_metrics.json"

with open(MODEL_METRICS_PATH, "r") as f:
    results = json.load(f)

# =========================================================
# SELECT MODEL
# =========================================================

# available_models = ["mobilenetv2"]

# selected_model = st.selectbox(
#     "Select Model",
#     available_models
# )

selected_model = "mobilenetv2"

st.subheader(f"Model: {selected_model}")

model_info = results["models"]["classification"][selected_model]

run_id = model_info["run_id"]

# =========================================================
# MLFLOW PATH
# =========================================================

MLRUNS_PATH = os.path.abspath(
    os.path.join(
        "models",
        "classification",
        "mlruns"
    )
)

mlflow.set_tracking_uri(
    f"file:///{MLRUNS_PATH.replace(os.sep, '/')}"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(run_id):

    # -------------------------------------------------
    # Find experiment directory
    # -------------------------------------------------

    experiment_dirs = os.listdir(MLRUNS_PATH)

    experiment_id = None

    for exp in experiment_dirs:

        possible_run = os.path.join(
            MLRUNS_PATH,
            exp,
            run_id
        )

        if os.path.exists(possible_run):

            experiment_id = exp
            break

    if experiment_id is None:
        raise Exception("Run ID not found")

    # -------------------------------------------------
    # Outputs directory
    # -------------------------------------------------

    outputs_dir = os.path.join(
        MLRUNS_PATH,
        experiment_id,
        run_id,
        "outputs"
    )

    output_folders = os.listdir(outputs_dir)

    yaml_path = os.path.join(
        outputs_dir,
        output_folders[0],
        "meta.yaml"
    )

    # -------------------------------------------------
    # Read YAML
    # -------------------------------------------------

    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    model_id = meta["destination_id"]

    model_path = os.path.join(
                                MLRUNS_PATH,
                                experiment_id,
                                # run_id,
                                "models",
                                model_id,
                                "artifacts",
                                "data",
                                "model.keras"
                            )

    model = tf.keras.models.load_model(model_path)

    return model

model = load_model(run_id)
# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = model_info.get("img_size", 224)

# =========================================================
# CLASS LABELS
# =========================================================

with open("utils/class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

CLASS_NAMES = list(CLASS_NAMES.keys())

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # DISPLAY IMAGE
    # -----------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    img = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    predicted_class = CLASS_NAMES[predicted_index]

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    st.success(f"Predicted Class: {predicted_class}")

    st.info(f"Confidence: {confidence:.4f}")

    # -----------------------------------------------------
    # SHOW ALL CLASS PROBABILITIES
    # -----------------------------------------------------

    # st.subheader("Class Probabilities")

    # for idx, class_name in enumerate(CLASS_NAMES):

    #     st.write(
    #         f"{class_name}: "
    #         f"{prediction[0][idx]:.4f}"
    #     )