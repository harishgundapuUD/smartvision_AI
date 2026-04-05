import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_model(base_model, NUM_CLASSES):
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(base_model.input, output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_classifiers(train_dir, val_dir, img_size=(224,224), batch=16, epochs=5):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch
    )
    NUM_CLASSES = len(train_ds.class_names)
    models = {
        "vgg16": VGG16(weights="imagenet", include_top=False),
        "resnet50": ResNet50(weights="imagenet", include_top=False),
        "mobilenet": MobileNetV2(weights="imagenet", include_top=False),
        "efficientnet": EfficientNetB0(weights="imagenet", include_top=False)
    }

    for name, base in models.items():
        print(f"Training {name}")
        model = build_model(base, NUM_CLASSES)
        model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        model.save(f"models/{name}.h5")
