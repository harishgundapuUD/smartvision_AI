import tensorflow as tf

def evaluate_classifiers(test_dir, img_size=(224,224), batch=16):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch
    )

    results = {}
    for name in ["vgg16", "resnet50", "mobilenet", "efficientnet"]:
        model = tf.keras.models.load_model(f"models/{name}.h5")
        results[name] = model.evaluate(test_ds)

    return results
