"""Train MobileNetV2 (transfer learning) on a folder-per-class dataset.

Usage: python train.py [--data dataset] [--epochs 10] [--fine-tune-epochs 5]
Expected layout: dataset/<class_name>/*.jpg
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
BATCH = 32
SEED = 42
EXTS = {".jpg", ".jpeg", ".png"}
MODELS = Path("models")


def collect(data_dir: Path):
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if len(classes) < 2:
        raise SystemExit(f"Found no class folders in '{data_dir}'. See README for the dataset layout.")
    paths, labels = [], []
    for idx, name in enumerate(classes):
        for p in sorted((data_dir / name).iterdir()):
            if p.suffix.lower() in EXTS:
                paths.append(str(p))
                labels.append(idx)
    return classes, paths, labels


def make_ds(paths, labels, training):
    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",  # only active during training
    )(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 preprocessing
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--fine-tune-epochs", type=int, default=5)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    classes, paths, labels = collect(Path(args.data))
    print("Classes:", classes, "| images:", len(paths))

    # Stratified 70/15/15 split on file paths: no image appears in two sets.
    p_train, p_rest, y_train, y_rest = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    p_val, p_test, y_val, y_test = train_test_split(
        p_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED)

    train_ds = make_ds(p_train, y_train, True)
    val_ds = make_ds(p_val, y_val, False)

    model, base = build_model(len(classes))
    stop = lambda: tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True)

    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[stop()])

    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:  # keep early layers frozen
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=[stop()])

    MODELS.mkdir(exist_ok=True)
    model.save(MODELS / "model.keras")
    (MODELS / "class_names.json").write_text(json.dumps(classes, indent=2))
    (MODELS / "splits.json").write_text(json.dumps({"test_paths": p_test, "test_labels": y_test}))
    print("Saved model, class names and test split to 'models/'. Now run: python evaluate.py")


if __name__ == "__main__":
    main()
