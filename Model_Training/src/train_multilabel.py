import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 8
EPOCHS = 25

LEARNING_RATE = 1e-4

NUM_CLASSES = 4

CLASS_NAMES = [
    "healthy",
    "iron_deficiency",
    "vitamin_b12_deficiency",
    "vitamin_d_deficiency"
]


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset_multilabel"
)

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

TRAIN_CSV = os.path.join(
    DATASET_DIR,
    "train.csv"
)

VALIDATION_CSV = os.path.join(
    DATASET_DIR,
    "validation.csv"
)

SYNTHETIC_CSV = os.path.join(
    DATASET_DIR,
    "synthetic_labels.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "multilabel_efficientnetb3.keras"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MULTI-LABEL EFFICIENTNETB3 TRAINING")
print("=" * 60)

train_df = pd.read_csv(
    TRAIN_CSV
)

validation_df = pd.read_csv(
    VALIDATION_CSV
)

synthetic_df = pd.read_csv(
    SYNTHETIC_CSV
)


print("\nReal training images:", len(train_df))
print("Synthetic training images:", len(synthetic_df))
print("Validation images:", len(validation_df))


# ============================================================
# PREPARE REAL TRAINING DATA
# ============================================================

real_train_df = train_df.copy()

real_train_df["filepath"] = (
    real_train_df["image"]
    .apply(
        lambda x: os.path.join(
            IMAGE_DIR,
            x
        )
    )
)


# ============================================================
# PREPARE SYNTHETIC TRAINING DATA
# ============================================================

synthetic_df["filepath"] = (
    synthetic_df["image"]
    .apply(
        lambda x: os.path.join(
            DATASET_DIR,
            x
        )
    )
)


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

combined_train_df = pd.concat(
    [
        real_train_df,
        synthetic_df
    ],
    ignore_index=True
)


print(
    "\nTotal training images:",
    len(combined_train_df)
)


# ============================================================
# VERIFY FILES
# ============================================================

missing_files = []

for filepath in combined_train_df["filepath"]:

    if not os.path.exists(filepath):

        missing_files.append(
            filepath
        )


if missing_files:

    print("\nMissing files:")

    for filepath in missing_files[:10]:

        print(filepath)

    raise FileNotFoundError(
        f"\n{len(missing_files)} image files are missing."
    )


# ============================================================
# VALIDATION FILEPATHS
# ============================================================

validation_df["filepath"] = (
    validation_df["image"]
    .apply(
        lambda x: os.path.join(
            IMAGE_DIR,
            x
        )
    )
)


# ============================================================
# TF DATASET CREATION
# ============================================================

def load_image(filepath, labels):

    image = tf.io.read_file(
        filepath
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, labels


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="augmentation"
)


def augment_image(
    image,
    labels
):

    image = augmentation(
        image,
        training=True
    )

    return image, labels


# ============================================================
# CREATE DATASETS
# ============================================================

train_paths = combined_train_df[
    "filepath"
].values

train_labels = combined_train_df[
    CLASS_NAMES
].values.astype(
    np.float32
)


validation_paths = validation_df[
    "filepath"
].values

validation_labels = validation_df[
    CLASS_NAMES
].values.astype(
    np.float32
)


train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (
        validation_paths,
        validation_labels
    )
)


train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_dataset = train_dataset.shuffle(
    500,
    seed=SEED
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

validation_dataset = validation_dataset.batch(
    BATCH_SIZE
)


train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding EfficientNetB3...")

base_model = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

# Freeze backbone initially

base_model.trainable = False


inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)