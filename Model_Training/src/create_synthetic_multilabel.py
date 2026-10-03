import os
import random
import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance, ImageFilter


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

OUTPUT_DIR = os.path.join(
    DATASET_DIR,
    "synthetic_images"
)

OUTPUT_CSV = os.path.join(
    DATASET_DIR,
    "synthetic_labels.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Number generated for each combination
IMAGES_PER_COMBINATION = 100


# ============================================================
# LABEL DEFINITIONS
# ============================================================

LABEL_COLUMNS = [
    "healthy",
    "iron_deficiency",
    "vitamin_b12_deficiency",
    "vitamin_d_deficiency"
]


COMBINATIONS = {

    "iron_b12": {
        "labels": [0, 1, 1, 0],
        "classes": [
            "iron_deficiency",
            "vitamin_b12_deficiency"
        ]
    },

    "iron_d": {
        "labels": [0, 1, 0, 1],
        "classes": [
            "iron_deficiency",
            "vitamin_d_deficiency"
        ]
    },

    "b12_d": {
        "labels": [0, 0, 1, 1],
        "classes": [
            "vitamin_b12_deficiency",
            "vitamin_d_deficiency"
        ]
    },

    "iron_b12_d": {
        "labels": [0, 1, 1, 1],
        "classes": [
            "iron_deficiency",
            "vitamin_b12_deficiency",
            "vitamin_d_deficiency"
        ]
    }
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING CSV
# ============================================================

print("=" * 60)
print("CREATING SYNTHETIC MULTI-LABEL DATA")
print("=" * 60)

if not os.path.exists(TRAIN_CSV):

    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )


train_df = pd.read_csv(
    TRAIN_CSV
)


print(
    "\nTraining images available:",
    len(train_df)
)


# ============================================================
# GET IMAGES FOR EACH DEFICIENCY
# ============================================================

deficiency_images = {}

for label in [
    "iron_deficiency",
    "vitamin_b12_deficiency",
    "vitamin_d_deficiency"
]:

    deficiency_images[label] = train_df[
        train_df[label] == 1
    ]["image"].tolist()

    print(
        f"{label}:",
        len(deficiency_images[label])
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(filename):

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    if not os.path.exists(path):

        return None

    try:

        img = Image.open(path).convert("RGB")

        return img

    except Exception as error:

        print(
            "Could not load:",
            filename,
            error
        )

        return None


# ============================================================
# RESIZE
# ============================================================

def resize_image(img, size=(300, 300)):

    return img.resize(
        size,
        Image.Resampling.LANCZOS
    )


# ============================================================
# AUGMENT IMAGE
# ============================================================

def augment_image(img):

    # Random horizontal flip
    if random.random() < 0.5:

        img = img.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    # Small rotation
    angle = random.uniform(
        -8,
        8
    )

    img = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC
    )

    # Brightness
    brightness = random.uniform(
        0.85,
        1.15
    )

    img = ImageEnhance.Brightness(
        img
    ).enhance(
        brightness
    )

    # Contrast
    contrast = random.uniform(
        0.85,
        1.15
    )

    img = ImageEnhance.Contrast(
        img
    ).enhance(
        contrast
    )

    # Slight sharpness variation
    sharpness = random.uniform(
        0.85,
        1.15
    )

    img = ImageEnhance.Sharpness(
        img
    ).enhance(
        sharpness
    )

    return img


# ============================================================
# CREATE SYNTHETIC COMBINATION
# ============================================================

def create_combination_image(
    images,
    output_size=(300, 300)
):

    # Resize all images
    processed = []

    for img in images:

        img = resize_image(
            img,
            output_size
        )

        processed.append(
            np.asarray(
                img,
                dtype=np.float32
            )
        )

    # --------------------------------------------------------
    # Weighted visual combination
    # --------------------------------------------------------

    if len(processed) == 2:

        alpha = random.uniform(
            0.35,
            0.65
        )

        combined = (
            alpha * processed[0]
            +
            (1 - alpha) * processed[1]
        )

    else:

        weights = np.random.dirichlet(
            np.ones(len(processed))
        )

        combined = np.zeros_like(
            processed[0]
        )

        for weight, img in zip(
            weights,
            processed
        ):

            combined += (
                weight * img
            )

    # Clip pixel values

    combined = np.clip(
        combined,
        0,
        255
    )

    combined = combined.astype(
        np.uint8
    )

    result = Image.fromarray(
        combined
    )

    # Apply final augmentation

    result = augment_image(
        result
    )

    return result


# ============================================================
# GENERATE DATA
# ============================================================

synthetic_rows = []

for combination_name, config in COMBINATIONS.items():

    print("\n" + "-" * 60)

    print(
        "Generating:",
        combination_name
    )

    classes = config["classes"]

    labels = config["labels"]

    for index in range(
        IMAGES_PER_COMBINATION
    ):

        selected_images = []

        # ----------------------------------------------------
        # Select source image for each deficiency
        # ----------------------------------------------------

        for class_name in classes:

            available = deficiency_images[
                class_name
            ]

            filename = random.choice(
                available
            )

            img = load_image(
                filename
            )

            if img is None:

                continue

            selected_images.append(
                img
            )

        # ----------------------------------------------------
        # Make sure we got enough images
        # ----------------------------------------------------

        if len(selected_images) != len(classes):

            print(
                "Skipping sample:",
                combination_name,
                index
            )

            continue

        # ----------------------------------------------------
        # Generate image
        # ----------------------------------------------------

        synthetic_img = create_combination_image(
            selected_images
        )

        filename = (
            f"synthetic_"
            f"{combination_name}_"
            f"{index + 1:04d}.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        synthetic_img.save(
            output_path,
            quality=95
        )

        # ----------------------------------------------------
        # Save label
        # ----------------------------------------------------

        row = {
            "image": os.path.join(
                "synthetic_images",
                filename
            ),
            "healthy": labels[0],
            "iron_deficiency": labels[1],
            "vitamin_b12_deficiency": labels[2],
            "vitamin_d_deficiency": labels[3]
        }

        synthetic_rows.append(
            row
        )


# ============================================================
# SAVE SYNTHETIC LABELS
# ============================================================

synthetic_df = pd.DataFrame(
    synthetic_rows
)

synthetic_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SYNTHETIC DATA GENERATION COMPLETE")
print("=" * 60)

print(
    "\nSynthetic images created:",
    len(synthetic_df)
)

print(
    "\nDistribution:"
)

print(
    synthetic_df[
        LABEL_COLUMNS
    ].sum()
)

print(
    "\nImages saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nLabels saved to:"
)

print(
    OUTPUT_CSV
)