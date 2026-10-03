import os
import shutil
import csv

# ============================================
# Paths
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "dataset_single_label", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_multilabel")

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_FILE = os.path.join(OUTPUT_DIR, "labels.csv")

# ============================================
# Class Mapping
# ============================================

CLASS_NAMES = [
    "healthy",
    "iron_deficiency",
    "vitamin_b12_deficiency",
    "vitamin_d_deficiency"
]

SOURCE_CLASSES = {
    "healthy_nails": "healthy",
    "iron_deficiency": "iron_deficiency",
    "vitamin_b12_deficiency": "vitamin_b12_deficiency",
    "vitamin_d_deficiency": "vitamin_d_deficiency"
}

# ============================================
# Create Output Directory
# ============================================

os.makedirs(IMAGE_DIR, exist_ok=True)

print("=" * 60)
print("CREATING MULTI-LABEL DATASET TEMPLATE")
print("=" * 60)

rows = []

# ============================================
# Read Existing Dataset
# ============================================

for source_folder, label_name in SOURCE_CLASSES.items():

    source_path = os.path.join(
        SOURCE_DIR,
        source_folder
    )

    if not os.path.exists(source_path):
        print("Missing folder:", source_path)
        continue

    files = os.listdir(source_path)

    print(
        f"\n{source_folder}: {len(files)} images"
    )

    for filename in files:

        source_file = os.path.join(
            source_path,
            filename
        )

        if not os.path.isfile(source_file):
            continue

        # Create unique filename
        new_filename = (
            label_name + "_" + filename
        )

        destination_file = os.path.join(
            IMAGE_DIR,
            new_filename
        )

        shutil.copy2(
            source_file,
            destination_file
        )

        # ====================================
        # Create initial single-label record
        # ====================================

        row = {
            "image": new_filename,
            "healthy": 0,
            "iron_deficiency": 0,
            "vitamin_b12_deficiency": 0,
            "vitamin_d_deficiency": 0
        }

        # Preserve the original dataset label
        row[label_name] = 1

        rows.append(row)

# ============================================
# Write CSV
# ============================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "image",
        "healthy",
        "iron_deficiency",
        "vitamin_b12_deficiency",
        "vitamin_d_deficiency"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(rows)

# ============================================
# Finished
# ============================================

print("\n" + "=" * 60)
print("MULTI-LABEL DATASET TEMPLATE CREATED")
print("=" * 60)

print("\nImages:", len(rows))
print("Image folder:", IMAGE_DIR)
print("CSV file:", CSV_FILE)

print("\nIMPORTANT:")
print("The existing labels are single-label annotations.")
print("Do NOT add additional disease labels unless")
print("you have valid evidence for those labels.")

print("\nNext step: review and annotate labels.csv")