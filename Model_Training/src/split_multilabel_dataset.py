import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================
# PATHS
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset_multilabel"
)

CSV_FILE = os.path.join(
    DATASET_DIR,
    "labels.csv"
)

# ============================================
# SETTINGS
# ============================================

RANDOM_SEED = 42

VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# ============================================
# LOAD DATA
# ============================================

print("=" * 60)
print("SPLITTING MULTI-LABEL DATASET")
print("=" * 60)

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Could not find labels.csv:\n{CSV_FILE}"
    )

df = pd.read_csv(CSV_FILE)

print("\nTotal images:", len(df))

# ============================================
# CHECK COLUMNS
# ============================================

label_columns = [
    "healthy",
    "iron_deficiency",
    "vitamin_b12_deficiency",
    "vitamin_d_deficiency"
]

required_columns = [
    "image"
] + label_columns

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

# ============================================
# STRATIFICATION
# ============================================

df["stratify_label"] = df[label_columns].idxmax(axis=1)

print("\nOriginal distribution:")
print(df["stratify_label"].value_counts())

# ============================================
# TRAIN / TEMP
# ============================================

train_df, temp_df = train_test_split(
    df,
    test_size=VALIDATION_SIZE + TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["stratify_label"]
)

# ============================================
# VALIDATION / TEST
# ============================================

relative_test_size = TEST_SIZE / (
    VALIDATION_SIZE + TEST_SIZE
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=relative_test_size,
    random_state=RANDOM_SEED,
    stratify=temp_df["stratify_label"]
)

# ============================================
# REMOVE HELPER COLUMN
# ============================================

train_df = train_df.drop(
    columns=["stratify_label"]
)

validation_df = validation_df.drop(
    columns=["stratify_label"]
)

test_df = test_df.drop(
    columns=["stratify_label"]
)

# ============================================
# SAVE CSV FILES
# ============================================

train_path = os.path.join(
    DATASET_DIR,
    "train.csv"
)

validation_path = os.path.join(
    DATASET_DIR,
    "validation.csv"
)

test_path = os.path.join(
    DATASET_DIR,
    "test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

# ============================================
# RESULTS
# ============================================

print("\n" + "=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)

print("\nTraining:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))

print("\nTraining distribution:")
print(train_df[label_columns].sum())

print("\nValidation distribution:")
print(validation_df[label_columns].sum())

print("\nTest distribution:")
print(test_df[label_columns].sum())

print("\nCreated files:")

print(train_path)
print(validation_path)
print(test_path)

print("\nDone!")