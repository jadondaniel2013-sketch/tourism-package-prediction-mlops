# Import Path so all inputs and outputs are resolved relative to the project folder.
from pathlib import Path
# Import json so cleaning and split metadata can be saved for reproducibility.
import json
# Import pandas for tabular data loading, cleaning, grouping, and saving.
import pandas as pd
# Import train_test_split so the cleaned dataset can be divided reproducibly into train and test sets.
from sklearn.model_selection import train_test_split

# Resolve the master project directory from the location of this script.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Define the raw GitHub-managed dataset location.
RAW_PATH = PROJECT_DIR / "data" / "tourism.csv"
# Define the cleaned full dataset output location.
CLEAN_PATH = PROJECT_DIR / "data" / "cleaned_tourism.csv"
# Define the training split output location.
TRAIN_PATH = PROJECT_DIR / "data" / "train.csv"
# Define the testing split output location.
TEST_PATH = PROJECT_DIR / "data" / "test.csv"
# Define the metadata output location that documents every preparation choice.
PREP_METADATA_PATH = PROJECT_DIR / "data" / "preparation_metadata.json"

# Load the supplied tourism dataset from the project's data folder.
df = pd.read_csv(RAW_PATH)
# Record the raw shape before cleaning for an auditable before-and-after comparison.
raw_shape = df.shape
# Count exact duplicate rows before cleaning so duplicate removal is transparent.
duplicate_rows_before = int(df.duplicated().sum())
# Remove exact duplicate rows because duplicated observations can bias model learning and evaluation.
df = df.drop_duplicates().copy()
# Drop the exported row-index column because it is a technical artifact, not a customer attribute.
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
# Drop CustomerID because it is a unique identifier with no stable predictive meaning for new customers.
df = df.drop(columns=["CustomerID"], errors="ignore")
# Standardize the obvious gender spelling variant so one real category is not treated as two categories.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
# Standardize the occupation spelling so the same occupation is represented consistently.
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})
# Align the extra 'Unmarried' label with 'Single', which is the terminology used in the supplied data dictionary.
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
# Separate the target name into a variable so later code is less error-prone and easier to maintain.
TARGET = "ProdTaken"
# Verify that the target survived cleaning and remains present for supervised learning.
if TARGET not in df.columns:
    raise ValueError(f"Target column {TARGET!r} is missing after data preparation.")
# Split the cleaned data into training and testing sets using stratification to preserve the minority buyer rate.
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df[TARGET],
)
# Save the complete cleaned dataset for reproducibility and exploratory analysis.
df.to_csv(CLEAN_PATH, index=False)
# Save the training split locally so the model-training job can consume a fixed reproducible split.
train_df.to_csv(TRAIN_PATH, index=False)
# Save the testing split locally so final evaluation is performed on data not used for training.
test_df.to_csv(TEST_PATH, index=False)
# Build preparation metadata so the repository records exactly what was changed and how the split was made.
metadata = {
    "raw_shape": [int(raw_shape[0]), int(raw_shape[1])],
    "clean_shape": [int(df.shape[0]), int(df.shape[1])],
    "duplicate_rows_removed": duplicate_rows_before,
    "dropped_columns": ["Unnamed: 0", "CustomerID"],
    "category_standardization": {
        "Gender": {"Fe Male": "Female"},
        "Occupation": {"Free Lancer": "Freelancer"},
        "MaritalStatus": {"Unmarried": "Single"},
    },
    "train_rows": int(train_df.shape[0]),
    "test_rows": int(test_df.shape[0]),
    "train_target_distribution": {str(key): int(value) for key, value in train_df[TARGET].value_counts().sort_index().items()},
    "test_target_distribution": {str(key): int(value) for key, value in test_df[TARGET].value_counts().sort_index().items()},
    "random_state": 42,
    "test_size": 0.20,
    "stratified_on": TARGET,
}
# Save the preparation metadata beside the datasets for transparent version-controlled lineage.
PREP_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
# Print a concise cleaning summary for notebook output and GitHub Actions logs.
print(f"Raw shape: {raw_shape} | Clean shape: {df.shape}")
# Print the number of duplicates removed so reviewers can see whether this cleaning step changed the data.
print(f"Duplicate rows removed: {duplicate_rows_before}")
# Print the train and test shapes so the split can be checked quickly.
print(f"Training rows: {train_df.shape[0]:,} | Testing rows: {test_df.shape[0]:,}")
# Print the buyer rate in each split to verify successful stratification.
print(f"Train buyer rate: {train_df[TARGET].mean():.4f} | Test buyer rate: {test_df[TARGET].mean():.4f}")
# Print all output paths so the produced artifacts are easy to locate.
print("Saved:", CLEAN_PATH, TRAIN_PATH, TEST_PATH, sep="\n - ")
