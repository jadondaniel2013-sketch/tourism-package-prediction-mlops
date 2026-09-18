# Import Path so file locations are resolved reliably from the script location.
from pathlib import Path
# Import hashlib so the raw data file can be fingerprinted for reproducibility.
import hashlib
# Import json so dataset registration metadata can be stored in a readable manifest.
import json
# Import pandas so the CSV file can be loaded and validated.
import pandas as pd

# Resolve the master project folder relative to this script.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Define the registered raw-data location inside the GitHub repository.
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
# Define the manifest file used to record schema, row counts, target distribution, and checksum.
MANIFEST_PATH = PROJECT_DIR / "data" / "data_manifest.json"

# Define every column expected from the supplied project data dictionary.
EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

# Stop early with a clear message if the raw dataset is not present in the GitHub project folder.
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Expected raw dataset was not found at: {DATA_PATH}")

# Load the raw tourism dataset from the GitHub-managed data folder.
df = pd.read_csv(DATA_PATH)
# Identify any columns required by the project specification but absent from the uploaded data.
missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
# Fail registration if the schema does not match the expected project schema.
if missing_columns:
    raise ValueError(f"Dataset registration failed; missing columns: {missing_columns}")
# Verify that the target is a binary classification target containing only 0 and 1.
if set(df["ProdTaken"].dropna().unique()) - {0, 1}:
    raise ValueError("ProdTaken must contain only the binary labels 0 and 1.")
# Compute a SHA-256 checksum so future data changes can be detected and audited.
sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
# Build a compact registration record that can be version-controlled with the dataset in GitHub.
manifest = {
    "file": str(DATA_PATH.relative_to(PROJECT_DIR)),
    "sha256": sha256,
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "column_names": df.columns.tolist(),
    "target_distribution": {str(key): int(value) for key, value in df["ProdTaken"].value_counts().sort_index().items()},
    "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
}
# Save the registration manifest in JSON format for GitHub version history and CI/CD traceability.
MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
# Print a clear success message for notebook output and GitHub Actions logs.
print("Dataset registration completed successfully.")
# Print the dataset shape so the registered data volume is visible in execution logs.
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
# Print the class distribution so class imbalance is visible before modeling.
print("Target distribution:", df["ProdTaken"].value_counts().sort_index().to_dict())
# Print a shortened checksum so a data version can be recognized quickly in logs.
print("SHA-256:", sha256[:16] + "...")
# Print the manifest path so reviewers can find the registration metadata in the repository.
print("Manifest saved to:", MANIFEST_PATH)
