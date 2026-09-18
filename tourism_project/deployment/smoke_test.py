# Import Path so the smoke test can locate the committed model reliably.
from pathlib import Path
# Import joblib so the serialized deployment model can be loaded without starting Streamlit.
import joblib
# Import pandas so a realistic single-customer input row can be created for a deployment check.
import pandas as pd

# Resolve the deployment folder that contains this smoke test and the model artifact.
DEPLOYMENT_DIR = Path(__file__).resolve().parent
# Load the serialized preprocessing-plus-classifier pipeline created by model training.
model = joblib.load(DEPLOYMENT_DIR / "best_tourism_model.joblib")
# Create one representative customer using only valid categories and numeric ranges from the training data.
sample = pd.DataFrame([
    {
        "Age": 35.0,
        "TypeofContact": "Company Invited",
        "CityTier": 3,
        "DurationOfPitch": 18.0,
        "Occupation": "Salaried",
        "Gender": "Male",
        "NumberOfPersonVisiting": 2,
        "NumberOfFollowups": 4.0,
        "ProductPitched": "Basic",
        "PreferredPropertyStar": 3.0,
        "MaritalStatus": "Single",
        "NumberOfTrips": 3.0,
        "Passport": 1,
        "PitchSatisfactionScore": 4,
        "OwnCar": 1,
        "NumberOfChildrenVisiting": 1.0,
        "Designation": "Executive",
        "MonthlyIncome": 22000.0,
    }
])
# Generate a purchase probability to verify that preprocessing and model inference work end to end.
probability = float(model.predict_proba(sample)[0, 1])
# Fail the smoke test if the model returns a value outside the valid probability range.
assert 0.0 <= probability <= 1.0, "Model returned an invalid probability."
# Print the probability so the CI/CD log visibly confirms successful inference.
print(f"Deployment smoke test passed. Sample purchase probability: {probability:.4f}")
