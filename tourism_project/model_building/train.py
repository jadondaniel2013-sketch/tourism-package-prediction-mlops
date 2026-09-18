# Import Path so all model artifacts are written to stable project-relative locations.
from pathlib import Path
# Import json so final metrics and best parameters can be stored in a machine-readable artifact.
import json
# Import os so the training script can read an optional MLflow tracking URI from the CI environment.
import os
# Import warnings so non-critical library warnings can be suppressed in automated logs.
import warnings
# Import joblib so the full preprocessing-plus-model pipeline can be serialized for deployment.
import joblib
# Import numpy for safe conversion of NumPy scalar values before JSON serialization.
import numpy as np
# Import pandas for loading the prepared train/test CSV files and saving experiment tables.
import pandas as pd
# Import XGBoost's classifier because the rubric permits XGBoost and it performs well on mixed tabular data.
from xgboost import XGBClassifier
# Import ColumnTransformer so numeric and categorical features can be preprocessed in one reproducible pipeline.
from sklearn.compose import ColumnTransformer
# Import SimpleImputer so the pipeline remains robust if future data contains missing values.
from sklearn.impute import SimpleImputer
# Import permutation_importance so business-facing feature influence can be assessed on held-out data.
from sklearn.inspection import permutation_importance
# Import evaluation metrics suitable for an imbalanced binary classification problem.
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
# Import GridSearchCV so defined hyperparameters can be tuned with cross-validation rather than chosen manually.
from sklearn.model_selection import GridSearchCV
# Import Pipeline so preprocessing and model estimation are always applied together consistently.
from sklearn.pipeline import Pipeline
# Import OneHotEncoder so categorical customer attributes can be represented numerically without imposing false order.
from sklearn.preprocessing import OneHotEncoder

# Suppress non-critical warnings so CI/CD logs focus on actionable information.
warnings.filterwarnings("ignore")
# Resolve the master project directory relative to this training script.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Define the prepared training data path created by prep.py.
TRAIN_PATH = PROJECT_DIR / "data" / "train.csv"
# Define the prepared testing data path created by prep.py.
TEST_PATH = PROJECT_DIR / "data" / "test.csv"
# Define where the deployable serialized model will be stored.
MODEL_PATH = PROJECT_DIR / "deployment" / "best_tourism_model.joblib"
# Define where final model metrics and selected parameters will be stored.
METRICS_PATH = PROJECT_DIR / "model_building" / "model_metrics.json"
# Define where every grid-search trial will be stored even if MLflow is unavailable locally.
EXPERIMENT_TABLE_PATH = PROJECT_DIR / "model_building" / "experiment_results.csv"
# Define where held-out permutation importances will be stored for interpretation.
IMPORTANCE_PATH = PROJECT_DIR / "model_building" / "feature_importance.csv"
# Define the supervised-learning target column.
TARGET = "ProdTaken"

# Load the fixed training split prepared by the previous pipeline stage.
train_df = pd.read_csv(TRAIN_PATH)
# Load the untouched testing split prepared by the previous pipeline stage.
test_df = pd.read_csv(TEST_PATH)
# Separate training predictors from the target.
X_train = train_df.drop(columns=[TARGET])
# Separate the training target from its predictors.
y_train = train_df[TARGET]
# Separate testing predictors from the target.
X_test = test_df.drop(columns=[TARGET])
# Separate the testing target from its predictors.
y_test = test_df[TARGET]
# Identify categorical columns from their pandas dtype so the pipeline adapts cleanly to the dataset schema.
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
# Identify numeric columns as every predictor that is not categorical.
numeric_features = [column for column in X_train.columns if column not in categorical_features]
# Build the numeric preprocessing branch using median imputation for robustness to future missing values.
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
# Build the categorical preprocessing branch with robust imputation followed by one-hot encoding.
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# Combine numeric and categorical preprocessing into one reusable transformer.
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)
# Calculate the negative-to-positive ratio so the search can test class balancing for the minority buyer class.
class_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
# Define the base XGBoost classifier with a fixed random state for reproducible results.
classifier = XGBClassifier(
    random_state=42,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=1,
    tree_method="hist",
)
# Combine preprocessing and classification into a single deployable pipeline that accepts raw customer fields.
model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
# Define a compact but meaningful hyperparameter grid suitable for both the assignment and GitHub Actions runtime.
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.10],
    "classifier__subsample": [0.80],
    "classifier__colsample_bytree": [0.80],
    "classifier__scale_pos_weight": [1.0, round(class_ratio, 2)],
}
# Configure three-fold cross-validated grid search using ROC-AUC, which evaluates ranking quality under class imbalance.
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=1,
    return_train_score=True,
)
# Fit every defined hyperparameter combination using only the training split.
grid_search.fit(X_train, y_train)
# Convert all cross-validation trial results into a DataFrame so every tuned parameter set is auditable.
experiment_results = pd.DataFrame(grid_search.cv_results_)
# Select the columns needed to review each parameter set, validation score, and score variability.
experiment_results = experiment_results[["params", "mean_test_score", "std_test_score", "rank_test_score", "mean_train_score"]]
# Save all tuning trials locally even when an MLflow server is not available in the current runtime.
experiment_results.to_csv(EXPERIMENT_TABLE_PATH, index=False)
# Read an optional MLflow tracking URI supplied by GitHub Actions; leave it blank during fully offline local execution.
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
# Try to use MLflow when it is installed, as it will be in the GitHub Actions environment from requirements.txt.
try:
    # Import MLflow only inside the guarded block so the notebook remains executable in a restricted offline runtime.
    import mlflow
    # Mark MLflow as available after the import succeeds.
    mlflow_available = True
except ImportError:
    # Mark MLflow as unavailable when this offline runtime does not contain the package.
    mlflow_available = False
# Continue with MLflow logging only when both the package and a tracking URI are available.
if mlflow_available and mlflow_tracking_uri:
    # Configure the tracking endpoint started by the GitHub Actions workflow.
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # Group all tourism model runs inside one named MLflow experiment.
    mlflow.set_experiment("tourism-package-prediction")
    # Start one parent run for the complete hyperparameter-search execution.
    with mlflow.start_run(run_name="xgboost-grid-search"):
        # Read the fitted grid-search results so every tested parameter combination can be logged.
        results = grid_search.cv_results_
        # Loop through every tuned parameter combination and its mean validation score.
        for index, parameter_set in enumerate(results["params"]):
            # Start a nested run so every trial has its own complete parameter record.
            with mlflow.start_run(run_name=f"trial-{index + 1}", nested=True):
                # Log every hyperparameter value tested in this trial.
                mlflow.log_params(parameter_set)
                # Log the mean cross-validated ROC-AUC achieved by this trial.
                mlflow.log_metric("mean_cv_roc_auc", float(results["mean_test_score"][index]))
        # Log the best hyperparameter combination again on the parent run for quick retrieval.
        mlflow.log_params({f"best_{key}": value for key, value in grid_search.best_params_.items()})
        # Log the best cross-validated ROC-AUC on the parent run.
        mlflow.log_metric("best_cv_roc_auc", float(grid_search.best_score_))
# Select the cross-validated best full preprocessing-plus-classifier pipeline.
best_model = grid_search.best_estimator_
# Generate training predictions so overfitting can be assessed by comparing train and test performance.
train_predictions = best_model.predict(X_train)
# Generate training purchase probabilities for ROC-AUC calculation.
train_probabilities = best_model.predict_proba(X_train)[:, 1]
# Generate held-out test predictions for unbiased final evaluation.
test_predictions = best_model.predict(X_test)
# Generate held-out test purchase probabilities for ranking quality and deployment use.
test_probabilities = best_model.predict_proba(X_test)[:, 1]
# Define a helper that calculates the same classification metrics consistently for any dataset split.
def calculate_metrics(y_true, predictions, probabilities):
    # Return metrics emphasizing both overall performance and minority-class quality.
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).astype(int).tolist(),
    }
# Calculate the training metrics for diagnostic comparison with the test metrics.
train_metrics = calculate_metrics(y_train, train_predictions, train_probabilities)
# Calculate the held-out testing metrics used for the final model assessment.
test_metrics = calculate_metrics(y_test, test_predictions, test_probabilities)
# Calculate permutation importance on the held-out test set to estimate business-level influence of original features.
importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=1,
)
# Convert the raw importance arrays into a readable feature-level table.
importance_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }
).sort_values("importance_mean", ascending=False)
# Save the feature-importance table so it can be reviewed independently of the notebook.
importance_df.to_csv(IMPORTANCE_PATH, index=False)
# Build the final metrics artifact with the best parameters, class ratio, and both train/test metrics.
metrics_payload = {
    "model": "XGBClassifier",
    "selection_metric": "roc_auc",
    "best_cv_roc_auc": float(grid_search.best_score_),
    "class_ratio_negative_to_positive": class_ratio,
    "best_params": {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in grid_search.best_params_.items()
    },
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
    "top_features_by_permutation_importance": importance_df.head(10).to_dict(orient="records"),
}
# Save the evaluation artifact as formatted JSON for GitHub review and CI/CD traceability.
METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
# Ensure the deployment directory exists because model training occurs before the Deployment section in the learner template.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
# Serialize the entire fitted preprocessing-plus-model pipeline so Streamlit can score raw user inputs directly.
joblib.dump(best_model, MODEL_PATH)
# If MLflow is available in GitHub Actions, log final metrics and model artifacts to the parent experiment run.
if mlflow_available and mlflow_tracking_uri:
    # Reopen the most recent parent run context by creating a final summary run for durable metrics and artifacts.
    with mlflow.start_run(run_name="best-model-summary"):
        # Log the best parameters without the pipeline prefix for a cleaner MLflow UI.
        mlflow.log_params({key: value for key, value in grid_search.best_params_.items()})
        # Log each held-out test metric except the confusion matrix, which is not a scalar.
        mlflow.log_metrics({key: value for key, value in test_metrics.items() if key != "confusion_matrix"})
        # Log the serialized deployable model as an experiment artifact.
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        # Log the JSON metrics artifact so run metadata and evaluation details stay together.
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="evaluation")
        # Log the feature-importance table for model interpretation.
        mlflow.log_artifact(str(IMPORTANCE_PATH), artifact_path="evaluation")
# Print the selected parameters so they are visible in notebook and GitHub Actions logs.
print("Best parameters:", grid_search.best_params_)
# Print the best cross-validated ROC-AUC achieved during tuning.
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
# Print the training metrics so any generalization gap can be diagnosed.
print("Training metrics:", train_metrics)
# Print the held-out testing metrics used for final evaluation.
print("Testing metrics:", test_metrics)
# Print the top ten held-out permutation importances for business interpretation.
print("Top features by permutation importance:")
# Print the top feature table without the pandas row index for clean logs.
print(importance_df.head(10).to_string(index=False))
# Print the deployable model location so reviewers can verify model registration in the GitHub project.
print("Saved deployable model to:", MODEL_PATH)
# Print the experiment-table location so all tuned parameter combinations can be audited.
print("Saved tuning log to:", EXPERIMENT_TABLE_PATH)
# Explain the local fallback when MLflow cannot run in a restricted offline environment.
if not (mlflow_available and mlflow_tracking_uri):
    # Print a transparent note rather than failing the notebook when MLflow is unavailable locally.
    print("MLflow server logging was skipped in this offline runtime; all trials were saved to experiment_results.csv. GitHub Actions installs MLflow and logs the same trials automatically.")
