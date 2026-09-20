# ==================== 1. Imports ====================
# Imports
from pathlib import Path
import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project configuration
warnings.filterwarnings("ignore")
PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_DIR / "data" / "train.csv"
TEST_PATH = PROJECT_DIR / "data" / "test.csv"
MODEL_PATH = PROJECT_DIR / "deployment" / "best_tourism_model.joblib"
METRICS_PATH = PROJECT_DIR / "model_building" / "model_metrics.json"
EXPERIMENT_TABLE_PATH = PROJECT_DIR / "model_building" / "experiment_results.csv"
IMPORTANCE_PATH = PROJECT_DIR / "model_building" / "feature_importance.csv"
TARGET = "ProdTaken"

# Load prepared data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_features = [column for column in X_train.columns if column not in categorical_features]
# Preprocessing and model
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)
# Compute class-weight ratio
class_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
classifier = XGBClassifier(
    random_state=42,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=1,
    tree_method="hist",
)
model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
# Hyperparameter tuning
# Tune key XGBoost parameters
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.10],
    "classifier__subsample": [0.80],
    "classifier__colsample_bytree": [0.80],
    "classifier__scale_pos_weight": [1.0, round(class_ratio, 2)],
}
# Use 3-fold ROC-AUC grid search
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=1,
    return_train_score=True,
)
grid_search.fit(X_train, y_train)
experiment_results = pd.DataFrame(grid_search.cv_results_)
experiment_results = experiment_results[["params", "mean_test_score", "std_test_score", "rank_test_score", "mean_train_score"]]
experiment_results.to_csv(EXPERIMENT_TABLE_PATH, index=False)
# MLflow tracking
# Use MLflow when a tracking URI is provided
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
try:
    import mlflow
    mlflow_available = True
except ImportError:
    mlflow_available = False
# Log tuning runs when MLflow is available
if mlflow_available and mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("tourism-package-prediction")
    with mlflow.start_run(run_name="xgboost-grid-search"):
        results = grid_search.cv_results_
        for index, parameter_set in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"trial-{index + 1}", nested=True):
                mlflow.log_params(parameter_set)
                mlflow.log_metric("mean_cv_roc_auc", float(results["mean_test_score"][index]))
        mlflow.log_params({f"best_{key}": value for key, value in grid_search.best_params_.items()})
        mlflow.log_metric("best_cv_roc_auc", float(grid_search.best_score_))
# Model evaluation
best_model = grid_search.best_estimator_
train_predictions = best_model.predict(X_train)
train_probabilities = best_model.predict_proba(X_train)[:, 1]
test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]
# Metric helper
def calculate_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).astype(int).tolist(),
    }
train_metrics = calculate_metrics(y_train, train_predictions, train_probabilities)
test_metrics = calculate_metrics(y_test, test_predictions, test_probabilities)
# Model interpretation
importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=1,
)
importance_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }
).sort_values("importance_mean", ascending=False)
importance_df.to_csv(IMPORTANCE_PATH, index=False)
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
METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
if mlflow_available and mlflow_tracking_uri:
    with mlflow.start_run(run_name="best-model-summary"):
        mlflow.log_params({key: value for key, value in grid_search.best_params_.items()})
        mlflow.log_metrics({key: value for key, value in test_metrics.items() if key != "confusion_matrix"})
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="evaluation")
        mlflow.log_artifact(str(IMPORTANCE_PATH), artifact_path="evaluation")
print("Best parameters:", grid_search.best_params_)
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
print("Training metrics:", train_metrics)
print("Testing metrics:", test_metrics)
print("Top features by permutation importance:")
print(importance_df.head(10).to_string(index=False))
print("Saved deployable model to:", MODEL_PATH)
print("Saved tuning log to:", EXPERIMENT_TABLE_PATH)
if not (mlflow_available and mlflow_tracking_uri):
    print("MLflow server logging was skipped in this offline runtime; all trials were saved to experiment_results.csv. GitHub Actions installs MLflow and logs the same trials automatically.")
