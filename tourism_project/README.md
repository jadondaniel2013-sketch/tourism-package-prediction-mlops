# Tourism Package Prediction - Full-Code MLOps Project

This repository implements the Tourism Package Prediction assignment using **GitHub**, **GitHub Actions**, **MLflow**, **GitHub Releases**, and **Streamlit Community Cloud**. Hugging Face is intentionally not used.

## End-to-end flow

1. `data/tourism.csv` is versioned in GitHub and validated by `data_register.py`.
2. `prep.py` cleans the data, removes non-predictive identifiers, standardizes inconsistent categories, and creates stratified train/test splits.
3. `train.py` tunes an XGBoost classifier with `GridSearchCV`, logs every parameter combination to MLflow in CI, evaluates the best model, saves feature importance, and serializes the complete preprocessing-plus-model pipeline.
4. GitHub Actions smoke-tests the model, commits deployment artifacts back to `main`, and registers each successful model as a versioned GitHub Release asset.
5. `deployment/app.py` is deployed from the same GitHub repository to Streamlit Community Cloud.

## Streamlit Community Cloud

Use Python **3.11**, branch **main**, and main file path:

`tourism_project/deployment/app.py`

After the first deployment, model/code commits to GitHub trigger Streamlit Community Cloud to rebuild the app automatically.
