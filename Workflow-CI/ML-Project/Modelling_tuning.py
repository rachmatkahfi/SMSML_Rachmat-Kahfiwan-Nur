# modelling_tuning.py
# -*- coding: utf-8 -*-
"""
Modelling Tuning with XGBoost + MLflow + DagsHub
"""

import os
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import dagshub

# =========================
# 0. Setup DagsHub & MLflow
# =========================
os.environ["MLFLOW_TRACKING_USERNAME"] = "rachmatkahfi"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "02debb683ed2736ae818c4e2a9baaab574738ac6"

dagshub.init(
    repo_owner="rachmatkahfi",
    repo_name="SMSML_Rachmat_Kahfiwan_Nur",
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/rachmatkahfi/SMSML_Rachmat_Kahfiwan_Nur.mlflow")

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv(
    "C:/Users/ASUS/OneDrive/Documents/GitHub/SMSML_Rachmat-Kahfiwan-Nur/Workflow-CI/ML-Project/data_balita_preprocessing.csv"
)

print("📂 Columns in dataset:", df.columns.tolist())

X = df.drop("Status Gizi", axis=1)
y = df["Status Gizi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 2. Hyperparameter Tuning
# =========================
param_grid = {
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
    "n_estimators": [100, 200],
}

grid = GridSearchCV(
    xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)

# =========================
# 3. Logging ke MLflow
# =========================
with mlflow.start_run():
    grid.fit(X_train, y_train)

    # Ambil model & parameter terbaik
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    # Prediksi & evaluasi
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Logging params & metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)

    # =========================
    # 4. Simpan & Log Model (FIX)
    # =========================
    local_model_path = "xgb_best_model"
    mlflow.xgboost.save_model(best_model, local_model_path)
    mlflow.log_artifacts(local_model_path, artifact_path="model")

    # =========================
    # 5. Print hasil
    # =========================
    print("✅ Best Params:", best_params)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\n📊 Confusion Matrix:\n", cm)
    print("\n📑 Classification Report:\n", classification_report(y_test, preds))