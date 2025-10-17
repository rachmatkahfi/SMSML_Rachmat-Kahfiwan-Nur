# modelling_tuning.py
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==================================================
# 1️⃣ Konfigurasi MLflow Tracking
# ==================================================
mlflow.set_tracking_uri("file:///C:/Users/ASUS/OneDrive/Documents/GitHub/SMSML_Rachmat-Kahfiwan-Nur/mlruns")
mlflow.set_experiment("Machine Learning - Status Gizi")

# ==================================================
# 2️⃣ Load Dataset
# ==================================================
df = pd.read_csv("data_balita_preprocessing.csv")
X = df.drop("Status Gizi", axis=1)
y = df["Status Gizi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================================================
# 3️⃣ Hyperparameter Tuning
# ==================================================
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

# ==================================================
# 4️⃣ Jalankan MLflow Run
# ==================================================
with mlflow.start_run() as run:
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Logging ke MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.xgboost.log_model(best_model, "model")

    print("Best Params:", best_params)
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print(f"\nModel saved to MLflow with Run ID: {run.info.run_id}")