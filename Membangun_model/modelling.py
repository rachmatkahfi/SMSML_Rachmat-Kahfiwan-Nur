import os
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import dagshub

# =========================
# 0. Setup DagsHub & MLflow
# =========================
# Simpan token dagshub kamu di sini (atau bisa pakai environment variable sistem)
os.environ["MLFLOW_TRACKING_USERNAME"] = "rachmatkahfi"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "02debb683ed2736ae818c4e2a9baaab574738ac6"

dagshub.init(repo_owner="rachmatkahfi", repo_name="Eksperimen_SML_Rachmat_Kahfiwan_Nur", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/rachmatkahfi/Eksperimen_SML_Rachmat_Kahfiwan_Nur.mlflow")
# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("C:/Users/ASUS/OneDrive/Documents/GitHub/Membangun_model/data_balita_preprocessing.csv")

# Cek kolom tersedia
print("Columns in dataset:", df.columns.tolist())

# Tentukan target (asumsi 'Status Gizi')
X = df.drop("Status Gizi", axis=1)
y = df["Status Gizi"]

# =========================
# 2. Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. Aktifkan Autolog MLflow
# =========================
mlflow.xgboost.autolog()

# =========================
# 4. Training Model
# =========================
with mlflow.start_run():
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # =========================
    # 5. Evaluasi Model
    # =========================
    preds = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    print("\n📊 Confusion Matrix:")
    print(cm)

    # Classification Report
    print("\n📑 Classification Report:")
    print(classification_report(y_test, preds))