from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
from prometheus_client import start_http_server, Counter, Histogram
import numpy as np
import time

# 1️⃣ Konfigurasi model dari MLflow
MODEL_URI = "runs:/c239fe5d79f44ffe95db4322e170235b/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

# 2️⃣ Buat FastAPI app
app = FastAPI(title="XGBoost Model API with Prometheus Metrics")

# 3️⃣ Definisikan metrik Prometheus
REQUEST_COUNT = Counter("http_request_total", "Jumlah request prediksi")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Waktu prediksi model (detik)")

# 4️⃣ Struktur input data
class InputData(BaseModel):
    features: list[float]  # pastikan list float agar schema jelas di Swagger

# 5️⃣ Endpoint prediksi
@app.post("/predict")
def predict(data: InputData):
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        # Pastikan input berbentuk numpy array 2D
        X = np.array(data.features, dtype=float).reshape(1, -1)

        # Jalankan prediksi
        prediction = model.predict(X)

        # Ukur waktu prediksi
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        # Ambil nilai prediksi tunggal (bukan list bertingkat)
        result = float(prediction[0]) if hasattr(prediction, "__len__") else float(prediction)

        # Tambahkan label teks opsional (mapping status gizi)
        label_map = {
            0.0: "Gizi Buruk",
            1.0: "Gizi Kurang",
            2.0: "Gizi Baik",
            3.0: "Gizi Lebih"
        }

        label = label_map.get(result, "Tidak Dikenal")

        return {
            "prediction_value": result,
            "prediction_label": label,
            "latency_seconds": round(latency, 4)
        }

    except Exception as e:
        # Tangkap error dan tampilkan di Swagger response
        return {"error": str(e)}

# 6️⃣ Jalankan Prometheus metrics server (port 8001)
start_http_server(8001)
