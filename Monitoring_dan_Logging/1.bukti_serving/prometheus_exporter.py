import joblib
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

# Load model
model = joblib.load('model.pkl')

# Sesuaikan nama kolom ini dengan data training Anda
MODEL_COLUMNS = ['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']

# Buat aplikasi Flask
app = Flask(__name__)
# Tambahkan monitoring Prometheus
metrics = PrometheusMetrics(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        input_data = pd.DataFrame([json_data], columns=MODEL_COLUMNS)
        
        prediction = model.predict(input_data)
        
        # Asumsi 0 = Gizi Baik, 1 = Gizi Kurang, 2 = Stunting, 3 = Gizi Lebih
        status_gizi_map = {0: 'Gizi Baik', 1: 'Gizi Kurang', 2: 'Stunting', 3: 'Gizi Lebih'}
        predicted_status = status_gizi_map.get(prediction[0], 'Unknown')

        return jsonify({'prediction': predicted_status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)