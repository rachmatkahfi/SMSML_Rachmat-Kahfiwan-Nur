from fastapi import FastAPI, Request
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# Load model hasil training (sesuaikan path dengan model kriteria 3)
model = joblib.load("C:/Users/ASUS/OneDrive/Documents/GitHub/SMSML_Rachmat-Kahfiwan-Nur/Workflow-CI/ML-Project/model.pkl")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features).tolist()
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
