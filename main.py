from fastapi import FastAPI
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model & scaler
model = load_model("/Users/dipteesomeshwarghusse/Documents/Claude/model/lstm_model.h5",compile=False)
scaler = joblib.load("/Users/dipteesomeshwarghusse/Documents/Claude/model/scaler.pkl")

@app.get("/")
def home():
    return {"message": "LSTM Stock Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        input_data = np.array(data["data"]).reshape(1, 60, 1)

        # Predict
        prediction = model.predict(input_data)

        # Inverse scale
        prediction = scaler.inverse_transform(prediction)

        return {
            "prediction": float(prediction[0][0])
        }

    except Exception as e:
        return {"error": str(e)}
