import shap
import joblib
import pandas as pd
from pathlib import Path

# Base project directory (XAI Model/)
BASE_DIR = Path(__file__).resolve().parents[2]

model = joblib.load(BASE_DIR / "models" / "heart_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "heart_scaler.pkl")
feature_names = joblib.load(BASE_DIR / "models" / "heart_features.pkl")
data = pd.read_csv(BASE_DIR / "datasets" / "heart.csv")

X = data.drop("target", axis=1)
explainer = shap.Explainer(model, X)

def explain_heart_prediction(patient_data):
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_scaled = scaler.transform(patient_df)

    shap_values = explainer(patient_scaled)

    return {
        "risk_probability": float(model.predict_proba(patient_scaled)[0][1]),
        "feature_contributions": {
            feature_names[i]: float(shap_values.values[0][i])
            for i in range(len(feature_names))
        },
        "patient_values": patient_df.to_dict(orient="records")[0]
    }
