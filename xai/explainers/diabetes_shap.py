import shap
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parents[2]

model = joblib.load(BASE_DIR / "models" / "diabetes_model.pkl")
feature_names = joblib.load(BASE_DIR / "models" / "diabetes_features.pkl")
data = pd.read_csv(BASE_DIR / "datasets" / "diabetes.csv")

X = data.drop("Outcome", axis=1)
explainer = shap.Explainer(model, X)

def explain_diabetes_prediction(patient_data):
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    risk_probability = model.predict_proba(patient_df)[0][1]
    shap_values = explainer(patient_df)
    shap_class_1 = shap_values.values[0, :, 1]
    feature_contributions = {
        feature_names[i]: float(shap_class_1[i])
        for i in range(len(feature_names))
    }

    return {
        "risk_probability": float(risk_probability),
        "feature_contributions": feature_contributions,
        "patient_values": patient_df.to_dict(orient="records")[0]
    }
