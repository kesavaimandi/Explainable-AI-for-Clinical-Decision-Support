import shap
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parents[2]

model = joblib.load(BASE_DIR / "models" / "breast_cancer_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "breast_cancer_scaler.pkl")
feature_names = joblib.load(BASE_DIR / "models" / "breast_cancer_features.pkl")
data = pd.read_csv(BASE_DIR / "datasets" / "breast_cancer.csv")

X = data[feature_names]
background = scaler.transform(X.sample(100, random_state=42))
explainer = shap.KernelExplainer(model.predict_proba, background)


def explain_breast_cancer_prediction(patient_data):
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_scaled = scaler.transform(patient_df)
    risk_probability = model.predict_proba(patient_scaled)[0][1]
    shap_values = explainer.shap_values(patient_scaled)

    if isinstance(shap_values, list):
        shap_class_1 = shap_values[1][0]
    else:
        shap_class_1 = shap_values[0]

    feature_contributions = {}
    for i, feature in enumerate(feature_names):
        val = shap_class_1[i]
        if isinstance(val, (list, tuple, np.ndarray)):
            val = val[0]
        feature_contributions[feature] = float(val)

    return {
        "risk_probability": float(risk_probability),
        "feature_contributions": feature_contributions,
        "patient_values": patient_df.to_dict(orient="records")[0],
    }
