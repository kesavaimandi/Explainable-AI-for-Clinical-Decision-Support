from xai.utils.diabetes.clinical_mapping import CLINICAL_MEANING

def generate_diabetes_doctor_explanation(explanation):
    risk = explanation["risk_probability"] * 100
    shap_vals = explanation["feature_contributions"]
    patient_vals = explanation["patient_values"]

    current_problems = []
    future_risks = []
    precautions = []
    protective_factors = []

    sorted_features = sorted(
        shap_vals.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_features = sorted_features[:4]

    for feature, shap_val in top_features:
        if feature not in CLINICAL_MEANING:
            continue
        meaning = CLINICAL_MEANING[feature]
        
        if shap_val > 0:
            if abs(shap_val) > 0.15 and "severe" in meaning:
                info = meaning["severe"]
            else:
                info = meaning.get("mild")

            if info:
                current_problems.append(info["present"])
                future_risks.append(info["future"])
                precautions.append(info["precaution"])

        elif shap_val < 0:
            protect = meaning.get("protective")
            if protect:
                protective_factors.append(protect["present"])

    if risk < 30:
        risk_level = "Low Risk"
    elif risk < 70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    if not current_problems:
        current_problems.append(
            "No dominant diabetic abnormalities detected by the model"
        )

    if not future_risks:
        future_risks.append(
            "Future diabetes risk depends on long-term metabolic control"
        )

    if not precautions:
        precautions.extend([
            "Maintain balanced diet and regular physical activity",
            "Routine blood glucose monitoring recommended"
        ])

    if not protective_factors:
        protective_factors.append(
            "Protective metabolic indicators are present"
        )

    return {
        "risk_level": risk_level,
        "risk_percentage": round(risk, 2),
        "current_problems": list(dict.fromkeys(current_problems)),
        "future_risks": list(dict.fromkeys(future_risks)),
        "precautions": list(dict.fromkeys(precautions)),
        "protective_factors": list(dict.fromkeys(protective_factors)),
    }
