from xai.utils.heart.clinical_mapping import CLINICAL_MEANING
from xai.utils.heart.clinical_thresholds import THRESHOLDS

def generate_doctor_explanation(explanation):
    risk = explanation["risk_probability"] * 100
    shap_vals = explanation["feature_contributions"]
    patient_vals = explanation["patient_values"]

    current_problems = []
    future_risks = []
    precautions = []
    protective_factors = []

    sorted_features = sorted(
        shap_vals.items(), key=lambda x: abs(x[1]), reverse=True
    )

    for feature, shap_val in sorted_features:
        if feature not in CLINICAL_MEANING:
            continue

        value = patient_vals.get(feature)

        if shap_val > 0:
            info = None

            if feature in THRESHOLDS:
                t = THRESHOLDS[feature]
                if "high" in t and value >= t["high"]:
                    info = CLINICAL_MEANING[feature].get("severe")
                elif "normal" in t and value >= t["normal"]:
                    info = CLINICAL_MEANING[feature].get("mild")

            if not info:
                info = CLINICAL_MEANING[feature].get("mild")

            if info:
                current_problems.append(info["present"])
                future_risks.append(info["future"])
                precautions.append(info["precaution"])

        elif shap_val < 0:
            protect = (
                CLINICAL_MEANING.get(feature, {})
                .get("protective", {})
                .get("present")
            )
            if protect:
                protective_factors.append(protect)

        if len(current_problems) >= 3:
            break

    risk_level = (
        "Low Risk" if risk < 30 else
        "Medium Risk" if risk < 70 else
        "High Risk"
    )

    return {
        "risk_level": risk_level,
        "risk_percentage": round(risk, 2),
        "current_problems": list(dict.fromkeys(current_problems)),
        "future_risks": list(dict.fromkeys(future_risks)),
        "precautions": list(dict.fromkeys(precautions)),
        "protective_factors": list(dict.fromkeys(protective_factors))
    }
