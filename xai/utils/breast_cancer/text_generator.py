from xai.utils.breast_cancer.clinical_mapping import CLINICAL_MEANING
def generate_breast_doctor_explanation(explanation):
    risk = explanation["risk_probability"] * 100
    shap_vals = explanation["feature_contributions"]
    sorted_features = sorted(
        shap_vals.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    current_problems = []
    future_risks = []
    precautions = []
    protective_factors = []

    malignant_indicators = []
    benign_indicators = []

    for feature, shap_val in sorted_features[:8]:

        if shap_val > 0:
            malignant_indicators.append(feature)

        elif shap_val < 0:
            benign_indicators.append(feature)

    if risk >= 70:
        current_problems.append(
            "Strong malignant tumor characteristics detected based on tissue morphology"
        )

        if any(f in malignant_indicators for f in [
            "radius_mean", "perimeter_mean", "area_mean",
            "radius_worst", "perimeter_worst", "area_worst"
        ]):
            current_problems.append(
                "Large and irregular tumor structure suggesting aggressive malignancy"
            )

        if any(f in malignant_indicators for f in [
            "concavity_mean", "concave_points_mean",
            "concavity_worst", "concave_points_worst"
        ]):
            current_problems.append(
                "Highly irregular tumor boundaries associated with invasive cancer"
            )

    elif risk >= 30:
        current_problems.append(
            "Mixed tumor characteristics observed requiring careful clinical evaluation"
        )

        if malignant_indicators:
            current_problems.append(
                "Some malignant tissue indicators are present"
            )

    else:
        current_problems.append(
            "No dominant malignant tumor characteristics detected"
        )

        if benign_indicators:
            current_problems.append(
                "Tumor features largely consistent with benign tissue patterns"
            )

    if risk >= 70:
        future_risks.extend([
            "High probability of aggressive breast cancer progression",
            "Increased likelihood of local tissue invasion",
            "Risk of metastasis if untreated"
        ])

    elif risk >= 30:
        future_risks.extend([
            "Potential progression toward malignant behavior",
            "Risk of tumor growth over time"
        ])

    else:
        future_risks.append(
            "Low probability of aggressive breast cancer progression"
        )


    if risk >= 70:
        precautions.extend([
            "Immediate oncological consultation recommended",
            "Histopathological confirmation advised",
            "Treatment planning including surgery or systemic therapy"
        ])

    elif risk >= 30:
        precautions.extend([
            "Close clinical monitoring advised",
            "Follow-up imaging and biopsy if necessary",
            "Regular oncologist consultation recommended"
        ])

    else:
        precautions.extend([
            "Routine breast cancer screening recommended",
            "Maintain regular follow-up with healthcare provider"
        ])

    if benign_indicators:
        protective_factors.append(
            "Presence of tissue characteristics associated with benign tumors"
        )

    if risk < 30:
        protective_factors.append(
            "Overall tumor profile suggests low malignancy risk"
        )

    if not protective_factors:
        protective_factors.append(
            "No strong protective tissue indicators identified"
        )

    if risk < 30:
        risk_level = "Low Risk"
    elif risk < 70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "risk_level": risk_level,
        "risk_percentage": round(risk, 2),
        "current_problems": list(dict.fromkeys(current_problems)),
        "future_risks": list(dict.fromkeys(future_risks)),
        "precautions": list(dict.fromkeys(precautions)),
        "protective_factors": list(dict.fromkeys(protective_factors))
    }
