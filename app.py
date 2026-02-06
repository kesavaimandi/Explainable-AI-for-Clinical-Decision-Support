from flask import Flask, render_template, request

# ---------------- HEART ----------------
from xai.explainers.heart_shap import explain_heart_prediction
from xai.utils.heart.text_generator import generate_doctor_explanation

# ---------------- DIABETES ----------------
from xai.explainers.diabetes_shap import explain_diabetes_prediction
from xai.utils.diabetes.text_generator import generate_diabetes_doctor_explanation

# ---------------- BREAST CANCER ----------------
from xai.explainers.breast_cancer_shap import explain_breast_cancer_prediction
from xai.utils.breast_cancer.text_generator import generate_breast_doctor_explanation

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template("index.html")

# ================= HEART =================
@app.route("/heart")
def heart_page():
    return render_template("heart.html")

@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    patient_data = [
        int(request.form["age"]),
        int(request.form["sex"]),
        int(request.form["cp"]),
        int(request.form["trestbps"]),
        int(request.form["chol"]),
        int(request.form["fbs"]),
        int(request.form["restecg"]),
        int(request.form["thalach"]),
        int(request.form["exang"]),
        float(request.form["oldpeak"]),
        int(request.form["slope"]),
        int(request.form["ca"]),
        int(request.form["thal"])
    ]

    explanation = explain_heart_prediction(patient_data)
    report = generate_doctor_explanation(explanation)

    return render_template("result.html", report=report, disease="Heart", back_url="/heart")

# ================= DIABETES =================
@app.route("/diabetes")
def diabetes_page():
    return render_template("diabetes.html")

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    patient_data = [
        int(request.form["Pregnancies"]),
        int(request.form["Glucose"]),
        int(request.form["BloodPressure"]),
        int(request.form["SkinThickness"]),
        int(request.form["Insulin"]),
        float(request.form["BMI"]),
        float(request.form["DiabetesPedigreeFunction"]),
        int(request.form["Age"])
    ]

    explanation = explain_diabetes_prediction(patient_data)
    report = generate_diabetes_doctor_explanation(explanation)

    return render_template("result.html", report=report, disease="Diabetes", back_url="/diabetes")

# ================= BREAST CANCER =================
@app.route("/breast_cancer")
def breast_cancer_page():
    return render_template("breast_cancer.html")

@app.route("/predict_breast_cancer", methods=["POST"])
def predict_breast_cancer():

    patient_data = [
        float(request.form["radius_mean"]),
        float(request.form["texture_mean"]),
        float(request.form["perimeter_mean"]),
        float(request.form["area_mean"]),
        float(request.form["smoothness_mean"]),
        float(request.form["compactness_mean"]),
        float(request.form["concavity_mean"]),
        float(request.form["concave_points_mean"]),
        float(request.form["symmetry_mean"]),
        float(request.form["fractal_dimension_mean"]),

        float(request.form["radius_se"]),
        float(request.form["texture_se"]),
        float(request.form["perimeter_se"]),
        float(request.form["area_se"]),
        float(request.form["smoothness_se"]),
        float(request.form["compactness_se"]),
        float(request.form["concavity_se"]),
        float(request.form["concave_points_se"]),
        float(request.form["symmetry_se"]),
        float(request.form["fractal_dimension_se"]),

        float(request.form["radius_worst"]),
        float(request.form["texture_worst"]),
        float(request.form["perimeter_worst"]),
        float(request.form["area_worst"]),
        float(request.form["smoothness_worst"]),
        float(request.form["compactness_worst"]),
        float(request.form["concavity_worst"]),
        float(request.form["concave_points_worst"]),
        float(request.form["symmetry_worst"]),
        float(request.form["fractal_dimension_worst"]),
    ]

    explanation = explain_breast_cancer_prediction(patient_data)
    report = generate_breast_doctor_explanation(explanation)

    return render_template("result.html", report=report, disease="Breast Cancer",back_url = "/breast_cancer")


if __name__ == "__main__":
    app.run(debug=True)
