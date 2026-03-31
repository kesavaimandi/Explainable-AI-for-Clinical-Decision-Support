# 🧠 XAI Clinical Decision Support System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> **A next-generation medical AI system that predicts disease risk and explains *why* — helping doctors make confident, informed clinical decisions.**

---

## 📌 Overview

The **XAI Clinical Decision Support System** is a Flask-based web application that uses machine learning models enhanced with **SHAP (SHapley Additive exPlanations)** to provide transparent, explainable predictions for three critical medical conditions:

- ❤️ **Heart Disease**
- 🩸 **Diabetes**
- 🎗️ **Breast Cancer**

Unlike traditional black-box AI models, this system generates **doctor-friendly clinical reports** — including current risk indicators, future health risks, recommended precautions, and protective factors — making it suitable for clinical decision support.

> ⚠️ **Disclaimer:** This system is a *doctor support tool* and does **NOT** replace professional medical judgment.

---

## ✨ Features

- 🔬 **Multi-Disease Prediction** — Supports Heart Disease, Diabetes, and Breast Cancer
- 📊 **SHAP-Based Explainability** — Every prediction is backed by feature-level SHAP explanations
- 🏥 **Clinical Report Generation** — Translates ML outputs into human-readable medical language
- 🎨 **Web Interface** — Clean, responsive Flask web UI for easy data entry and report viewing
- 🧪 **Jupyter Notebooks** — Exploratory Data Analysis and model training notebooks included
- 📦 **Pre-trained Models** — Ready-to-use `.pkl` model, scaler, and feature files

---

## 🗂️ Project Structure

```
XAI Model/
│
├── app.py                          # Main Flask application (routes & prediction logic)
├── requirements.txt                # Python dependencies
│
├── datasets/                       # Raw medical datasets
│   ├── breast_cancer.csv
│   ├── diabetes.csv
│   └── heart.csv
│
├── models/                         # Pre-trained ML model artifacts
│   ├── breast_cancer_model.pkl
│   ├── breast_cancer_scaler.pkl
│   ├── breast_cancer_features.pkl
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── diabetes_features.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   └── heart_features.pkl
│
├── notebook/                       # Jupyter notebooks for EDA & model training
│   ├── breast_cancer.ipynb
│   ├── diabetes.ipynb
│   ├── heart.ipynb
│   └── org/                        # Original/draft notebooks
│
├── xai/
│   ├── explainers/                 # SHAP explainer logic per disease
│   │   ├── heart_shap.py
│   │   ├── diabetes_shap.py
│   │   └── breast_cancer_shap.py
│   │
│   └── utils/                      # Clinical text generation & mapping
│       ├── heart/
│       │   ├── clinical_mapping.py
│       │   ├── clinical_thresholds.py
│       │   └── text_generator.py
│       ├── diabetes/
│       │   ├── clinical_mapping.py
│       │   └── text_generator.py
│       └── breast_cancer/
│           ├── clinical_mapping.py
│           └── text_generator.py
│
├── templates/                      # Jinja2 HTML templates
│   ├── index.html
│   ├── heart.html
│   ├── diabetes.html
│   ├── breast_cancer.html
│   └── result.html
│
├── static/
│   └── style.css                   # CSS styling
│
├── run_heart_xai.py                # Standalone CLI runner for heart prediction
├── run_diabetes_xai.py             # Standalone CLI runner for diabetes prediction
└── run_breast_cancer_xai.py        # Standalone CLI runner for breast cancer prediction
```

---

## 🩺 Supported Diseases & Input Features

### ❤️ Heart Disease (13 Features)
| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (1 = Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type |

### 🩸 Diabetes (8 Features)
| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (μU/ml) |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Genetic diabetes likelihood function |
| `Age` | Age of the patient |

### 🎗️ Breast Cancer (30 Features)
Includes mean, standard error, and worst values for 10 cell nucleus measurements: `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave_points`, `symmetry`, and `fractal_dimension`.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.12+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/xai-clinical-decision-support.git
cd xai-clinical-decision-support
```

### 2. Install dependencies

```bash
pip install flask shap joblib scikit-learn pandas numpy
```

> The `requirements.txt` currently lists only `shap` and `joblib`. For the full web app, install all of the above.

### 3. Fix model paths (Important!)

The SHAP explainer files currently use **absolute Windows paths**. Before running, update the paths in the following files to use relative paths:

**`xai/explainers/heart_shap.py`**
```python
# Replace hardcoded paths like:
# model = joblib.load("C:\\Users\\imand\\...\\heart_model.pkl")

# With relative paths:
import os
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model = joblib.load(os.path.join(BASE, "models", "heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE, "models", "heart_scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE, "models", "heart_features.pkl"))
data = pd.read_csv(os.path.join(BASE, "datasets", "heart.csv"))
```

Apply the same pattern to `diabetes_shap.py` and `breast_cancer_shap.py`.

### 4. Run the application

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 🚀 Usage

### Web Application

1. Navigate to `http://127.0.0.1:5000`
2. Select a disease module from the landing page
3. Enter the patient's clinical parameters in the form
4. Submit the form to receive a **Clinical Decision Report** including:
   - Risk Level (Low / Medium / High)
   - Risk Percentage
   - Current Possible Problems
   - Future Health Risks
   - Recommended Precautions
   - Protective Factors

### Command-Line (Standalone Runners)

```bash
# Heart Disease prediction
python run_heart_xai.py

# Diabetes prediction
python run_diabetes_xai.py

# Breast Cancer prediction
python run_breast_cancer_xai.py
```

---

## 🔬 How It Works

```
Patient Input → Feature Preprocessing (Scaler) → ML Model Prediction
                                                        ↓
                                              SHAP Explainer
                                                        ↓
                                         Feature Contribution Values
                                                        ↓
                                      Clinical Text Generator (Mapping)
                                                        ↓
                                        Doctor-Friendly Clinical Report
```

1. **Patient data** is entered via the web form
2. Data is **scaled** using the pre-fitted `StandardScaler`
3. The **ML model** generates a risk probability
4. **SHAP** computes each feature's contribution to the prediction
5. The **Clinical Text Generator** maps SHAP values to medical language using a clinical knowledge dictionary
6. A structured **clinical report** is returned to the doctor

---

## 📓 Notebooks

Jupyter notebooks are provided for each disease in `notebook/`:

- **EDA** — Distribution analysis, correlation matrices, missing value handling
- **Model Training** — Feature selection, model fitting, cross-validation
- **SHAP Visualization** — Summary plots, force plots, dependency plots

Run notebooks with:
```bash
jupyter notebook
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| ML / AI | scikit-learn, SHAP |
| Data | pandas, numpy |
| Model Persistence | joblib |
| Frontend | HTML5, CSS3, Jinja2 |
| Notebooks | Jupyter |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Developed as a clinical AI research project.  
For questions or feedback, open an issue on GitHub.

---

> *"The goal of explainable AI in medicine is not to replace the physician — it is to give them a trusted second opinion."*
