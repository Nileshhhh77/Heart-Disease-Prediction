# ❤️ Heart Disease Prediction System

### Virtual Internship — Week 4 Capstone Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nileshhhh77-heart-disease-prediction.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

A complete, end-to-end Machine Learning pipeline that predicts whether a patient is likely to have heart disease based on their medical attributes. Built as the **Week 4 Capstone Project** for a Virtual Internship.

The project covers the full ML lifecycle:

- ✅ Data cleaning and preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Model training — Logistic Regression & Decision Tree
- ✅ Model evaluation — Accuracy, Precision, Recall, F1 Score
- ✅ Confusion Matrix & ROC Curve visualization
- ✅ Live web application deployed on Streamlit Cloud

---

## 🚀 Live Demo

### 🌐 [**Try the App**](https://nileshhhh77-heart-disease-prediction-app-hptizc.streamlit.app/)

No installation needed — open the link, enter patient details, and get an instant prediction.

---

## 📊 Dataset

**Source:** [Heart Disease UCI Dataset — Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

| Detail | Value |
|---|---|
| Total Samples | 920 patients |
| Total Features | 16 columns |
| Target | Binary — 0 = No Disease, 1 = Disease |

### Features Used

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = Male, 0 = Female |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting electrocardiographic results |
| `thalch` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia |

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Logistic Regression, Decision Tree) |
| Visualization | Matplotlib, Seaborn |
| Web Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── app.py                   # Streamlit web application
├── week4_capstone.py        # Full ML pipeline script
├── heart_disease_uci.csv    # Dataset
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── confusion_matrix.png     # Confusion matrix visualization
└── roc_curve.png            # ROC curve visualization
```

---

## 🔧 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- Git (optional, for cloning)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Nileshhhh77/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Full Pipeline (Training & Evaluation)
```bash
python week4_capstone.py
```
This loads and cleans the data, trains both models, prints accuracy/precision/recall/F1, and generates `confusion_matrix.png` and `roc_curve.png`.

### Step 4 — Run the Web App
```bash
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~85% | ~83% | ~87% | ~85% | ~0.90 |
| Decision Tree | ~82% | ~81% | ~84% | ~82% | ~0.84 |

**Key Insight:** Logistic Regression performed slightly better on this dataset and is the model deployed in the live web app.

---

## 📈 Visualizations

The pipeline generates two key visualizations saved as image files:

- **`confusion_matrix.png`** — side-by-side confusion matrices for both models
- **`roc_curve.png`** — ROC curves with AUC scores for both models

---

## 🌐 Deployment

Deployed on **Streamlit Community Cloud**:

🔗 **https://nileshhhh77-heart-disease-prediction-app-hptizc.streamlit.app/**

**How it works:** code is pushed to GitHub → Streamlit Cloud builds the app from `requirements.txt` and `app.py` → the app redeploys automatically on every new push to `main`.

---

## 📝 Documentation & Presentation

- **Project Report:** Full written documentation covering data cleaning, EDA, model selection, evaluation, and conclusions (see Google Drive link in submission).
- **Video Presentation:** A 5–7 minute walkthrough covering the dataset, pipeline, evaluation metrics, and a live demo of the app.

---

## 👤 Author

**Nilesh**
Virtual Internship — Week 4 Capstone Project

---

## ⚠️ Disclaimer

This application is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions regarding a medical condition.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the original Heart Disease dataset
- Kaggle for hosting the dataset
- Streamlit for making deployment simple
- Scikit-learn for the machine learning tools

---

⭐ If you found this project useful, consider starring the repository on GitHub!