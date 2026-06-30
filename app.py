# ============================================================
# HEART DISEASE PREDICTION - WEB APP (Streamlit)
# Models: Logistic Regression & Decision Tree (Capstone Compliant)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ---------- Page Setup ----------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")


# ---------- Train Model (Cached) ----------
@st.cache_resource
def load_best_model():
    df = pd.read_csv('heart_disease_uci.csv')

    # Clean
    df.drop(columns=['id', 'dataset'], inplace=True)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['num'], inplace=True)

    # Fill missing
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_cols = [c for c in num_cols if c != 'target']
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    # Encode
    le = LabelEncoder()
    for c in cat_cols:
        df[c] = le.fit_transform(df[c].astype(str))

    # Scale
    X = df.drop(columns=['target'])
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train LR and DT
    lr = LogisticRegression(max_iter=1000, random_state=42)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    # Pick best based on test accuracy
    if accuracy_score(y_test, lr.predict(X_test)) >= accuracy_score(y_test, dt.predict(X_test)):
        return lr, scaler, X.columns.tolist(), "Logistic Regression"
    else:
        return dt, scaler, X.columns.tolist(), "Decision Tree"


model, scaler, feature_names, model_name = load_best_model()

# ---------- UI ----------
st.title("❤️ Heart Disease Prediction System")
st.markdown(f"**Live Model:** `{model_name}` (Trained on 920 patients)")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    trestbps = st.number_input("Resting BP", 50, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    thalch = st.number_input("Max Heart Rate", 50, 250, 150)
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1)
    ca = st.number_input("Major Vessels (0-3)", 0, 3, 0)

with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    fbs = st.selectbox("Fasting Sugar > 120", ["FALSE", "TRUE"])
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
    exang = st.selectbox("Exercise Angina", ["FALSE", "TRUE"])
    slope = st.selectbox("ST Slope", ["upsloping", "flat", "downsloping"])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

if st.button("🔍 Predict"):
    # Map inputs to numbers
    sex_enc = 1 if sex == "Male" else 0
    cp_map = {"typical angina": 3, "atypical angina": 0, "non-anginal": 2, "asymptomatic": 1}
    fbs_enc = 1 if fbs == "TRUE" else 0
    restecg_map = {"normal": 1, "lv hypertrophy": 0, "st-t abnormality": 2}
    exang_enc = 1 if exang == "TRUE" else 0
    slope_map = {"upsloping": 2, "flat": 1, "downsloping": 0}
    thal_map = {"normal": 2, "fixed defect": 0, "reversable defect": 1}

    input_df = pd.DataFrame([[
        age, sex_enc, cp_map[cp], trestbps, chol, fbs_enc,
        restecg_map[restecg], thalch, exang_enc, oldpeak,
        slope_map[slope], ca, thal_map[thal]
    ]], columns=feature_names)

    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if pred == 1:
        st.error(f"⚠️ **High Risk!** ({prob * 100:.1f}% probability)")
    else:
        st.success(f" **Low Risk!** ({(1 - prob) * 100:.1f}% probability)")

    st.caption("Disclaimer: For educational purposes only.")

st.markdown("---")
st.caption("Built by Nilesh | Virtual Internship Week 4 Capstone")