# ============================================================
# VIRTUAL INTERNSHIP - WEEK 4 CAPSTONE PROJECT
# ============================================================
# Name        : Nilesh
# Project     : Heart Disease Prediction System
# Models Used : Logistic Regression, Decision Tree
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   PHASE 1: LOADING DATA")
print("=" * 60)

# Load data
df = pd.read_csv('heart_disease_uci.csv')
print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop useless columns
df.drop(columns=['id', 'dataset'], inplace=True)

# Convert target to binary (0 = No Disease, 1 = Disease)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=['num'], inplace=True)

# Handle missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols = [c for c in num_cols if c != 'target']
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# Encode categorical
le = LabelEncoder()
for c in cat_cols:
    df[c] = le.fit_transform(df[c].astype(str))

# Split and Scale
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\n" + "=" * 60)
print("   PHASE 2: TRAINING MODELS")
print("=" * 60)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# Model 2: Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

# Evaluate
print("\n--- Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.4f}")

print("\n--- Decision Tree ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_dt):.4f}")

print("\n" + "=" * 60)
print("   PHASE 3: CONFUSION MATRIX & ROC CURVE")
print("=" * 60)

# Plot Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Decision Tree')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print(" Saved confusion_matrix.png")

# Plot ROC
plt.figure(figsize=(8, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = auc(fpr_lr, tpr_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
auc_dt = auc(fpr_dt, tpr_dt)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.4f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curve.png')
print(" Saved roc_curve.png")

print("\n" + "=" * 60)
print("   CAPSTONE COMPLETED SUCCESSFULLY!")
print("=" * 60)