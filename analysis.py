# Breast Cancer Tumour Classification (MDA512 Assignment 2)
# Models: Logistic Regression, SVM, Random Forest
# Metrics: Accuracy + Precision/Recall/F1 for Malignant class + Confusion Matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # <<< CHANGED: Import SVC for Support Vector Machine
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1) Load dataset
# =========================
DATA_PATH = "tumordata.csv"
df = pd.read_csv(DATA_PATH)
df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df = df.dropna(subset=["diagnosis"]).copy()
df["diagnosis"] = df["diagnosis"].astype(int)

print("Dataset shape:", df.shape)
print("Class counts (0=Benign, 1=Malignant):")
print(df["diagnosis"].value_counts())

# =========================
# 2) Features / Target
# =========================
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# 3) Models (use pipelines)
# =========================
models = {
    "Logistic Regression": Pipeline(steps=[
        ("scaler", StandardScaler()),
        # Removed class_weight for a standard result
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "SVM": Pipeline(steps=[
        ("scaler", StandardScaler()),
        # Added SVM model
        ("model", SVC(kernel='rbf', probability=False, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
        # Removed class_weight for a standard result
    )
}

# =========================
# 4) Train + Evaluate
# =========================
rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, pos_label=1, average="binary"
    )

    rows.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (Malignant)": precision,
        "Recall (Malignant)": recall,
        "F1-Score (Malignant)": f1
    })

    print("\n==============================")
    print("Model:", name)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(rows).sort_values(by="Recall (Malignant)", ascending=False)
print("\n===== Model Performance Summary (sorted by Malignant Recall) =====")
print(results_df)

results_df.to_csv("model_performance_summary.csv", index=False)
print("\nSaved: model_performance_summary.csv")

# =========================
# 5) Feature importance (Random Forest)
# =========================
# Need to access the model step in the pipeline if RF is in a pipeline
# In this case, RF is standalone, so direct access is fine.
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_

fi = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(fi.head(10))

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=fi.head(10))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
