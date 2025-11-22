import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== Loading all UNSW-NB15 parts ===")

# Load column names
features = pd.read_csv("NUSW-NB15_features.csv", encoding="ISO-8859-1")
column_names = features["Name"].tolist()

# Load all 4 dataset parts
df1 = pd.read_csv("UNSW-NB15_1.csv", header=None, names=column_names, low_memory=False)
df2 = pd.read_csv("UNSW-NB15_2.csv", header=None, names=column_names, low_memory=False)
df3 = pd.read_csv("UNSW-NB15_3.csv", header=None, names=column_names, low_memory=False)
df4 = pd.read_csv("UNSW-NB15_4.csv", header=None, names=column_names, low_memory=False)

# Merge
df = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)

print("Merged dataset shape:", df.shape)
print(df.head())

print("\n=== Cleaning dataset ===")

# 1. Drop rows where Label is NaN
df = df.dropna(subset=['Label'])

# 2. Convert Label to integer
df['Label'] = df['Label'].astype(int)

# 3. Fix space inside column name
df = df.rename(columns={"ct_src_ ltm": "ct_src_ltm"})

# 4. Fill attack_cat missing values with "Normal"
df['attack_cat'] = df['attack_cat'].fillna("Normal")

# 5. Remove IP addresses (cannot be used directly as features)
df = df.drop(columns=['srcip', 'dstip'])

# 6. Identify categorical columns
categorical_cols = ['proto', 'state', 'service', 'attack_cat']

# 7. Convert numerical-looking columns to numeric
for col in df.columns:
    if col not in categorical_cols + ['Label']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 8. Replace NaN numeric values with 0
df = df.fillna(0)

# 9. Encode categorical features
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

print("Cleaning done. Shape after cleaning:", df.shape)
print(df.head())

from sklearn.model_selection import train_test_split

print("\n=== Train/Test Split (Stratified) ===")

X = df.drop(columns=["Label"])
y = df["Label"]

# Stratified split to keep same attack/normal ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

print("Label distribution in TRAIN:")
print(y_train.value_counts())

print("\nLabel distribution in TEST:")
print(y_test.value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("\n=== Training RandomForest ===")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

rf.fit(X_train, y_train)

print("Training completed.")

# --- Evaluation ---
print("\n=== Evaluating RandomForest ===")
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - RandomForest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("rf_confusion_matrix.png")

print("Confusion matrix saved as rf_confusion_matrix.png")

# --- Save model ---
joblib.dump(rf, "rf_ids_model.pkl")
print("Model saved as rf_ids_model.pkl")

#################################################################

# --- Feature Importance ---
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10,6))
plt.title("Top 20 Most Important Features")
plt.bar(range(20), importances[indices][:20])
plt.xticks(range(20), feature_names[indices][:20], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance.png")

print("Feature importance saved as feature_importance.png")


#################################################################
# ======================= XGBOOST MODEL =========================
#################################################################

from xgboost import XGBClassifier

print("\n=== Training XGBoost ===")

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=12,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist"  # FAST training method
)

xgb.fit(X_train, y_train)
print("XGBoost training completed!")

# ------- Evaluation -------
print("\n=== Evaluating XGBoost ===")
y_pred_xgb = xgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

# ------- Confusion Matrix -------
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(7,5))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix.png")

print("XGBoost confusion matrix saved as xgb_confusion_matrix.png")

# ------- Save Model -------
joblib.dump(xgb, "xgb_ids_model.pkl")
print("XGBoost model saved as xgb_ids_model.pkl")
