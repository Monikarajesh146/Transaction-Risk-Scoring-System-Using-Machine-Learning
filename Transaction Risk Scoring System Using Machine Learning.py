# Machine Learning-Based Financial Fraud Detection System


# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from xgboost import XGBClassifier


# Upload Dataset

uploaded = files.upload()


# Load Dataset

df = pd.read_csv("credit_card_fraud_2025.csv")

print(df.head())


# Basic Data Exploration

print(df.shape)

print(df.info())

print(df.isnull().sum())

print(df['Fraud_Flag'].value_counts())


# Exploratory Data Analysis (EDA)

# Fraud vs Non-Fraud Transactions

sns.countplot(x='Fraud_Flag', data=df)

plt.title("Fraud vs Non-Fraud Transactions")

plt.show()


# Fraud by Card Type

plt.figure(figsize=(10,5))

sns.countplot(
    x='Card_Type',
    hue='Fraud_Flag',
    data=df
)

plt.title("Fraud by Card Type")

plt.xticks(rotation=45)

plt.show()


# Fraud by Transaction Type

plt.figure(figsize=(10,5))

sns.countplot(
    x='Transaction_Type',
    hue='Fraud_Flag',
    data=df
)

plt.title("Fraud by Transaction Type")

plt.xticks(rotation=45)

plt.show()


# Transaction Amount Distribution

plt.figure(figsize=(10,5))

sns.histplot(
    df['Amount'],
    bins=50
)

plt.title("Transaction Amount Distribution")

plt.show()


# Fraud vs Distance From Home

plt.figure(figsize=(10,5))

sns.boxplot(
    x='Fraud_Flag',
    y='Distance_From_Home',
    data=df
)

plt.title("Fraud vs Distance From Home")

plt.show()


# Data Preprocessing

df_original = df.copy()

df = df.drop([
    'Transaction_ID',
    'Customer_ID',
    'Merchant_ID',
    'Transaction_Date'
], axis=1)

df = pd.get_dummies(df, drop_first=True)


# Split Features and Target

X = df.drop('Fraud_Flag', axis=1)

y = df['Fraud_Flag']


# Check Class Imbalance

print(y.value_counts())

print(y.value_counts(normalize=True))


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# Logistic Regression

lr = LogisticRegression(
    class_weight='balanced',
    random_state=42
)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

lr_acc = accuracy_score(y_test, y_pred_lr)

lr_f1 = f1_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", lr_acc)

print(classification_report(y_test, y_pred_lr))


# Decision Tree

dt = DecisionTreeClassifier(
    class_weight='balanced',
    random_state=42
)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_acc = accuracy_score(y_test, y_pred_dt)

dt_f1 = f1_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", dt_acc)

print(classification_report(y_test, y_pred_dt))


# Random Forest

rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)

rf_f1 = f1_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", rf_acc)

print(classification_report(y_test, y_pred_rf))


# XGBoost

scale_value = len(y[y == 0]) / len(y[y == 1])

xgb = XGBClassifier(
    scale_pos_weight=scale_value,
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

xgb_acc = accuracy_score(y_test, y_pred_xgb)

xgb_f1 = f1_score(y_test, y_pred_xgb)

print("XGBoost Accuracy:", xgb_acc)

print("XGBoost F1-Score:", xgb_f1)

print(classification_report(y_test, y_pred_xgb))


# Model Comparison Using F1-Score

models = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'XGBoost'
]

scores = [
    lr_f1,
    dt_f1,
    rf_f1,
    xgb_f1
]

plt.figure(figsize=(8,5))

plt.bar(models, scores)

plt.title("Model F1-Score Comparison")

plt.ylabel("F1-Score")

plt.show()


# Feature Importance - Random Forest

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print(feature_importance.head(10))


# Top 10 Important Features

top_features = feature_importance.head(10)

plt.figure(figsize=(10,5))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()

plt.show()


# Business Insight

fraud_distance = df_original[
    df_original['Fraud_Flag'] == 1
]['Distance_From_Home'].median()

normal_distance = df_original[
    df_original['Fraud_Flag'] == 0
]['Distance_From_Home'].median()

print("Median Distance - Fraud:", fraud_distance)

print("Median Distance - Normal:", normal_distance)

print(
    "Distance from home alone was not a strong fraud indicator "
    "in this dataset."
)
