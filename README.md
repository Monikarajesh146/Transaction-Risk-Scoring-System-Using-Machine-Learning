# Machine Learning-Based Financial Fraud Detection System

## Project Overview

This project focuses on identifying high-risk and potentially fraudulent financial transactions using Machine Learning techniques. The workflow includes data preprocessing, exploratory data analysis (EDA), class imbalance handling, model training, evaluation, and feature importance analysis.

---

## Dataset Information

- Dataset: Credit Card Fraud Detection Dataset
- Source: Kaggle
- Total Records: 500,000 transactions
- Dataset Type: Synthetic financial transaction dataset used for educational and analytical purposes

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Google Colab

---

## Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Project Workflow

- Data Collection
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Categorical Encoding & Feature Selection
- Handling Class Imbalance
- Model Training
- Model Evaluation
- Feature Importance Analysis
- Business Insight Generation

---

## Class Imbalance Handling

The dataset was highly imbalanced, with fraudulent transactions representing only 1.5% of total records. Imbalance-aware techniques such as class weighting and `scale_pos_weight` were applied during model training to improve fraud detection performance.

---

## Model Evaluation

Multiple Machine Learning models including Logistic Regression, Decision Tree, Random Forest, and XGBoost were trained and evaluated using imbalance-aware techniques and classification metrics such as Precision, Recall, and F1-Score.

---

## Key Business Insight

The median transaction distance from home was nearly identical for both fraudulent and legitimate transactions (3.47 vs 3.48), indicating that distance alone was not a strong fraud indicator in this dataset. This highlights the importance of combining multiple behavioral and transactional features for effective fraud detection.

---

## Project Visualizations

### Model Comparison

![Model Comparison](screenshots/model_comparison.png)

### Feature Importance

![Feature Importance](screenshots/feature_importance.png)

### Fraud Distribution

![Fraud Distribution](screenshots/fraud_distribution.png)

---

## Dataset Link

https://www.kaggle.com/datasets/prince7489/credit-card-fraud-2025

---

## How to Run

1. Open the notebook in Google Colab
2. Upload the dataset file:
   `credit_card_fraud_2025.csv`
3. Run all cells sequentially

---

## Project Outcome

This project demonstrates an end-to-end Machine Learning workflow for transaction risk analysis, including preprocessing, imbalance-aware modeling, evaluation, visualization, and business-oriented insights.
