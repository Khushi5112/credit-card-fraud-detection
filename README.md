# Credit Card Fraud Detection using Machine Learning

An end-to-end machine learning project that detects fraudulent credit card transactions, trained on the Kaggle Credit Card Fraud Detection dataset (280,000+ transactions).

## Overview
- Handled severe class imbalance (fraud is <1% of the data) using **SMOTE**
- Trained a **Random Forest** classifier — achieved **96% accuracy** and **83% recall**
- Deployed as an interactive **Streamlit web app** for real-time fraud prediction

## Tech Stack
Python, Scikit-learn, Pandas, imbalanced-learn (SMOTE), Streamlit, Random Forest

## Files
- `app.py` — Streamlit web app for real-time fraud prediction (upload CSV or manually check a transaction)
- `fraud_model.pkl` — trained Random Forest model
- `model_columns.pkl` — feature columns used by the model

## How to run
```bash
pip install streamlit pandas scikit-learn joblib
streamlit run app.py
```

## Dataset
[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
