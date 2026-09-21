"""
Credit Card Fraud Detection - Streamlit App
--------------------------------------------
Loads the model trained in Credit_Card_Fraud_Detection.ipynb
(fraud_model.pkl + model_columns.pkl must be in the same folder as this file)
and predicts fraud risk for transactions in real time.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

# ---------- Load model ----------
MODEL_PATH = "fraud_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

@st.cache_resource
def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns

model, model_columns = load_model()

st.title("💳 Credit Card Fraud Detection")
st.caption("Random Forest model trained on the Kaggle Credit Card Fraud dataset, balanced with SMOTE.")

if model is None:
    st.error(
        "Model files not found. Place `fraud_model.pkl` and `model_columns.pkl` "
        "(downloaded from the training notebook, Step 10) in the same folder as this app.py."
    )
    st.stop()

tab1, tab2 = st.tabs(["📁 Upload transactions (CSV)", "✍️ Manual single check"])

# ---------- Tab 1: batch CSV prediction ----------
with tab1:
    st.subheader("Upload a CSV of transactions")
    st.write(
        "The file should have the same columns as the original dataset "
        "(`Time`, `V1`...`V28`, `Amount` — `Class` is optional/ignored if present)."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        data = raw.copy()

        # Recreate the same preprocessing used in training
        if "Amount" in data.columns:
            data["scaled_amount"] = (data["Amount"] - data["Amount"].mean()) / data["Amount"].std()
            data = data.drop("Amount", axis=1)
        if "Time" in data.columns:
            data["scaled_time"] = (data["Time"] - data["Time"].mean()) / data["Time"].std()
            data = data.drop("Time", axis=1)
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        # Align columns to what the model expects
        missing = [c for c in model_columns if c not in data.columns]
        if missing:
            st.error(f"Uploaded file is missing expected columns: {missing}")
        else:
            data = data[model_columns]
            probs = model.predict_proba(data)[:, 1]
            preds = model.predict(data)

            results = raw.copy()
            results["Fraud Risk (%)"] = (probs * 100).round(2)
            results["Prediction"] = ["🚨 Fraud" if p == 1 else "✅ Genuine" for p in preds]

            flagged = int(preds.sum())
            st.success(f"Scanned {len(results)} transactions — {flagged} flagged as fraud.")

            st.dataframe(
                results.sort_values("Fraud Risk (%)", ascending=False),
                use_container_width=True
            )

            st.download_button(
                "Download results as CSV",
                results.to_csv(index=False).encode("utf-8"),
                "fraud_predictions.csv",
                "text/csv"
            )

# ---------- Tab 2: manual single transaction ----------
with tab2:
    st.subheader("Check a single transaction")
    st.info(
        "V1-V28 are anonymized PCA features from the original dataset and aren't "
        "something a real user would type in by hand — this simplified form is for "
        "demoing the model. In a production system, these would be computed "
        "automatically from real transaction data."
    )

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
    with col2:
        time_val = st.number_input("Time (seconds since first transaction in dataset)", min_value=0.0, value=50000.0, step=1000.0)

    st.write("Optional: adjust a few of the most influential anonymized features (defaults = average transaction).")
    v_cols = [c for c in model_columns if c.startswith("V")]
    top_v = v_cols[:4] if len(v_cols) >= 4 else v_cols  # keep the manual form short
    v_values = {}
    cols = st.columns(len(top_v)) if top_v else []
    for c, col in zip(top_v, cols):
        with col:
            v_values[c] = st.slider(c, -10.0, 10.0, 0.0, 0.1)

    if st.button("Predict Fraud Risk", type="primary"):
        row = {c: 0.0 for c in model_columns if c.startswith("V")}
        row.update(v_values)
        row["scaled_amount"] = (amount - 88.35) / 250.12  # approx dataset mean/std for Amount
        row["scaled_time"] = (time_val - 94813.86) / 47488.15  # approx dataset mean/std for Time

        input_df = pd.DataFrame([row])[model_columns]
        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        st.metric("Fraud Risk", f"{prob*100:.2f}%")
        if pred == 1:
            st.error("🚨 This transaction is predicted as FRAUD.")
        else:
            st.success("✅ This transaction is predicted as GENUINE.")

st.divider()
st.caption("Built with scikit-learn, imbalanced-learn (SMOTE), and Streamlit.")
