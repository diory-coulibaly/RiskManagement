


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
import io

st.title("📈 Logistic Regression on GOOG Returns with Lag Analysis")
# 📷 Quick preview of expected CSV format
from PIL import Image
image = Image.open("Goog.JPG")
st.image(image, caption="CSV Format: Stocks, SP500", use_container_width=True, output_format="JPEG")

st.sidebar.header("Upload Files")
goog_file = st.sidebar.file_uploader("Upload GOOG CSV (semicolon-separated)", type="csv")
sp500_file = st.sidebar.file_uploader("Upload S&P500 CSV (semicolon-separated)", type="csv")

lags = st.sidebar.multiselect("Select lags to test (in days)", [1, 2, 3, 5, 10], default=[1, 2, 5])
run_button = st.sidebar.button("Run Model")

def load_and_prepare(goog_file, sp500_file):
    goog = pd.read_csv(goog_file, sep=";")[["Date", "Adj.Close"]].rename(columns={"Adj.Close": "goog_price"})
    sp500 = pd.read_csv(sp500_file, sep=";")[["Date", "Adj.Close"]].rename(columns={"Adj.Close": "sp_price"})
    for df in [goog, sp500]:
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    df = pd.merge(goog, sp500, on="Date")
    df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y')
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    df["goog_ret"] = df["goog_price"].pct_change()
    df["sp_ret"] = df["sp_price"].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df

def create_lag_features(df, lag):
    df = df.copy()
    df["goog_lag"] = df["goog_ret"].shift(-lag)
    df["sp_lag"] = df["sp_ret"].shift(-lag)
    df["goog_up"] = (df["goog_ret"] >= 0).astype(int)
    return df.dropna().reset_index(drop=True)

def run_model(df):
    X = df[["goog_lag", "sp_lag"]]
    y = df["goog_up"]
    model = LogisticRegression()
    model.fit(X, y)
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return y, y_pred, y_pred_proba, model

if run_button and goog_file and sp500_file:
    df_raw = load_and_prepare(goog_file, sp500_file)
    summary = []
    roc_data = []

    for lag in lags:
        df_lag = create_lag_features(df_raw, lag)
        y_true, y_pred, y_proba, model = run_model(df_lag)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        summary.append({
            "Lag": lag, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1 Score": f1, "AUC": auc
        })

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_data.append((lag, fpr, tpr, auc))

    df_summary = pd.DataFrame(summary).sort_values("Lag")
    st.subheader("📊 Model Performance Summary")
    st.dataframe(df_summary.set_index("Lag").style.format("{:.3f}"))

    # Plot AUC + Accuracy vs Lags
    st.subheader("📈 Accuracy & AUC by Lag")
    fig, ax = plt.subplots()
    ax.plot(df_summary["Lag"], df_summary["Accuracy"], marker="o", label="Accuracy", color="blue")
    ax.plot(df_summary["Lag"], df_summary["AUC"], marker="o", label="AUC", linestyle="--", color="green")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Score")
    ax.set_title("Performance by Lag")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ROC Curve
    st.subheader("📉 ROC Curve by Lag")
    fig2, ax2 = plt.subplots()
    for lag, fpr, tpr, auc_val in roc_data:
        ax2.plot(fpr, tpr, label=f"Lag {lag} (AUC={auc_val:.2f})")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Download results
    csv = df_summary.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Summary as CSV", data=csv, file_name="lag_model_summary.csv", mime='text/csv')
else:
    st.info("⬅️ Upload both files and click 'Run Model'.")
