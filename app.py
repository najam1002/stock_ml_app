import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.title("📈 AI/ML Stock Prediction & Analysis App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your OHLCV CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Detect date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    # Detect Close column
    close_col = None
    for candidate in ["Close", "close", "Adj Close", "adj_close", "Closing Price", "Price"]:
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        st.error("❌ No 'Close' column found in the uploaded CSV.")
    else:
        # Ensure numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# Feature Engineering: Shifted Close as target
df["Target"] = df[close_col].shift(-1)  # next day's close
df = df.dropna()

# Features (OHLCV except Date & Target)
features = ["Open", "High", "Low", "Volume"]
features = [f for f in features if f in df.columns]

X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
y = pd.to_numeric(df["Target"], errors="coerce").fillna(0)
