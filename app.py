import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("Stock Price Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Ensure numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Feature Engineering: Shifted Close as target
    close_col = "Close"  # change if your CSV has a different close column name
    df["Target"] = df[close_col].shift(-1)  # next day's close
    df = df.dropna()

    # Features (OHLCV except Date & Target)
    features = ["Open", "High", "Low", "Volume"]
    features = [f for f in features if f in df.columns]

    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df["Target"], errors="coerce").fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.pr_
