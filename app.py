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
    if close_col not in df.columns:
        st.error(f"Column '{close_col}' not found in the CSV.")
    else:
        df["Target"] = pd.to_numeric(df[close_col].shift(-1), errors="coerce")
        df = df.dropna()

        # Automatically detect numeric columns for features (exclude Target)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if "Target" in numeric_cols:
            numeric_cols.remove("Target")
        X = df[numeric_cols].fillna(0)
        y = df["Target"].fillna(0)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")

        # Predict next day
        st.subheader("Predict Next Day Close")
        last_row = X.tail(1)
        next_day_pred = model.predict(last_row)[0]
        st.write(f"Predicted Next Day Close: {next_day_pred:.2f}")
