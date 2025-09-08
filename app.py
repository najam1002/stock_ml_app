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
    close_col = "Close"  # Change if CSV has different column
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
        st.subheader("Predict Future Close Prices")
        last_row = X.tail(1).copy()  # Start from the latest row
        days_to_predict = st.number_input("Enter number of future days to predict:", min_value=1, max_value=30, value=5)

        future_prices = []
        for i in range(days_to_predict):
            next_price = model.predict(last_row)[0]
            future_prices.append(next_price)
            # Update last_row to simulate next day input
            last_row[close_col] = next_price
            # Keep other features the same (can be improved with more complex methods)
        
        st.write(f"Predicted Close Prices for next {days_to_predict} days:")
        for i, price in enumerate(future_prices, 1):
            st.write(f"Day {i}: {price:.2f}")
