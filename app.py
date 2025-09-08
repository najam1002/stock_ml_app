import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.title("AI/ML Stock Prediction & Analysis App")

# ----------------- Upload Data -----------------
uploaded_file = st.file_uploader("Upload CSV with OHLCV data", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]  # Clean column names
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Select numeric columns for Close
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in CSV.")
    else:
        close_col = st.selectbox("Select Close Column", numeric_cols)
        volume_col = st.selectbox("Select Volume Column", numeric_cols)

        # ----------------- Feature Engineering -----------------
        st.subheader("Feature Engineering")

        # Basic OHLCV derived features
        df["HL_PCT"] = (df["high"] - df["low"]) / df["close"] * 100
        df["OC_PCT"] = (df["close"] - df["open"]) / df["open"] * 100
        df["VOL_PCT"] = df[volume_col].pct_change().fillna(0)

        # Optional: SMAs, EMA, RSI, Bollinger, etc.
        df["SMA_5"] = df[close_col].rolling(5).mean()
        df["SMA_20"] = df[close_col].rolling(20).mean()
        df["SMA_diff_5_20"] = df["SMA_5"] - df["SMA_20"]
        df = df.dropna()

        st.write("Engineered Features:")
        st.dataframe(df.tail())

        # ----------------- ML Model: Linear Regression -----------------
        st.subheader("Train ML Model for Next Day Close Prediction")

        df["Target"] = df[close_col].shift(-1)
        df = df.dropna()

        feature_cols = ["open", "high", "low", "HL_PCT", "OC_PCT", "VOL_PCT",
                        "SMA_5", "SMA_20", "SMA_diff_5_20"]
        feature_cols = [f for f in feature_cols if f in df.columns]

        X = df[feature_cols]
        y = df["Target"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")

        # ----------------- Multi-day Prediction -----------------
        st.subheader("Predict Future Close Prices")
        last_row = X.tail(1).copy()
        days_to_predict = st.number_input("Enter number of future days:", min_value=1, max_value=30, value=5)

        future_prices = []
        for i in range(days_to_predict):
            next_price = model.predict(last_row)[0]
            future_prices.append(next_price)
            # Update features for next day (simplified approach)
            last_row["open"] = next_price
            last_row["high"] = next_price * 1.01  # approx high
            last_row["low"] = next_price * 0.99   # approx low
            last_row["HL_PCT"] = (last_row["high"] - last_row["low"]) / next_price * 100
            last_row["OC_PCT"] = (next_price - last_row["open"]) / last_row["open"] * 100
            last_row["SMA_5"] = (last_row["SMA_5"]*4 + next_price)/5
            last_row["SMA_20"] = (last_row["SMA_20"]*19 + next_price)/20
            last_row["SMA_diff_5_20"] = last_row["SMA_5"] - last_row["SMA_20"]
            last_row["VOL_PCT"] = 0

        st.write(f"Predicted Close Prices for next {days_to_predict} days:")
        for i, price in enumerate(future_prices, 1):
            st.write(f"Day {i}: {price:.2f}")

        # ----------------- Backtesting / Metrics -----------------
        st.subheader("Backtesting & Metrics")

        df["Predicted"] = model.predict(X)
        df["Returns"] = df["close"].pct_change().fillna(0)
        df["Strategy"] = df["Predicted"].pct_change().shift(-1).fillna(0)

        # Sharpe Ratio
        sharpe_ratio = (df["Strategy"].mean() / df["Strategy"].std()) * np.sqrt(252)
        st.write(f"Strategy Sharpe Ratio: {sharpe_ratio:.2f}")

        # Max Drawdown
        cumulative = (1 + df["Strategy"]).cumprod()
        max_dd = (cumulative.cummax() - cumulative).max()
        st.write(f"Max Drawdown: {max_dd:.2f}")

        # ----------------- Visualization -----------------
        st.subheader("Price Chart & Predictions")
        plt.figure(figsize=(12,6))
        plt.plot(df["close"], label="Actual Close")
        plt.plot(df["Predicted"], label="Predicted Close")
        plt.title("Stock Close Price vs Prediction")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)
