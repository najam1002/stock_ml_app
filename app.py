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
        # Feature Engineering: Shifted Close as target
        df["Target"] = df[close_col].shift(-1)  # next day's close
        df = df.dropna()

        # Features (OHLCV except Date & Target)
        features = ["Open", "High", "Low", "Volume"]
        features = [f for f in features if f in df.columns]

        X = df[features]
        y = df["Target"]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"📉 Mean Squared Error: {mse:.4f}")
        st.write(f"📊 R² Score: {r2:.4f}")

        # Plot actual vs predicted
        st.subheader("📊 Actual vs Predicted Closing Prices")
        fig, ax = plt.subplots()
        ax.plot(y_test.index, y_test, label="Actual", color="blue")
        ax.plot(y_test.index, y_pred, label="Predicted", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        st.pyplot(fig)

        # Predict Next Day Close
        last_row = df[features].iloc[[-1]]
        next_pred = model.predict(last_row)[0]

        st.success(f"🔮 Predicted Next Day Closing Price: **{next_pred:.2f}**")

else:
    st.info("👆 Please upload a CSV file with OHLCV data.")
