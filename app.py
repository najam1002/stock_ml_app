import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("📈 AI/ML Stock Prediction App")

st.write("Upload OHLCV data (CSV) to generate predictions and visualize trends.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file with OHLCV data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Basic features
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()

    # Simple model: predict Close based on previous Close
    df['Prev_Close'] = df['Close'].shift(1)
    df = df.dropna()

    X = df[['Prev_Close']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    df['Predicted_Close'] = model.predict(X)

    # Plot
    st.subheader("Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.plot(df['Close'].values, label='Actual')
    ax.plot(df['Predicted_Close'].values, label='Predicted')
    ax.legend()
    st.pyplot(fig)

    st.success("Prediction completed ✅")
else:
    st.info("Please upload a CSV file to continue.")
