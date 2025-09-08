import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Stock Prediction & Analysis App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your OHLCV CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Detect date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    # Detect Close column (case-insensitive & variations)
    close_col = None
    for candidate in ["Close", "close", "Adj Close", "adj_close", "Closing Price", "Price"]:
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        st.error("❌ No 'Close' column found in the uploaded CSV. Please ensure the file has one.")
    else:
        # Calculate Returns
        df["Return"] = df[close_col].pct_change()

        st.subheader("Data with Returns")
        st.write(df.head())

        # Plot Closing Price
        st.subheader("📊 Closing Price Over Time")
        fig, ax = plt.subplots()
        ax.plot(df.index, df[close_col], label="Closing Price", color="blue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Plot Returns
        st.subheader("📊 Returns Over Time")
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Return"], label="Returns", color="green")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend()
        st.pyplot(fig)

        st.success("✅ Data processed successfully!")

else:
    st.info("👆 Please upload a CSV file with OHLCV data to begin.")
