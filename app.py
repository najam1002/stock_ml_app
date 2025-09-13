# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional imports (wrapped)
HAS_PANDAS_TA = False
HAS_XGBOOST = False
HAS_TF = False
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except Exception:
    HAS_PANDAS_TA = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except Exception:
    HAS_TF = False

st.set_page_config(layout="wide", page_title="Stock ML App (Extended)")

# ----------------- Helpers -----------------
def safe_to_numeric(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

def add_basic_derived_features(df, close_col, volume_col=None):
    # Keep original columns as lowercase compatibility
    c = close_col
    df = df.copy()
    # percent ranges
    if "high" in df.columns and "low" in df.columns:
        df["HL_PCT"] = (df["high"] - df["low"]) / df[c] * 100
    if "open" in df.columns:
        df["OC_PCT"] = (df[c] - df["open"]) / df["open"] * 100
    if volume_col and volume_col in df.columns:
        df["VOL_PCT"] = df[volume_col].pct_change().fillna(0)
    return df

def add_indicators(df, close_col, use_ema=True, use_rsi=True, use_bbands=True, use_macd=True):
    df = df.copy()
    # try pandas_ta if available (more robust)
    if HAS_PANDAS_TA:
        if use_ema:
            df["EMA_12"] = ta.ema(df[close_col], length=12)
            df["EMA_26"] = ta.ema(df[close_col], length=26)
        if use_rsi:
            df["RSI_14"] = ta.rsi(df[close_col], length=14)
        if use_bbands:
            bb = ta.bbands(df[close_col], length=20, std=2)
            # bb contains columns like BBM_20_2.0, BBL_20_2.0, BBU_20_2.0, etc.
            for col in bb.columns:
                df[col] = bb[col]
        if use_macd:
            macd = ta.macd(df[close_col])
            for col in macd.columns:
                df[col] = macd[col]
    else:
        # fallback minimal implementations
        if use_ema:
            df["EMA_12"] = df[close_col].ewm(span=12, adjust=False).mean()
            df["EMA_26"] = df[close_col].ewm(span=26, adjust=False).mean()
        if use_rsi:
            delta = df[close_col].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.rolling(14).mean()
            ma_down = down.rolling(14).mean()
            rs = ma_up / (ma_down + 1e-9)
            df["RSI_14"] = 100 - (100 / (1 + rs))
        if use_bbands:
            ma = df[close_col].rolling(20).mean()
            std = df[close_col].rolling(20).std()
            df["BB_MID"] = ma
            df["BB_UP"] = ma + 2 * std
            df["BB_LOW"] = ma - 2 * std
        if use_macd:
            ema12 = df[close_col].ewm(span=12, adjust=False).mean()
            ema26 = df[close_col].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, verbosity=0)
    model.fit(X_train, y_train)
    return model

def create_lstm_model(n_input, n_features):
    # simple LSTM
    model = Sequential()
    model.add(LSTM(64, input_shape=(n_input, n_features), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def series_to_supervised(data, n_input=20, n_out=1):
    # create X,y for LSTM: sliding windows
    X, y = [], []
    for i in range(len(data) - n_input - n_out + 1):
        X.append(data[i:(i + n_input), :])
        y.append(data[i + n_input:i + n_input + n_out, 0])  # predict close (index 0 in array)
    return np.array(X), np.array(y)

def calc_sharpe(returns, annualization=252):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(annualization)

def max_drawdown(cum_returns):
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (running_max - cum_returns) / running_max
    return np.max(drawdown)

def directional_accuracy(y_true, y_pred):
    # direction of change (next day)
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    # align lengths
    m = min(len(true_dir), len(pred_dir))
    if m <= 0:
        return 0.0
    return (true_dir[:m] == pred_dir[:m]).sum() / m

# ----------------- Streamlit UI -----------------
st.title("Stock Prediction App — Extended (keep old + add new)")

# Upload CSV
uploaded_file = st.file_uploader("Choose your CSV file (OHLCV)", type=["csv", "txt"])
if not uploaded_file:
    st.info("Upload a CSV file containing OHLCV (and optional indicators).")
    st.stop()

# Load df
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV: {e}")
    st.stop()

st.subheader("Raw Data (first rows)")
df.columns = [c.strip() for c in df.columns]
st.dataframe(df.head())

# Keep old safe numeric conversion
df = safe_to_numeric(df)

# Show columns and pick Close & Volume (only numeric)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Please upload OHLCV numeric data.")
    st.stop()

st.markdown("**Select Close & Volume columns (numeric only):**")
col1, col2 = st.columns(2)
with col1:
    close_col = st.selectbox("Close column", numeric_cols, index=0)
with col2:
    # try to pick Volume-like column default
    default_vol = None
    for cand in numeric_cols:
        if "vol" in cand.lower() or "quantity" in cand.lower():
            default_vol = cand
            break
    if default_vol and default_vol in numeric_cols:
        default_idx = numeric_cols.index(default_vol)
    else:
        default_idx = 0 if len(numeric_cols) > 1 else 0
    volume_col = st.selectbox("Volume column (optional)", [None] + numeric_cols, index=default_idx+1 if default_idx+1 < len([None]+numeric_cols) else 0)

# Option: keep previous SMMA logic unchanged (if present)
st.markdown("**Legacy/Existing behavior preserved:** The previous SMMA/OHLC pipeline is retained. Below are additional options to expand features & models.")

# Feature engineering options
st.subheader("Feature Engineering Options")
fe_col1, fe_col2 = st.columns(2)
with fe_col1:
    add_basic = st.checkbox("Add basic derived features (HL%, OC%, Volume %)", value=True)
    add_smmas = st.checkbox("Keep existing SMMA features (if present in CSV)", value=True)
with fe_col2:
    use_ema = st.checkbox("Add EMA (12/26)", value=True)
    use_rsi = st.checkbox("Add RSI (14)", value=True)
    use_bbands = st.checkbox("Add Bollinger Bands (20,2)", value=True)
    use_macd = st.checkbox("Add MACD", value=True)

# Model options
st.subheader("Model & Training Options")
models_available = ["Linear Regression"]
if HAS_XGBOOST:
    models_available.append("XGBoost")
if HAS_TF:
    models_available.append("LSTM")
model_choice = st.selectbox("Select model", models_available, index=0)

# Train/test split and walk-forward
test_size = st.slider("Test set size (fraction)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
use_walk_forward = st.checkbox("Use walk-forward (rolling) validation", value=False)
lookback_window = st.number_input("Lookback window (days) for LSTM / walk-forward", min_value=5, max_value=252, value=30)

# Create features
df_proc = df.copy()
if add_basic:
    df_proc = add_basic_derived_features(df_proc, close_col, volume_col)
if use_ema or use_rsi or use_bbands or use_macd:
    df_proc = add_indicators(df_proc, close_col, use_ema=use_ema, use_rsi=use_rsi, use_bbands=use_bbands, use_macd=use_macd)

# Ensure no infinite or object columns in features
df_proc = df_proc.replace([np.inf, -np.inf], np.nan)
df_proc = df_proc.dropna(how="all")

# Candidate feature columns: numeric columns except target (we will set target next)
numeric_cols_proc = df_proc.select_dtypes(include=[np.number]).columns.tolist()
st.write("Numeric columns detected for modeling:", numeric_cols_proc)

# Target creation (next day's close) — keep previous behavior
df_proc["Target"] = pd.to_numeric(df_proc[close_col].shift(-1), errors="coerce")
df_proc = df_proc.dropna().reset_index(drop=True)

# Final feature list: exclude Target
feature_cols = [c for c in numeric_cols_proc if c != "Target"]
# If there are too many features, let user pick a subset
st.subheader("Features to use for model (numeric columns)")
selected_features = st.multiselect("Select feature columns (at least Close/other indicators recommended)", feature_cols, default=[c for c in ["open","high","low",close_col] if c in feature_cols])

if not selected_features:
    st.error("Please select at least one feature column.")
    st.stop()

# Prepare X,y
X = df_proc[selected_features].fillna(0)
y = df_proc["Target"].fillna(0)

# Train/Test split or walk-forward
if use_walk_forward and model_choice != "LSTM":
    st.info("Walk-forward will be simulated by retraining on expanding window and predicting next fold. (LSTM walk-forward uses sliding windows.)")

# Train the selected model
st.subheader("Train & Evaluate")
train_button = st.button("Train model now")
if not train_button:
    st.info("Click 'Train model now' to train and evaluate the selected model.")
    st.stop()

# Training logic
with st.spinner("Training model..."):
    # For LSTM we need to treat data differently
    results = {}
    if model_choice == "Linear Regression":
        if not use_walk_forward:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            model = train_linear_regression(X_train, y_train)
            y_pred = model.predict(X_test)
            # full predictions on entire X for backtest display
            y_pred_full = model.predict(X)
        else:
            # simple expanding-window walk-forward
            n = len(X)
            test_size_count = max(1, int(n * test_size))
            preds = []
            trues = []
            last_test_index = None
            for start in range(0, n - test_size_count, test_size_count):
                train_idx = list(range(0, start + test_size_count))
                test_idx = list(range(start + test_size_count, min(start + 2*test_size_count, n)))
                if not test_idx:
                    break
                mt = LinearRegression()
                mt.fit(X.iloc[train_idx], y.iloc[train_idx])
                p = mt.predict(X.iloc[test_idx])
                preds.extend(p.tolist())
                trues.extend(y.iloc[test_idx].tolist())
                last_test_index = test_idx
            y_test = np.array(trues)
            y_pred = np.array(preds)
            # simplified full preds for plotting (train on all)
            model = train_linear_regression(X, y)
            y_pred_full = model.predict(X)
    elif model_choice == "XGBoost" and HAS_XGBOOST:
        if not use_walk_forward:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            model = train_xgboost(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_full = model.predict(X)
        else:
            # expanding window with xgboost
            n = len(X)
            test_size_count = max(1, int(n * test_size))
            preds = []
            trues = []
            for start in range(0, n - test_size_count, test_size_count):
                train_idx = list(range(0, start + test_size_count))
                test_idx = list(range(start + test_size_count, min(start + 2*test_size_count, n)))
                if not test_idx:
                    break
                mt = XGBRegressor(n_estimators=100, learning_rate=0.05)
                mt.fit(X.iloc[train_idx], y.iloc[train_idx])
                p = mt.predict(X.iloc[test_idx])
                preds.extend(p.tolist())
                trues.extend(y.iloc[test_idx].tolist())
            y_test = np.array(trues)
            y_pred = np.array(preds)
            model = train_xgboost(X, y)
            y_pred_full = model.predict(X)
    elif model_choice == "LSTM" and HAS_TF:
        # prepare data: use selected_features (ensure close_col is first in array so we predict its future)
        arr = X.values
        # we want to scale? minimal approach: normalize by mean/std per column
        mu = arr.mean(axis=0)
        sigma = arr.std(axis=0) + 1e-9
        arr_scaled = (arr - mu) / sigma
        # create supervised windows
        n_input = int(lookback_window)
        X_l, y_l = series_to_supervised(np.hstack([ (df_proc[close_col].values.reshape(-1,1)), arr_scaled]), n_input=n_input, n_out=1)
        y_l = y_l.reshape(-1)
        # train/test split
        split_idx = int(len(X_l)*(1-test_size))
        X_train, X_test = X_l[:split_idx], X_l[split_idx:]
        y_train, y_test = y_l[:split_idx], y_l[split_idx:]
        # build model
        lstm_model = create_lstm_model(n_input, X_l.shape[2])
        # train
        lstm_model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        # predict
        y_pred_scaled = lstm_model.predict(X_test).reshape(-1)
        # invert scaling on predicted close: predicted * sigma_close + mu_close
        # recall close is first column, so mu[0], sigma[0]
        y_pred = y_pred_scaled * sigma[0] + mu[0]
        y_test = y_test * sigma[0] + mu[0]
        # for full series preds, do a rolling predict (not perfect but usable)
        full_preds = []
        inp = X_l.copy()
        for i in range(inp.shape[0]):
            p = lstm_model.predict(inp[i:i+1])
            full_preds.append((p.reshape(-1)[0]*sigma[0] + mu[0]))
        y_pred_full = np.array(full_preds[:len(df_proc)])
        model = lstm_model
    else:
        st.error("Selected model is not available in this environment. Please choose Linear Regression.")
        st.stop()

    # Compute metrics
    # Ensure y_test and y_pred exist
    y_test_arr = np.array(y_test).reshape(-1)
    y_pred_arr = np.array(y_pred).reshape(-1)
    mse = mean_squared_error(y_test_arr, y_pred_arr) if len(y_test_arr) > 0 else float("nan")
    rmse = np.sqrt(mse) if not np.isnan(mse) else float("nan")
    mae = mean_absolute_error(y_test_arr, y_pred_arr) if len(y_test_arr) > 0 else float("nan")
    r2 = r2_score(y_test_arr, y_pred_arr) if len(y_test_arr) > 0 else float("nan")
    dir_acc = directional_accuracy(y_test_arr, y_pred_arr) if len(y_test_arr) > 1 else float("nan")

    # Backtest style strategy: predicted next returns used as signals
    df_back = df_proc.copy()
    try:
        df_back["Predicted"] = y_pred_full
    except Exception:
        df_back["Predicted"] = np.nan

    # Strategy returns: use predicted change as signal (buy if predicted up)
    df_back["Actual_Returns"] = df_back[close_col].pct_change().fillna(0)
    df_back["Predicted_Returns"] = df_back["Predicted"].pct_change().fillna(0)
    # strategy: take sign(predicted_return) as position, calculate strategy returns = position shift * actual_returns
    df_back["Position"] = np.sign(df_back["Predicted_Returns"]).shift(1).fillna(0)
    df_back["Strategy_Returns"] = df_back["Position"] * df_back["Actual_Returns"]
    # cumulative
    cum_strat = (1 + df_back["Strategy_Returns"]).cumprod().fillna(1)
    cum_buy = (1 + df_back["Actual_Returns"]).cumprod().fillna(1)
    sharpe = calc_sharpe(df_back["Strategy_Returns"].dropna())
    mdd = max_drawdown(cum_strat.values)

# Show metrics & plots
st.subheader("Evaluation Metrics")
colm1, colm2, colm3 = st.columns(3)
colm1.metric("MSE (test)", f"{mse:.4f}" if not np.isnan(mse) else "N/A")
colm1.metric("RMSE (test)", f"{rmse:.4f}" if not np.isnan(rmse) else "N/A")
colm2.metric("MAE (test)", f"{mae:.4f}" if not np.isnan(mae) else "N/A")
colm2.metric("R² (test)", f"{r2:.4f}" if not np.isnan(r2) else "N/A")
colm3.metric("Directional Accuracy", f"{dir_acc*100:.2f}%" if not np.isnan(dir_acc) else "N/A")
colm3.metric("Sharpe Ratio (strategy)", f"{sharpe:.2f}")

st.write(f"Max Drawdown: {mdd:.4f}")

# Plot Actual vs Predicted (test region)
st.subheader("Actual vs Predicted (Test set)")
fig, ax = plt.subplots(figsize=(10, 4))
if model_choice == "LSTM" and HAS_TF:
    # y_test and y_pred are arrays after inverse scaling
    ax.plot(y_test_arr, label="Actual (test)")
    ax.plot(y_pred_arr, label="Predicted (test)")
else:
    try:
        # For non-LSTM, we may have indices for test slices - show direct overlay
        ax.plot(y_test_arr, label="Actual (test)")
        ax.plot(y_pred_arr, label="Predicted (test)")
    except Exception:
        ax.text(0.5, 0.5, "No test predictions available", ha='center')
ax.legend()
st.pyplot(fig)

# Plot cumulative returns: Strategy vs Buy-and-hold
st.subheader("Equity Curve: Strategy vs Buy & Hold")
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(cum_buy.values, label="Buy & Hold (actual)", alpha=0.8)
ax2.plot(cum_strat.values, label="Strategy (predicted signal)", alpha=0.8)
ax2.legend()
st.pyplot(fig2)

# Show last rows and predictions table
st.subheader("Recent Data and Predicted Close (Full series predicted)")
display_df = df_back[[c for c in df_back.columns if c in [close_col,"Predicted","Position","Strategy_Returns"]]].tail(50)
st.dataframe(display_df)

# Multi-day prediction using last row recursive approach (keeps previous logic)
st.subheader("Multi-day Forecast (recursive)")
days_to_predict = st.number_input("Number of future days to predict:", min_value=1, max_value=90, value=10)
do_forecast = st.button("Forecast future prices")
if do_forecast:
    # Start from last available feature row
    last_feat = X.iloc[-1:].copy()
    future_prices = []
    for i in range(days_to_predict):
        if model_choice == "LSTM" and HAS_TF:
            # For LSTM do a simplified approach: construct normalized window from last lookback rows
            # create input window from end of df_proc
            arr_all = X.values
            mu = arr_all.mean(axis=0)
            sigma = arr_all.std(axis=0)+1e-9
            if len(arr_all) < lookback_window:
                st.error("Not enough rows for LSTM forecast window.")
                break
            window = arr_all[-lookback_window:]
            # scale and predict
            win_scaled = (window - mu) / sigma
            inp = np.expand_dims(win_scaled, axis=0)
            p_scaled = model.predict(inp)
            p = p_scaled.reshape(-1)[0] * (arr_all[:,0].std()+1e-9) + (arr_all[:,0].mean())  # approx
            pred_price = p
        else:
            pred_price = model.predict(last_feat)[0]
        future_prices.append(pred_price)
        # update last_feat by shifting features: set close_col's column to pred_price, and simple approximations for others
        if close_col in last_feat.columns:
            # If feature includes 'close' explicitly, set it
            if close_col in last_feat.columns:
                last_feat.at[last_feat.index[0], close_col] = pred_price
        # very simple heuristic updates for other features (to keep recursion plausible)
        if "high" in last_feat.columns:
            last_feat["high"] = max(last_feat["high"].iloc[0], pred_price)*1.002
        if "low" in last_feat.columns:
            last_feat["low"] = min(last_feat["low"].iloc[0], pred_price)*0.998
        # update engineered indicators conservatively (rolling mean for EMA/SMA)
        for col in last_feat.columns:
            if col.startswith("EMA") or col.startswith("SMA"):
                last_feat[col] = (last_feat[col].iloc[0]*0.9 + pred_price*0.1)
    # show results
    future_df = pd.DataFrame({"Day": list(range(1, len(future_prices)+1)), "Predicted_Close": future_prices})
    st.table(future_df)
    # also plot
    fig3, ax3 = plt.subplots(figsize=(10,4))
    historical = df_proc[close_col].iloc[-100:].tolist()
    ax3.plot(range(-len(historical),0), historical, label="History (last 100)")
    ax3.plot(range(1, len(future_prices)+1), future_prices, label="Forecast", marker='o')
    ax3.legend()
    st.pyplot(fig3)

st.markdown("---")
st.write("Notes:")
st.write("- Linear Regression is preserved exactly as prior baseline. The app adds indicator-driven features and allows model comparison.")
st.write("- Advanced models (XGBoost/LSTM) are enabled only if the environment has the packages installed. If not, stick with Linear Regression which always runs.")
st.write("- For production-grade forecasting, please consider more data, hyperparameter tuning, feature scaling & cross-validation (walk-forward).")

st.markdown("**If you want, I can now:**")
st.markdown("- Push an updated `app.py` to your GitHub repo and redeploy the Streamlit app (I can give exact commands/PR message).")
st.markdown("- Produce a concise report (PDF) summarizing the new results for TATASTEEL example you shared.")
