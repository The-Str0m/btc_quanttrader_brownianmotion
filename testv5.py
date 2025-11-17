import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
TICKER = "BTC-USD"
PERIOD = "2y"
INTERVAL = "1h"
TRANSACTION_COST = 0.0005   # 0.05% per trade
HORIZ_HOURS = 1             # trading horizon
HOURS_PER_YEAR = 365 * 24

# ----------------------------
# DOWNLOAD DATA
# ----------------------------
print("Downloading historical data...")
data = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
data = data[["Close"]].copy()
data.index = pd.to_datetime(data.index)

# ----------------------------
# COMPUTE LOG RETURNS
# ----------------------------
log_ret = np.log(data["Close"] / data["Close"].shift(1)).dropna()
dt = 1 / HOURS_PER_YEAR

# Estimate GBM parameters
mu = float(log_ret.mean() / dt)
sigma = float(log_ret.std() / np.sqrt(dt))
print(f"Estimated drift: {mu:.5f}, volatility: {sigma:.5f}")

# ----------------------------
# SIMULATE PREDICTED RETURN (GBM ONE-HORIZON STEP)
# ----------------------------
# GBM expected return: r_pred = mu * dt
predicted_ret = mu * dt  # expected hourly return
# Generate signal: 1 = long, -1 = short
signal = np.where(predicted_ret > 0, 1, -1)

# ----------------------------
# BACKTEST
# ----------------------------
df = data.copy()
df = df.iloc[1:].copy()  # align with log returns
df["pred_signal"] = signal
df["future_ret"] = df["Close"].pct_change(HORIZ_HOURS).shift(-HORIZ_HOURS)
df["prev_signal"] = df["pred_signal"].shift(1).fillna(0)
df["trade"] = ((df["pred_signal"] != df["prev_signal"]) & (df["pred_signal"] != 0)).astype(int)
df["tcost"] = df["trade"] * TRANSACTION_COST
df["strategy_ret"] = df["pred_signal"] * df["future_ret"] - df["tcost"]
df["equity"] = (1 + df["strategy_ret"]).cumprod()

# Metrics
total_return = df["equity"].iloc[-1] - 1
mean_r = df["strategy_ret"].mean()
std_r = df["strategy_ret"].std()
sharpe = (mean_r / std_r) * np.sqrt(HOURS_PER_YEAR) if std_r != 0 else np.nan

print("\n=== BACKTEST RESULTS ===")
print(f"Total return: {total_return*100:.2f}%")
print(f"Sharpe ratio: {sharpe:.2f}")
print("Signal counts:", df["pred_signal"].value_counts().to_dict())

# ----------------------------
# PLOT EQUITY
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(df["equity"], label="GBM Strategy Equity")
plt.title("GBM-based BTC Strategy (1h horizon)")
plt.xlabel("Time")
plt.ylabel("Equity (start=1)")
plt.grid(True)
plt.legend()
plt.show()
