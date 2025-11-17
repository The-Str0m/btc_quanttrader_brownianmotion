import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "BTC-USD"
PERIOD = "2y"
INTERVAL = "1h"
TRANSACTION_COST = 0.0005
HORIZ_HOURS = 1
HOURS_PER_YEAR = 365 * 24

data = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
if data.empty:
    raise ValueError("No data returned.")
data = data[["Close"]].copy()
data.index = pd.to_datetime(data.index)

log_ret = np.log(data["Close"] / data["Close"].shift(1)).dropna()
dt = HORIZ_HOURS / HOURS_PER_YEAR

mu = float(log_ret.mean() / dt)
sigma = float(log_ret.std() / np.sqrt(dt))
print(f"Estimated drift (mu): {mu:.5f}")
print(f"Estimated volatility (sigma): {sigma:.5f}")

predicted_ret = mu * dt
signal = np.where(predicted_ret > 0, 1, -1).item()
print(f"Predicted return (%): {predicted_ret*100:.5f}%")
print(f"Signal: {signal}")

df = data.copy()
df = df.iloc[1:].copy()
df["pred_signal"] = signal
df["future_ret"] = df["Close"].pct_change(HORIZ_HOURS).shift(-HORIZ_HOURS)
df["prev_signal"] = df["pred_signal"].shift(1).fillna(0)
df["trade"] = ((df["pred_signal"] != df["prev_signal"]) & (df["pred_signal"] != 0)).astype(int)
df["tcost"] = df["trade"] * TRANSACTION_COST
df["strategy_ret"] = df["pred_signal"] * df["future_ret"] - df["tcost"]
df["strategy_ret"] = df["strategy_ret"].fillna(0)
df["equity"] = (1 + df["strategy_ret"]).cumprod()

total_return = df["equity"].iloc[-1] - 1
mean_r = df["strategy_ret"].mean()
std_r = df["strategy_ret"].std()
sharpe = (mean_r / std_r) * np.sqrt(HOURS_PER_YEAR) if std_r != 0 else np.nan

print("\n=== BACKTEST RESULTS ===")
print(f"Total return: {total_return*100:.2f}%")
print(f"Sharpe ratio: {sharpe:.2f}")
print("Signal counts:", df["pred_signal"].value_counts().to_dict())

plt.figure(figsize=(12,5))
plt.plot(df["equity"], label="GBM Strategy Equity")
plt.title(f"GBM-based {TICKER} Strategy ({HORIZ_HOURS}h horizon)")
plt.xlabel("Time")
plt.ylabel("Equity (start=1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


'''
FINAL RESULTS

Total return: 156.58% (working percentage)
Sharpe ratio: 1.22
Signal counts: {1: 17532}
'''
