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
ROLLING_WINDOW = 24

data = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
if data.empty:
    raise ValueError("No data returned.")

df = data[["Close"]].copy()
if isinstance(df.index, pd.MultiIndex):
    df.index = df.index.get_level_values(-1)
df.columns = pd.Index(['Close'])
df.index = pd.to_datetime(df.index)

dt = HORIZ_HOURS / HOURS_PER_YEAR
log_ret = np.log(df["Close"] / df["Close"].shift(1))

rolling_metrics = pd.DataFrame(index=df.index)
rolling_metrics["mu"] = log_ret.rolling(window=ROLLING_WINDOW).mean() / dt
rolling_metrics["sigma"] = log_ret.rolling(window=ROLLING_WINDOW).std() / np.sqrt(dt)
rolling_metrics["predicted_ret"] = rolling_metrics["mu"] * dt
rolling_metrics["pred_signal"] = np.where(rolling_metrics["predicted_ret"] > 0, 1, -1)

df = df.merge(rolling_metrics[["pred_signal"]], left_index=True, right_index=True, how='left')
df.dropna(subset=["pred_signal"], inplace=True)
df["pred_signal"] = df["pred_signal"].astype(int)

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

buy_hold_equity = (1 + df["Close"].pct_change()).fillna(1).cumprod()
buy_hold_return = buy_hold_equity.iloc[-1] - 1

print("\n=== BACKTEST RESULTS ===")
print(f"Rolling Window: {ROLLING_WINDOW} hours")
print(f"Total return (Strategy): {total_return*100:.2f}%")
print(f"Total return (Buy & Hold): {buy_hold_return*100:.2f}%")
print(f"Sharpe ratio: {sharpe:.2f}")
print("Signal counts:", df["pred_signal"].value_counts().to_dict())

plt.figure(figsize=(12, 5))
plt.plot(df["equity"], label=f"GBM Strategy Equity ({ROLLING_WINDOW}h rolling)")
plt.plot(buy_hold_equity, label="Buy & Hold Benchmark", linestyle='--', alpha=0.7)
plt.title(f"Dynamic GBM-based {TICKER} Strategy ({HORIZ_HOURS}h horizon)")
plt.xlabel("Time")
plt.ylabel("Equity (start=1)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


'''
FINAL RESULTS (loses money)

Rolling Window: 24 hours
Total return (Strategy): -72.11%
Total return (Buy & Hold): 156.50%
Sharpe ratio: -1.09
Signal counts: {1: 9256, -1: 8277} 
'''
