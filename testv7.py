import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================
# CONFIGURATION
# ============================
TICKER = "BTC-USD"
PERIOD = "2y"
INTERVAL = "1h"
TRANSACTION_COST = 0.0005   # 0.05% per trade
HORIZ_HOURS = 1             # trading horizon
HOURS_PER_YEAR = 365 * 24
ROLLING_WINDOW = 24         # Window size for parameter estimation (e.g., last 24 hours)

# ============================
# 1. DATA ACQUISITION & PREP
# ============================
print(f"Downloading historical data for {TICKER}...")
try:
    data = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    if data.empty:
        raise ValueError("No data returned from yfinance.")
    
    df = data[["Close"]].copy()
    
    # CRITICAL FIX: Flatten the index and columns immediately to prevent merge errors
    # Convert index to simple DatetimeIndex (in case of MultiIndex)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(-1)
        
    # Convert column names to a simple Index (in case of MultiIndex columns from yfinance)
    df.columns = pd.Index(['Close'])
    df.index = pd.to_datetime(df.index)
    
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# ============================
# 2. ROLLING GBM PARAMETER ESTIMATION
# ============================
print(f"Estimating GBM parameters using a rolling {ROLLING_WINDOW}-hour window...")

# Time step (fraction of a year)
dt = HORIZ_HOURS / HOURS_PER_YEAR

# Calculate hourly log returns
log_ret = np.log(df["Close"] / df["Close"].shift(1))

# Create a temporary DataFrame for rolling metrics
rolling_metrics = pd.DataFrame(index=df.index)

# Calculate Rolling Annualized Drift (mu) and Volatility (sigma)
rolling_metrics["mu"] = log_ret.rolling(window=ROLLING_WINDOW).mean() / dt
rolling_metrics["sigma"] = log_ret.rolling(window=ROLLING_WINDOW).std() / np.sqrt(dt)

# ============================
# 3. DYNAMIC SIGNAL GENERATION & ALIGNMENT
# ============================
# Calculate Predicted Return
# GBM expected return over the trading horizon: $r_{\text{pred}} = \mu \cdot \Delta t$
rolling_metrics["predicted_ret"] = rolling_metrics["mu"] * dt

# Generate dynamic signal: 1 = long, -1 = short
rolling_metrics["pred_signal"] = np.where(rolling_metrics["predicted_ret"] > 0, 1, -1)

# Merge signal back into the main DataFrame (This should now work flawlessly)
df = df.merge(rolling_metrics[["pred_signal"]], left_index=True, right_index=True, how='left')

# Remove initial NaNs caused by the rolling window calculation
# Drop NaNs based on the signal column which contains NaNs in the first ROLLING_WINDOW - 1 rows.
df.dropna(subset=["pred_signal"], inplace=True) 

# Ensure 'pred_signal' is integer type after cleanup
df["pred_signal"] = df["pred_signal"].astype(int)

print(f"Rolling calculation started on: {df.index[0]}")
print("Strategy signal is now dynamic (switches between 1 and -1).")

# ============================
# 4. BACKTEST EXECUTION
# ============================
# Calculate the actual future return over the trading horizon
df["future_ret"] = df["Close"].pct_change(HORIZ_HOURS).shift(-HORIZ_HOURS)

# Determine trade entry points and costs
df["prev_signal"] = df["pred_signal"].shift(1).fillna(0)
# 'trade' is 1 if the signal changes (a transaction occurs)
df["trade"] = ((df["pred_signal"] != df["prev_signal"]) & (df["pred_signal"] != 0)).astype(int)
df["tcost"] = df["trade"] * TRANSACTION_COST

# Calculate strategy return: position * actual_return - transaction_cost
df["strategy_ret"] = df["pred_signal"] * df["future_ret"] - df["tcost"]

# --- FIX for NaN Propagation ---
# Fill NaNs with 0 (for the last incomplete period).
df["strategy_ret"] = df["strategy_ret"].fillna(0)

# Calculate equity curve: $(1 + r_t) \cdot (1 + r_{t-1}) \cdots$
df["equity"] = (1 + df["strategy_ret"]).cumprod()


# ============================
# 5. METRICS & RESULTS
# ============================
total_return = df["equity"].iloc[-1] - 1
mean_r = df["strategy_ret"].mean()
std_r = df["strategy_ret"].std()

# Annualized Sharpe Ratio: $\frac{\bar{R}}{\sigma_R} \cdot \sqrt{\text{HOURS\_PER\_YEAR}}$
sharpe = (mean_r / std_r) * np.sqrt(HOURS_PER_YEAR) if std_r != 0 else np.nan

# Calculate Buy and Hold benchmark return for comparison
buy_hold_equity = (1 + df["Close"].pct_change()).fillna(1).cumprod()
buy_hold_return = buy_hold_equity.iloc[-1] - 1


print("\n=== BACKTEST RESULTS ===")
print(f"Rolling Window: {ROLLING_WINDOW} hours")
print(f"Total return (Strategy): {total_return*100:.2f}%")
print(f"Total return (Buy & Hold): {buy_hold_return*100:.2f}%")
print(f"Sharpe ratio: {sharpe:.2f}")
print("Signal counts:", df["pred_signal"].value_counts().to_dict())

# ============================
# 6. VISUALIZATION
# ============================
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