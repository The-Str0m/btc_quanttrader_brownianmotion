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

# ============================
# 1. DATA ACQUISITION & PREP
# ============================
print(f"Downloading historical data for {TICKER}...")
try:
    data = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    if data.empty:
        raise ValueError("No data returned from yfinance.")
    
    # Use only the Close price
    data = data[["Close"]].copy()
    data.index = pd.to_datetime(data.index)
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()


# ============================
# 2. GBM PARAMETER ESTIMATION
# ============================
# Compute log returns: $R_t = \ln(P_t / P_{t-1})$
log_ret = np.log(data["Close"] / data["Close"].shift(1)).dropna()

# Time step (fraction of a year)
dt = HORIZ_HOURS / HOURS_PER_YEAR

# Estimate GBM parameters (annualized)
mu = float(log_ret.mean() / dt)
sigma = float(log_ret.std() / np.sqrt(dt))
print(f"Estimated drift (mu): {mu:.5f} (annualized)")
print(f"Estimated volatility (sigma): {sigma:.5f} (annualized)")

# ============================
# 3. SIGNAL GENERATION (GBM EXPECTED RETURN)
# ============================
# GBM expected return over the trading horizon: $r_{\text{pred}} = \mu \cdot \Delta t$
predicted_ret = mu * dt  # expected hourly return

# Generate signal: 1 = long, -1 = short. Use .item() to extract the scalar value.
# FIX: np.where returns a 0-D array, .item() converts it to a standard Python scalar.
signal = np.where(predicted_ret > 0, 1, -1).item() 

print(f"Predicted return for {HORIZ_HOURS}h: {predicted_ret*100:.5f}%")
print(f"Trade Signal: {'LONG (1)' if signal == 1 else 'SHORT (-1)'}")


# ============================
# 4. BACKTEST EXECUTION
# ============================
df = data.copy()
# Align the data to start where the log returns calculation begins
df = df.iloc[1:].copy()
# The signal is constant across the entire backtest (static model)
df["pred_signal"] = signal

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
# Fill NaNs with 0. This handles the last incomplete period (where future_ret is NaN) 
# and ensures the cumulative product starts at 1.0.
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

print("\n=== BACKTEST RESULTS ===")
print(f"Total return: {total_return*100:.2f}%")
print(f"Sharpe ratio: {sharpe:.2f}")
print("Signal counts:", df["pred_signal"].value_counts().to_dict())

# ============================
# 6. VISUALIZATION
# ============================
plt.figure(figsize=(12, 5))
plt.plot(df["equity"], label="GBM Strategy Equity", color='tab:blue')
plt.title(f"GBM-based {TICKER} Strategy ({HORIZ_HOURS}h horizon)")
plt.xlabel("Time")
plt.ylabel("Equity (start=1)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()