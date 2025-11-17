"""
BTC hourly daily-retrain ML backtest
- 729 days hourly BTC-USD
- target: 3-hour forward return (ternary: short / flat / long) using RET_THRESHOLD
- daily expanding-window retrain: for each test day, train on all prior data
  and tune probability threshold on last VAL_DAYS of training (validation)
- mapping: model classes (0=short,1=flat,2=long) -> probabilities used to decide trades
- trade PnL uses the same FUTURE_HORIZ_HOURS (3h) return (aligned)
"""
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta

# ----------------------------
# CONFIG
# ----------------------------
TICKER = "BTC-USD"
PERIOD = "729d"            # within Yahoo 730-day limit for 1h
INTERVAL = "1h"
TEST_DAYS = 7              # final N days to predict (daily retrain)
FUTURE_HORIZ_HOURS = 3     # target horizon and trading horizon
RET_THRESHOLD = 0.0015     # 0.15% threshold for ternary target: >thr long, < -thr short
MIN_TRAIN_BARS = 500       # skip retrain if training insufficient
VAL_DAYS = 14              # use last VAL_DAYS of train window as validation to tune threshold
PROB_GRID = np.linspace(0.51, 0.9, 40)  # grid of upper thresholds to test (lower = 1-upper)
TRANSACTION_COST = 0.0005  # roundtrip cost per trade (0.05% default)
HOURS_PER_YEAR = 365 * 24  # crypto annualization (24/7)
RANDOM_STATE = 42

# ----------------------------
# 1) DOWNLOAD
# ----------------------------
print(f"Downloading {TICKER} {PERIOD} {INTERVAL} ...")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
if df.empty:
    raise RuntimeError("No data downloaded. Check network / ticker / yfinance limits.")

df.index = pd.to_datetime(df.index)
df = df.dropna()

# Flatten MultiIndex if present: ('Close','BTC-USD') -> 'Close'
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
else:
    df.columns = list(df.columns)

# ----------------------------
# 2) FEATURES
# ----------------------------
print("Feature engineering...")
df["ret_1h"] = df["Close"].pct_change()
df["future_ret"] = df["Close"].shift(-FUTURE_HORIZ_HOURS) / df["Close"] - 1  # aligned to target

# typical price & VWAP-like (grouped by calendar day to reset)
df["typ_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
groups = df.index.normalize()
df["vwap_num"] = (df["typ_price"] * df["Volume"]).groupby(groups).cumsum()
df["vwap_den"] = df["Volume"].groupby(groups).cumsum()
df["VWAP"] = df["vwap_num"] / (df["vwap_den"] + 1e-12)

# EMAs / SMA
df["ema_9"] = ta.ema(df["Close"], length=9)
df["ema_21"] = ta.ema(df["Close"], length=21)
df["ema_diff_9_21"] = df["ema_9"] - df["ema_21"]
df["sma_20"] = ta.sma(df["Close"], length=20)
df["dist_close_sma20"] = (df["Close"] - df["sma_20"]) / (df["sma_20"] + 1e-12)

# MACD
macd = ta.macd(df["Close"])
if macd is not None and not macd.empty:
    df["macd"] = macd.iloc[:,0]
    if macd.shape[1] > 1: df["macd_sig"] = macd.iloc[:,1]
    if macd.shape[1] > 2: df["macd_hist"] = macd.iloc[:,2]

# Momentum
df["rsi_14"] = ta.rsi(df["Close"], length=14)
stoch = ta.stoch(df["High"], df["Low"], df["Close"])
if stoch is not None and not stoch.empty:
    k_candidates = [c for c in stoch.columns if "k" in c.lower()]
    d_candidates = [c for c in stoch.columns if "d" in c.lower()]
    df["stoch_k"] = stoch[k_candidates[0]] if k_candidates else stoch.iloc[:,0]
    if stoch.shape[1] > 1:
        df["stoch_d"] = stoch[d_candidates[0]] if d_candidates else stoch.iloc[:,1]

# Volatility and Bollinger
df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
df["vol_20"] = df["ret_1h"].rolling(20).std()
bb = ta.bbands(df["Close"], length=20, std=2)
if bb is not None and not bb.empty:
    df["bb_upper"] = bb.iloc[:,0]; df["bb_mid"] = bb.iloc[:,1]; df["bb_lower"] = bb.iloc[:,2]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-12)

# Volume-based
df["obv"] = ta.obv(df["Close"], df["Volume"])
df["vol_z"] = (df["Volume"] - df["Volume"].rolling(50).mean()) / (df["Volume"].rolling(50).std() + 1e-12)

# Price action
df["body"] = (df["Close"] - df["Open"]).abs()
df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]
df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-12)

# lagged returns
df["ret_2h"] = df["Close"].pct_change(2)
df["ret_3h"] = df["Close"].pct_change(3)
df["ret_6h"] = df["Close"].pct_change(6)

# zscore
df["zscore_20"] = (df["Close"] - df["Close"].rolling(20).mean()) / (df["Close"].rolling(20).std() + 1e-12)

# drop NaNs created by indicators
df = df.dropna()
df = df.sort_index()

# ----------------------------
# 3) TERNARY TARGET (short= -1, flat=0, long=1) FOR TRAINING
# ----------------------------
df["Target_raw"] = df["future_ret"]
def make_ternary(x, thr):
    if x > thr: return 1
    if x < -thr: return -1
    return 0

df["Target"] = df["Target_raw"].apply(lambda x: make_ternary(x, RET_THRESHOLD))
# map to model labels: short=0, flat=1, long=2
map_to_model = {-1:0, 0:1, 1:2}
df["Target_m"] = df["Target"].map(map_to_model)

# drop NAs
df = df.dropna()
if df.empty:
    raise RuntimeError("No rows remain after feature/target engineering.")

# features list (keep existing)
features = [
    "VWAP","ema_diff_9_21","sma_20","dist_close_sma20",
    "macd","macd_sig","macd_hist",
    "rsi_14","stoch_k","stoch_d",
    "atr_14","vol_20","bb_width",
    "obv","vol_z",
    "body","upper_wick","lower_wick","range_pct",
    "ret_1h","ret_2h","ret_3h","ret_6h","zscore_20"
]
features = [f for f in features if f in df.columns]
print("Features used:", features)

# ----------------------------
# 4) DAILY RETRAINING (EXPANDING WINDOW) ON LAST TEST_DAYS
# ----------------------------
unique_dates = sorted(df.index.date)
if len(unique_dates) < TEST_DAYS + 2:
    raise RuntimeError("Not enough unique days in data for test window.")

test_days = unique_dates[-TEST_DAYS:]
print("Test days:", test_days)

# storage for predictions
pred_rows = []

# model params for multiclass
model_params = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=3,
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

for day in test_days:
    # training = all rows with date < day
    train_mask = df.index.date < day
    test_mask = df.index.date == day

    X_train_all = df.loc[train_mask, features]
    y_train_all = df.loc[train_mask, "Target_m"]

    X_test = df.loc[test_mask, features]
    y_test_true = df.loc[test_mask, "Target"]         # -1/0/1 raw for eval
    future_ret_test = df.loc[test_mask, "future_ret"] # aligned 3h returns to compute PnL

    if len(X_train_all) < MIN_TRAIN_BARS or X_test.empty:
        print(f"Skipping {day}: train bars={len(X_train_all)}, test bars={len(X_test)}")
        continue

    # Validation split: last VAL_DAYS of the training period
    train_dates = sorted(df.loc[train_mask].index.date)
    val_cut_date = train_dates[-VAL_DAYS] if len(train_dates) > VAL_DAYS else train_dates[0]
    val_mask = (df.index.date >= val_cut_date) & (df.index.date < day)
    train_mask2 = (df.index.date < val_cut_date)

    X_train = df.loc[train_mask2, features]
    y_train = df.loc[train_mask2, "Target_m"]
    X_val = df.loc[val_mask, features]
    y_val_true = df.loc[val_mask, "Target"]           # -1/0/1
    future_ret_val = df.loc[val_mask, "future_ret"]   # 3h returns for validation

    # If val partition too small, fall back to a time-based last N rows (e.g., last 200 rows)
    if X_val.shape[0] < 50:
        last_n = 200
        X_train = df.loc[train_mask, features].iloc[:-last_n]
        y_train = df.loc[train_mask, "Target_m"].iloc[:-last_n]
        X_val = df.loc[train_mask, features].iloc[-last_n:]
        y_val_true = df.loc[train_mask, "Target"].iloc[-last_n:]
        future_ret_val = df.loc[train_mask, "future_ret"].iloc[-last_n:]

    # Train multiclass model on X_train
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    # Predict probabilities on validation, tune threshold grid to maximize Sharpe on validation
    prob_val = model.predict_proba(X_val)  # columns = [short, flat, long]
    best_sharpe = -np.inf
    best_u = None

    for u in PROB_GRID:
        # build signals on validation using threshold u
        # long if prob_long > u, short if prob_short > u, else flat
        prob_short = prob_val[:, 0]
        prob_long  = prob_val[:, 2]

        sig_val = np.zeros_like(prob_long, dtype=int)
        # if both exceed u, pick the higher-prob class
        both_long = prob_long > u
        both_short = prob_short > u
        # naive priority: if both true choose np.sign(prob_long - prob_short)
        # implement by choosing max probability
        for i in range(len(sig_val)):
            ps = prob_short[i]; pl = prob_long[i]
            if pl > u and pl >= ps:
                sig_val[i] = 1
            elif ps > u and ps > pl:
                sig_val[i] = -1
            else:
                sig_val[i] = 0

        # compute val returns using future_ret_val (aligned 3h returns)
        ret_series = sig_val * future_ret_val.values
        if ret_series.std() == 0 or np.isnan(ret_series.std()):
            sh = -np.inf
        else:
            sh = (ret_series.mean() / (ret_series.std())) * np.sqrt(HOURS_PER_YEAR)
        if sh > best_sharpe:
            best_sharpe = sh
            best_u = u

    if best_u is None:
        best_u = 0.6  # fallback
    # print chosen threshold for this day
    print(f"{day} -> tuned threshold u = {best_u:.3f}, val_sharpe = {best_sharpe:.3f}")

    # Predict on test day
    prob_test = model.predict_proba(X_test)
    prob_short_t = prob_test[:, 0]
    prob_long_t = prob_test[:, 2]

    # Build signals using best_u
    sig_test = np.zeros_like(prob_long_t, dtype=int)
    for i in range(len(sig_test)):
        ps = prob_short_t[i]; pl = prob_long_t[i]
        if pl > best_u and pl >= ps:
            sig_test[i] = 1
        elif ps > best_u and ps > pl:
            sig_test[i] = -1
        else:
            sig_test[i] = 0

    # record rows
    for ts, s, p_up, p_short, true_lbl, fr in zip(X_test.index, sig_test, prob_long_t, prob_short_t, y_test_true, future_ret_test):
        pred_rows.append({
            "timestamp": ts, "signal": int(s), "prob_long": float(p_up), "prob_short": float(p_short),
            "true": int(true_lbl), "future_ret": float(fr), "tuned_u": float(best_u)
        })

# assemble prediction DF
pred_df = pd.DataFrame(pred_rows).set_index("timestamp").sort_index()
if pred_df.empty:
    raise RuntimeError("No predictions were generated. Possibly not enough data or TEST_DAYS too large.")

# ----------------------------
# 5) BACKTEST using aligned FUTURE_HORIZ_HOURS returns
# ----------------------------
backtest = pred_df.merge(df[["future_ret"]], left_index=True, right_index=True, how="left")
# ensure future_ret from pred_df is used (they should match)
backtest["future_ret"] = backtest["future_ret_x"].fillna(backtest["future_ret_y"])

# prev signal for trade count
backtest["prev_signal"] = backtest["signal"].shift(1).fillna(0)
backtest["trade"] = ((backtest["signal"] != backtest["prev_signal"]) & (backtest["signal"] != 0)).astype(int)
backtest["tcost"] = backtest["trade"] * TRANSACTION_COST

# strategy return uses aligned future_ret (3h)
backtest["strategy_ret_raw"] = backtest["signal"] * backtest["future_ret"]
backtest["strategy_ret"] = backtest["strategy_ret_raw"] - backtest["tcost"]
backtest["equity"] = (1 + backtest["strategy_ret"]).cumprod()

# metrics
total_return = backtest["equity"].iloc[-1] - 1
mean_ret = backtest["strategy_ret"].mean()
std_ret = backtest["strategy_ret"].std()
sharpe = (mean_ret / std_ret) * np.sqrt(HOURS_PER_YEAR) if std_ret != 0 else np.nan

# classification metrics (raw label based)
try:
    # compute accuracy of model label (long vs rest) on predicted rows using prob>0.5 raw label
    raw_label = (backtest["prob_long"] > 0.5).astype(int)
    # true_up: map true ternary to up(1)/not-up(0)
    true_up = (backtest["true"] == 1).astype(int)
    accuracy = (raw_label == true_up).mean()
    auc = roc_auc_score(true_up, backtest["prob_long"])
except Exception:
    accuracy = np.nan; auc = np.nan

print("\n=== BACKTEST SUMMARY (daily retrain, ternary target, tuned thresholds) ===")
print(f"Test days: {test_days[0]} -> {test_days[-1]}  (predicted bars: {len(backtest)})")
print(f"Total return (test window): {total_return*100:.2f}%")
print(f"Annualized Sharpe (hourly basis, 24/7): {sharpe:.2f}")
print(f"Accuracy (raw prob>0.5 vs up): {accuracy:.4f}")
print(f"AUC (prob_long): {auc:.4f}")
print("Signal counts:", backtest["signal"].value_counts().to_dict())
print("\nFirst prediction rows:\n", pred_df.head())

# ----------------------------
# 6) PLOT + SAVE
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(backtest["equity"], label="Strategy Equity")
plt.title(f"{TICKER} - daily retrain (expanding), ternary target, tuned threshold (3h horizon)")
plt.xlabel("Time")
plt.ylabel("Equity (start=1)")
plt.grid(True)
plt.legend()
plt.show()

backtest.to_csv("btc_daily_retrain_ternary_tuned_backtest.csv")
print("Saved backtest -> btc_daily_retrain_ternary_tuned_backtest.csv")
