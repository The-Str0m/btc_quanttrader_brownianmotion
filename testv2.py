"""
BTC daily-retrain ML backtest with probability calibration + EV filter (rolling 729d, 1h horizon)

Key ideas implemented:
- Train multiclass XGBoost (short/flat/long) on past 729 days (rolling window), retrain each day
- Use a validation slice inside the train window to:
    * calibrate raw model probabilities (Isotonic) for long and short
    * estimate expected future return as a function of calibrated prob (via bin averages)
- For each test-bar compute EV_long = p_long_cal * EF_long(p_long_cal)
  and EV_short = p_short_cal * (-EF_short(p_short_cal)).
- Enter long if EV_long > TRANSACTION_COST + MIN_EV and EV_long > EV_short.
  Enter short similarly. Otherwise flat.
- Use aligned FUTURE_HORIZ_HOURS returns to compute PnL.
- Save per-bar predictions and a per-day summary CSV.
"""
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm
import os

# ----------------------------
# CONFIG
# ----------------------------
TICKER = "BTC-USD"
PERIOD = "729d"
INTERVAL = "1h"
TRAIN_WINDOW_DAYS = 729
TEST_DAYS = 21
FUTURE_HORIZ_HOURS = 1     # aligned trade horizon
RET_THRESHOLD = 0.0008     # ternary target threshold (0.08%)
VAL_DAYS = 14              # validation slice for calibration & EF estimation
MIN_TRAIN_BARS = 500
N_BINS = 10                # bins for EF estimation
TRANSACTION_COST = 0.0005  # roundtrip cost (0.05%)
MIN_EV = 0.0               # minimum EV above costs to trade
HOURS_PER_YEAR = 365 * 24
RANDOM_STATE = 42
VERBOSE = True
OUT_CSV = "btc_ev_calibrated_backtest.csv"
SUMMARY_CSV = "btc_ev_calibrated_summary.csv"

# ----------------------------
# 1) DOWNLOAD & PREP
# ----------------------------
print(f"Downloading {TICKER} {PERIOD} {INTERVAL} ...")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
if df.empty:
    raise RuntimeError("No data downloaded. Check network / ticker / yfinance limits.")

# normalize index and tz
df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
else:
    df.index = df.index.tz_convert("UTC")

# flatten columns if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

# ----------------------------
# 2) FEATURES & TARGET
# ----------------------------
print("Engineering features...")
df["ret_1h"] = df["Close"].pct_change()
df["future_ret"] = df["Close"].shift(-FUTURE_HORIZ_HOURS) / df["Close"] - 1

# Typical price & VWAP (cumulative ok for crypto)
df["typ_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
df["vwap_num"] = (df["typ_price"] * df["Volume"]).cumsum()
df["vwap_den"] = df["Volume"].cumsum()
df["VWAP"] = df["vwap_num"] / (df["vwap_den"] + 1e-12)

# moving averages & diffs
df["sma_20"] = df["Close"].rolling(20).mean()
df["ema_9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["ema_diff_9_21"] = df["ema_9"] - df["ema_21"]
df["dist_close_sma20"] = (df["Close"] - df["sma_20"]) / (df["sma_20"] + 1e-12)

# MACD, RSI, Stoch, ATR, BB width robustly
macd = ta.macd(df["Close"])
if macd is not None and not macd.empty:
    df["macd"] = macd.iloc[:, 0]
    if macd.shape[1] > 1: df["macd_sig"] = macd.iloc[:, 1]
    if macd.shape[1] > 2: df["macd_hist"] = macd.iloc[:, 2]

df["rsi_14"] = ta.rsi(df["Close"], length=14)
stoch = ta.stoch(df["High"], df["Low"], df["Close"])
if stoch is not None and not stoch.empty:
    stoch_cols = [c.lower() for c in stoch.columns]
    k_idx = next((i for i, c in enumerate(stoch_cols) if "k" in c), 0)
    d_idx = next((i for i, c in enumerate(stoch_cols) if "d" in c and i != k_idx), (1 if stoch.shape[1] > 1 else 0))
    df["stoch_k"] = stoch.iloc[:, k_idx]
    if stoch.shape[1] > 1:
        df["stoch_d"] = stoch.iloc[:, d_idx]

df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
bb = ta.bbands(df["Close"], length=20, std=2)
if bb is not None and not bb.empty:
    df["bb_upper"] = bb.iloc[:, 0]; df["bb_mid"] = bb.iloc[:, 1]; df["bb_lower"] = bb.iloc[:, 2]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-12)

# volume + price action
df["obv"] = ta.obv(df["Close"], df["Volume"]) if "Volume" in df.columns else 0.0
df["vol_20"] = df["ret_1h"].rolling(20).std()
df["vol_z"] = (df["Volume"] - df["Volume"].rolling(50).mean()) / (df["Volume"].rolling(50).std() + 1e-12)
df["body"] = df["Close"] - df["Open"]
df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-12)
df["ret_2h"] = df["Close"].pct_change(2)
df["ret_3h"] = df["Close"].pct_change(3)
df["ret_6h"] = df["Close"].pct_change(6)
df["zscore_20"] = (df["Close"] - df["Close"].rolling(20).mean()) / (df["Close"].rolling(20).std() + 1e-12)

df = df.dropna().sort_index()
print("Rows after features:", len(df))

# ternary target
def to_tern(ret, thr):
    if ret > thr: return 1
    if ret < -thr: return -1
    return 0

df["Target"] = df["future_ret"].apply(lambda x: to_tern(x, RET_THRESHOLD))
# model labels: short=0, flat=1, long=2
map_to_model = {-1:0, 0:1, 1:2}
df["Target_m"] = df["Target"].map(map_to_model)

df = df.dropna()
if df.empty:
    raise RuntimeError("No rows remain after engineering.")

features = [
    "VWAP","ema_diff_9_21","sma_20","dist_close_sma20",
    "macd","macd_sig","macd_hist","rsi_14","stoch_k","stoch_d",
    "atr_14","vol_20","bb_width","obv","vol_z",
    "body","upper_wick","lower_wick","range_pct",
    "ret_1h","ret_2h","ret_3h","ret_6h","zscore_20"
]
features = [f for f in features if f in df.columns]
print("Features used:", features)

# ----------------------------
# 3) ROLLING DAILY RETRAIN + CALIBRATION + EV FILTER
# ----------------------------
unique_dates = sorted({t.date() for t in df.index})
test_days = unique_dates[-TEST_DAYS:]
print("Test days:", test_days)

pred_rows = []
summary_rows = []

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
    day_start = pd.Timestamp(day).tz_localize("UTC")
    train_start = day_start - pd.Timedelta(days=TRAIN_WINDOW_DAYS)

    train_mask = (df.index >= train_start) & (df.index.date < day)
    test_mask = df.index.date == day

    X_train_all = df.loc[train_mask, features]
    y_train_all = df.loc[train_mask, "Target_m"]

    X_test = df.loc[test_mask, features]
    y_test_raw = df.loc[test_mask, "Target"]
    future_ret_test = df.loc[test_mask, "future_ret"]

    if len(X_train_all) < MIN_TRAIN_BARS or X_test.empty:
        if VERBOSE:
            print(f"Skipping {day}: train_bars={len(X_train_all)}, test_bars={len(X_test)}")
        continue

    # Validation split by last VAL_DAYS of train window
    train_dates = sorted({t.date() for t in df.loc[train_mask].index})
    if len(train_dates) > VAL_DAYS:
        val_cut_date = train_dates[-VAL_DAYS]
        val_cut_ts = pd.Timestamp(val_cut_date).tz_localize("UTC")
        val_mask = (df.index >= val_cut_ts) & (df.index.date < day)
        train_mask2 = (df.index >= train_start) & (df.index < val_cut_ts)
    else:
        # fallback: last N rows
        N = min(800, max(200, int(len(X_train_all) * 0.2)))
        idx_all = df.loc[train_mask].index
        train_mask2 = df.index.isin(idx_all[:-N]) if len(idx_all) > N else df.index.isin([])
        val_mask = df.index.isin(idx_all[-N:]) if len(idx_all) > N else df.index.isin([])

    X_train = df.loc[train_mask2, features]
    y_train = df.loc[train_mask2, "Target_m"]
    X_val = df.loc[val_mask, features]
    y_val_raw = df.loc[val_mask, "Target"]
    future_ret_val = df.loc[val_mask, "future_ret"]

    # Train multiclass model on X_train
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    # Predict probs on validation
    prob_val = model.predict_proba(X_val)  # cols: short, flat, long
    prob_short_val = prob_val[:, 0]
    prob_flat_val  = prob_val[:, 1]
    prob_long_val  = prob_val[:, 2]

    # Calibrate probabilities with Isotonic on validation: map raw -> calibrated
    # For long: true label = (y_val_raw == 1)
    try:
        iso_long = IsotonicRegression(out_of_bounds="clip")
        iso_long.fit(prob_long_val, (y_val_raw == 1).astype(int))
    except Exception:
        iso_long = None

    try:
        iso_short = IsotonicRegression(out_of_bounds="clip")
        iso_short.fit(prob_short_val, (y_val_raw == -1).astype(int))
    except Exception:
        iso_short = None

    # Estimate Expected Future Return as function of calibrated prob using VAL bins
    # Use calibrated probs on val (or raw if iso failed)
    if iso_long is not None:
        prob_long_val_cal = iso_long.predict(prob_long_val)
    else:
        prob_long_val_cal = prob_long_val

    if iso_short is not None:
        prob_short_val_cal = iso_short.predict(prob_short_val)
    else:
        prob_short_val_cal = prob_short_val

    # bin and compute mean future_ret per prob bin
    if len(prob_long_val_cal) >= N_BINS:
        bins = np.linspace(0.0, 1.0, N_BINS + 1)
        bin_idx = np.digitize(prob_long_val_cal, bins) - 1
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        ef_long = np.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            mask_i = (bin_idx == i)
            if mask_i.sum() > 0:
                ef_long[i] = future_ret_val[mask_i].mean()
            else:
                ef_long[i] = 0.0
    else:
        # fallback linear regression slope
        ef_long = None

    if len(prob_short_val_cal) >= N_BINS:
        bins_s = np.linspace(0.0, 1.0, N_BINS + 1)
        bin_idx_s = np.digitize(prob_short_val_cal, bins_s) - 1
        bin_centers_s = (bins_s[:-1] + bins_s[1:]) / 2.0
        ef_short = np.zeros(len(bin_centers_s))
        for i in range(len(bin_centers_s)):
            mask_i = (bin_idx_s == i)
            if mask_i.sum() > 0:
                ef_short[i] = future_ret_val[mask_i].mean()
            else:
                ef_short[i] = 0.0
    else:
        ef_short = None

    # Predict on test day
    prob_test = model.predict_proba(X_test)
    prob_short_t = prob_test[:, 0]
    prob_long_t = prob_test[:, 2]

    # Calibrate test probs
    if iso_long is not None:
        prob_long_t_cal = iso_long.predict(prob_long_t)
    else:
        prob_long_t_cal = prob_long_t

    if iso_short is not None:
        prob_short_t_cal = iso_short.predict(prob_short_t)
    else:
        prob_short_t_cal = prob_short_t

    # Map calibrated probs -> EF via interpolation on validation bins
    def map_ef(prob_cal, bin_centers, ef_vals):
        if ef_vals is None:
            # fallback: assume EF = mean(future_ret_val) * (prob_cal - 0.5)*2  (weak linear proxy)
            m = future_ret_val.mean() if len(future_ret_val) > 0 else 0.0
            return m * (prob_cal - 0.5) * 2.0
        else:
            # monotonic interp between 0..1
            return np.interp(prob_cal, bin_centers, ef_vals)

    if ef_long is not None:
        ef_long_vals = ef_long
        centers_long = bin_centers
    else:
        ef_long_vals = None
        centers_long = None

    if ef_short is not None:
        ef_short_vals = ef_short
        centers_short = bin_centers_s
    else:
        ef_short_vals = None
        centers_short = None

    ef_long_test = map_ef(prob_long_t_cal, centers_long, ef_long_vals)
    ef_short_test = map_ef(prob_short_t_cal, centers_short, ef_short_vals)

    # EV computations
    ev_long = prob_long_t_cal * ef_long_test
    ev_short = prob_short_t_cal * (-ef_short_test)  # if future_ret negative on avg, this becomes positive

    # Decide signals using EV and transaction cost
    sigs = np.zeros_like(ev_long, dtype=int)
    for i in range(len(sigs)):
        if ev_long[i] > TRANSACTION_COST + MIN_EV and ev_long[i] > ev_short[i]:
            sigs[i] = 1
        elif ev_short[i] > TRANSACTION_COST + MIN_EV and ev_short[i] > ev_long[i]:
            sigs[i] = -1
        else:
            sigs[i] = 0

    # record predictions
    for ts, s, pl, ps, plc, psc, efl, efs in zip(
        X_test.index, sigs, prob_long_t, prob_short_t, prob_long_t_cal, prob_short_t_cal, ef_long_test, ef_short_test
    ):
        pred_rows.append({
            "timestamp": ts,
            "signal": int(s),
            "prob_long_raw": float(pl),
            "prob_short_raw": float(ps),
            "prob_long_cal": float(plc),
            "prob_short_cal": float(psc),
            "ef_long": float(efl),
            "ef_short": float(efs),
            "future_ret": float(df.at[ts, "future_ret"]),
            "true": int(df.at[ts, "Target"])
        })

    # summary row per day
    n_trades = int((sigs != 0).sum())
    avg_ev_long = np.nanmean(ev_long) if len(ev_long) > 0 else np.nan
    avg_ev_short = np.nanmean(ev_short) if len(ev_short) > 0 else np.nan
    summary_rows.append({
        "day": str(day),
        "train_bars": len(X_train_all),
        "val_bars": len(X_val),
        "test_bars": len(X_test),
        "chosen_min_ev": MIN_EV,
        "avg_ev_long": avg_ev_long,
        "avg_ev_short": avg_ev_short,
        "n_trades": n_trades
    })

# ----------------------------
# 4) BACKTEST
# ----------------------------
pred_df = pd.DataFrame(pred_rows).set_index("timestamp").sort_index()
if pred_df.empty:
    raise RuntimeError("No predictions generated. Try lowering MIN_TRAIN_BARS or TEST_DAYS.")

# Align returns & compute PnL -> use aligned future_ret
pred_df["future_ret_aligned"] = pred_df["future_ret"]
pred_df["prev_signal"] = pred_df["signal"].shift(1).fillna(0)
pred_df["trade"] = ((pred_df["signal"] != pred_df["prev_signal"]) & (pred_df["signal"] != 0)).astype(int)
pred_df["tcost"] = pred_df["trade"] * TRANSACTION_COST

pred_df["strategy_ret_raw"] = pred_df["signal"] * pred_df["future_ret_aligned"]
pred_df["strategy_ret"] = pred_df["strategy_ret_raw"] - pred_df["tcost"]
pred_df["equity"] = (1 + pred_df["strategy_ret"]).cumprod()

# metrics
total_return = pred_df["equity"].iloc[-1] - 1
mean_r = pred_df["strategy_ret"].mean()
std_r = pred_df["strategy_ret"].std()
sharpe = (mean_r / std_r) * np.sqrt(HOURS_PER_YEAR) if std_r != 0 else np.nan

# classification diagnostics for long vs rest
try:
    y_true_up = (pred_df["true"] == 1).astype(int)
    y_prob_long = pred_df["prob_long_cal"]
    acc = accuracy_score(y_true_up, (y_prob_long > 0.5).astype(int))
    auc = roc_auc_score(y_true_up, y_prob_long)
except Exception:
    acc = np.nan; auc = np.nan

print("\n=== BACKTEST RESULTS (EV-filtered) ===")
print(f"Predicted bars: {len(pred_df)}")
print(f"Total return: {total_return*100:.2f}%")
print(f"Annualized Sharpe (24/7): {sharpe:.2f}")
print(f"Accuracy (raw>0.5): {acc:.4f}")
print(f"AUC (prob_long calibrated): {auc:.4f}")
print("Signal counts:", pred_df["signal"].value_counts().to_dict())

# Save outputs
pred_df.to_csv(OUT_CSV)
pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
print(f"Saved predictions -> {OUT_CSV}")
print(f"Saved daily summary -> {SUMMARY_CSV}")

# Plot equity
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(pred_df["equity"], label="Equity (EV-filtered)")
plt.title("BTC EV-filtered strategy (calibrated probabilities, rolling-729d daily retrain)")
plt.xlabel("Time")
plt.ylabel("Equity (start=1)")
plt.grid(True)
plt.legend()
plt.show()
