import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt

TICKER = "BTC-USD"
PERIOD = "729d"
INTERVAL = "1h"

TRAIN_WINDOW_DAYS = 729
TEST_DAYS = 21
FUTURE_HORIZ_HOURS = 4
VAL_DAYS = 60
MIN_TRAIN_BARS = 800
N_BINS = 20
EWMA_SPAN = 3
THR_GRID = np.linspace(0.0008, 0.0045, 20)
TRANSACTION_COST = 0.0005
HOURS_PER_YEAR = 365 * 24
RANDOM_STATE = 42

OUT_PRED_CSV = "btc_4h_regression_ev_predictions.csv"
OUT_SUMMARY_CSV = "btc_4h_regression_ev_summary.csv"

print(f"Downloading {TICKER} {PERIOD} {INTERVAL} ...")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
if df.empty:
    raise RuntimeError("No data downloaded. Check network / ticker / yfinance limits.")

df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
else:
    df.index = df.index.tz_convert("UTC")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

df["ret_1h"] = df["Close"].pct_change()
df["future_ret"] = df["Close"].shift(-FUTURE_HORIZ_HOURS) / df["Close"] - 1

df["typ_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
df["vwap_num"] = (df["typ_price"] * df["Volume"]).cumsum()
df["vwap_den"] = df["Volume"].cumsum()
df["VWAP"] = df["vwap_num"] / (df["vwap_den"] + 1e-12)

df["sma_20"] = df["Close"].rolling(20).mean()
df["ema_9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["ema_diff_9_21"] = df["ema_9"] - df["ema_21"]
df["dist_close_sma20"] = (df["Close"] - df["sma_20"]) / (df["sma_20"] + 1e-12)

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
    if stoch.shape[1] > 1: df["stoch_d"] = stoch.iloc[:, d_idx]

df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
bb = ta.bbands(df["Close"], length=20, std=2)
if bb is not None and not bb.empty:
    df["bb_upper"] = bb.iloc[:, 0]; df["bb_mid"] = bb.iloc[:, 1]; df["bb_lower"] = bb.iloc[:, 2]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-12)

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

FEATURES = [
    "VWAP", "ema_diff_9_21", "sma_20", "dist_close_sma20",
    "macd", "macd_sig", "macd_hist", "rsi_14", "stoch_k", "stoch_d",
    "atr_14", "vol_20", "bb_width", "obv", "vol_z",
    "body", "upper_wick", "lower_wick", "range_pct",
    "ret_1h", "ret_2h", "ret_3h", "ret_6h", "zscore_20"
]
FEATURES = [f for f in FEATURES if f in df.columns]
print("Using features:", len(FEATURES))

unique_dates = sorted({t.date() for t in df.index})
test_days = unique_dates[-TEST_DAYS:]
print("Test days:", test_days)

pred_rows = []
summary_rows = []

reg_params = dict(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

for day in test_days:
    day_start = pd.Timestamp(day).tz_localize("UTC")
    train_start = day_start - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    train_mask = (df.index >= train_start) & (df.index.date < day)
    test_mask = df.index.date == day

    X_train_all = df.loc[train_mask, FEATURES]
    y_train_all = df.loc[train_mask, "future_ret"]

    X_test = df.loc[test_mask, FEATURES]
    y_test_true = df.loc[test_mask, "future_ret"]

    if len(X_train_all) < MIN_TRAIN_BARS or X_test.empty:
        print(f"Skipping {day}: train_bars={len(X_train_all)}, test_bars={len(X_test)}")
        continue

    train_dates = sorted({t.date() for t in df.loc[train_mask].index})
    if len(train_dates) > VAL_DAYS:
        val_cut_date = train_dates[-VAL_DAYS]
        val_cut_ts = pd.Timestamp(val_cut_date).tz_localize("UTC")
        val_mask = (df.index >= val_cut_ts) & (df.index.date < day)
        train_mask2 = (df.index >= train_start) & (df.index < val_cut_ts)
    else:
        N = min(1000, max(300, int(len(X_train_all) * 0.2)))
        idx_all = df.loc[train_mask].index
        train_mask2 = df.index.isin(idx_all[:-N]) if len(idx_all) > N else df.index.isin([])
        val_mask = df.index.isin(idx_all[-N:]) if len(idx_all) > N else df.index.isin([])

    X_train = df.loc[train_mask2, FEATURES]
    y_train = df.loc[train_mask2, "future_ret"]
    X_val = df.loc[val_mask, FEATURES]
    y_val = df.loc[val_mask, "future_ret"]

    model = XGBRegressor(**reg_params)
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    if len(pred_val) >= N_BINS:
        edges = np.percentile(pred_val, np.linspace(0, 100, N_BINS + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            bin_centers = np.array([pred_val.mean()])
            eff = np.array([y_val.mean()])
        else:
            bin_idx = np.digitize(pred_val, edges) - 1
            centers = []
            eff_list = []
            for i in range(len(edges) - 1):
                mask_i = (bin_idx == i)
                if mask_i.sum() > 0:
                    centers.append(pred_val[mask_i].mean())
                    eff_list.append(y_val[mask_i].mean())
            bin_centers = np.array(centers)
            eff = np.array(eff_list)
    else:
        bin_centers = None
        eff = None

    def map_cal(preds):
        if bin_centers is None or len(bin_centers) < 2:
            if len(pred_val) >= 10:
                slope = np.cov(pred_val, y_val)[0, 1] / (np.var(pred_val) + 1e-12)
                intercept = y_val.mean() - slope * pred_val.mean()
                return preds * slope + intercept
            else:
                return preds
        return np.interp(preds, bin_centers, eff)

    pred_val_cal = map_cal(pred_val)
    smooth_val = pd.Series(pred_val_cal).ewm(span=EWMA_SPAN, adjust=False).mean().values

    best_sh = -np.inf
    best_thr = None
    for thr in THR_GRID:
        sig_val = np.zeros_like(smooth_val)
        sig_val[smooth_val > thr] = 1
        sig_val[smooth_val < -thr] = -1
        ret_series = sig_val * y_val.values
        if ret_series.std() == 0 or np.isnan(ret_series.std()):
            sh = -np.inf
        else:
            sh = (ret_series.mean() / ret_series.std()) * np.sqrt(HOURS_PER_YEAR)
        if sh > best_sh:
            best_sh = sh
            best_thr = thr
    if best_thr is None:
        best_thr = THR_GRID[len(THR_GRID) // 2]

    pred_test = model.predict(X_test)
    pred_test_cal = map_cal(pred_test)
    smooth_test = pd.Series(pred_test_cal).ewm(span=EWMA_SPAN, adjust=False).mean().values

    sig_test = np.zeros_like(smooth_test, dtype=int)
    sig_test[smooth_test > best_thr] = 1
    sig_test[smooth_test < -best_thr] = -1

    for ts, raw_p, cal_p, smooth_p, s, fut in zip(X_test.index, pred_test, pred_test_cal, smooth_test, sig_test, y_test_true):
        pred_rows.append({
            "timestamp": ts,
            "pred_raw": float(raw_p),
            "pred_cal": float(cal_p),
            "pred_smooth": float(smooth_p),
            "signal": int(s),
            "future_ret": float(fut),
            "chosen_thr": float(best_thr),
            "val_sharpe": float(best_sh)
        })

    summary_rows.append({
        "day": str(day),
        "train_bars": len(X_train_all),
        "val_bars": len(X_val),
        "test_bars": len(X_test),
        "chosen_thr": float(best_thr),
        "val_sharpe": float(best_sh)
    })

pred_df = pd.DataFrame(pred_rows).set_index("timestamp").sort_index()
if pred_df.empty:
    raise RuntimeError("No predictions generated — adjust TEST_DAYS or MIN_TRAIN_BARS.")

pred_df["prev_signal"] = pred_df["signal"].shift(1).fillna(0)
pred_df["trade"] = ((pred_df["signal"] != pred_df["prev_signal"]) & (pred_df["signal"] != 0)).astype(int)
pred_df["tcost"] = pred_df["trade"] * TRANSACTION_COST
pred_df["strategy_ret_raw"] = pred_df["signal"] * pred_df["future_ret"]
pred_df["strategy_ret"] = pred_df["strategy_ret_raw"] - pred_df["tcost"]
pred_df["equity"] = (1 + pred_df["strategy_ret"]).cumprod()

total_return = pred_df["equity"].iloc[-1] - 1
mean_r = pred_df["strategy_ret"].mean()
std_r = pred_df["strategy_ret"].std()
sharpe = (mean_r / std_r) * np.sqrt(HOURS_PER_YEAR) if std_r != 0 else np.nan

signal_counts = pred_df["signal"].value_counts().to_dict()
acc = (np.sign(pred_df["future_ret"]) == pred_df["signal"]).mean()

y_true = pred_df["future_ret"].values
y_pred = pred_df["pred_smooth"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = np.mean(np.abs(y_true - y_pred))
r2 = r2_score(y_true, y_pred)
pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]

true_up = (y_true > 0).astype(int)
pred_score = y_pred
try:
    auc_up = roc_auc_score(true_up, pred_score)
except ValueError:
    auc_up = np.nan

pred_sign = np.sign(y_pred)
hit_rate = (pred_sign == np.sign(y_true)).mean()
accuracy_up = (pred_sign[true_up == 1] == 1).mean() if true_up.sum() > 0 else np.nan

equity = pred_df["equity"].values
total_return = equity[-1] - 1
returns = pred_df["strategy_ret"].values
mean_r = returns.mean()
std_r = returns.std()
sharpe = (mean_r / std_r) * np.sqrt(HOURS_PER_YEAR) if std_r != 0 else np.nan
max_dd = np.max(np.maximum.accumulate(equity) - equity)
profit_factor = returns[returns > 0].sum() / (-returns[returns < 0].sum() + 1e-12)
trade_count = pred_df["trade"].sum()

print("\n=== 4H REGRESSION EV FULL METRIC SUMMARY ===")
print(f"Predicted bars: {len(pred_df)}")
print(f"Total return: {total_return*100:.2f}%")
print(f"Annualized Sharpe (24/7): {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Trade count: {trade_count}")
print(f"Hit rate (directional): {hit_rate:.4f}, Accuracy on up moves: {accuracy_up:.4f}")
print(f"AUC (up vs down): {auc_up:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(pred_df.index, pred_df["equity"], label="Equity (4H Regression-EV)", color="blue")
plt.title("BTC 4H Regression EV Strategy Equity Curve")
plt.xlabel("Time")
plt.ylabel("Equity (starting at 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

'''
FINAL RESULTS 

Predicted bars: 483
Total return: 9.61%
Annualized Sharpe (24/7): 4.55
Max Drawdown: 0.08
Profit Factor: 1.29
Trade count: 47
Hit rate (directional): 0.5093, Accuracy on up moves: 0.2558
AUC (up vs down): 0.4751
'''
