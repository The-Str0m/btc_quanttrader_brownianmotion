import warnings
import os
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

TICKER = "BTC-USD"
PERIOD = "729d"
INTERVAL = "1h"
TRAIN_WINDOW_DAYS = 729
TEST_DAYS = 21
FUTURE_HORIZ_HOURS = 1
RET_THRESHOLD = 0.0008
VAL_DAYS = 14
MIN_TRAIN_BARS = 500
N_BINS = 10
TRANSACTION_COST = 0.0005
MIN_EV = 0.0
HOURS_PER_YEAR = 365*24
RANDOM_STATE = 42
OUT_CSV = "btc_ev_calibrated_backtest.csv"
SUMMARY_CSV = "btc_ev_calibrated_summary.csv"

print("Downloading data...")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
else:
    df.index = df.index.tz_convert("UTC")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

df = df[["Open","High","Low","Close","Volume"]].copy()

df["ret_1h"] = df["Close"].pct_change()
df["future_ret"] = df["Close"].shift(-FUTURE_HORIZ_HOURS)/df["Close"]-1
df["typ_price"] = (df["High"]+df["Low"]+df["Close"])/3
df["VWAP"] = (df["typ_price"]*df["Volume"]).cumsum()/(df["Volume"].cumsum()+1e-12)
df["sma_20"] = df["Close"].rolling(20).mean()
df["ema_9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["ema_diff_9_21"] = df["ema_9"] - df["ema_21"]
df["dist_close_sma20"] = (df["Close"] - df["sma_20"])/(df["sma_20"]+1e-12)

macd = ta.macd(df["Close"])
df["macd"] = macd.iloc[:,0] if macd is not None else 0
df["rsi_14"] = ta.rsi(df["Close"], length=14)

stoch = ta.stoch(df["High"], df["Low"], df["Close"])
if stoch is not None:
    df["stoch_k"] = stoch.iloc[:,0]
    df["stoch_d"] = stoch.iloc[:,1] if stoch.shape[1]>1 else 0

df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
bb = ta.bbands(df["Close"])
df["bb_width"] = (bb.iloc[:,0]-bb.iloc[:,2])/(bb.iloc[:,1]+1e-12) if bb is not None else 0
df["obv"] = ta.obv(df["Close"], df["Volume"])
df["vol_20"] = df["ret_1h"].rolling(20).std()
df["vol_z"] = (df["Volume"] - df["Volume"].rolling(50).mean())/(df["Volume"].rolling(50).std()+1e-12)
df["body"] = df["Close"] - df["Open"]
df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]
df["range_pct"] = (df["High"] - df["Low"])/(df["Close"]+1e-12)
df["ret_2h"] = df["Close"].pct_change(2)
df["ret_3h"] = df["Close"].pct_change(3)
df["ret_6h"] = df["Close"].pct_change(6)
df["zscore_20"] = (df["Close"] - df["Close"].rolling(20).mean())/(df["Close"].rolling(20).std()+1e-12)

df.dropna(inplace=True)

df["Target"] = df["future_ret"].apply(lambda x: 1 if x>RET_THRESHOLD else (-1 if x<-RET_THRESHOLD else 0))
df["Target_m"] = df["Target"].map({-1:0,0:1,1:2})

features = ["VWAP","ema_diff_9_21","sma_20","dist_close_sma20","macd","rsi_14","stoch_k","stoch_d",
            "atr_14","vol_20","bb_width","obv","vol_z","body","upper_wick","lower_wick","range_pct",
            "ret_1h","ret_2h","ret_3h","ret_6h","zscore_20"]
features = [f for f in features if f in df.columns]

unique_dates = sorted({t.date() for t in df.index})[-TEST_DAYS:]
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
    random_state=RANDOM_STATE
)

for i, day in enumerate(unique_dates, 1):
    print(f"Processing test day {i}/{len(unique_dates)}: {day}")
    day_start = pd.Timestamp(day).tz_localize("UTC")
    train_start = day_start - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    train_mask = (df.index >= train_start) & (df.index.date < day)
    test_mask = df.index.date == day

    X_train_all = df.loc[train_mask, features]
    y_train_all = df.loc[train_mask, "Target_m"]
    X_test = df.loc[test_mask, features]
    future_ret_test = df.loc[test_mask, "future_ret"]

    if len(X_train_all) < MIN_TRAIN_BARS or X_test.empty:
        print(f"Skipping day {day} due to insufficient data")
        continue

    train_dates = sorted({t.date() for t in df.loc[train_mask].index})
    val_cut_ts = pd.Timestamp(train_dates[-VAL_DAYS]).tz_localize("UTC") if len(train_dates) > VAL_DAYS else df.index[0]
    train_mask2 = (df.index >= train_start) & (df.index < val_cut_ts)
    val_mask = (df.index >= val_cut_ts) & (df.index.date < day)

    X_train = df.loc[train_mask2, features]
    y_train = df.loc[train_mask2, "Target_m"]
    X_val = df.loc[val_mask, features]
    y_val_raw = df.loc[val_mask, "Target"]
    future_ret_val = df.loc[val_mask, "future_ret"]

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    prob_val = model.predict_proba(X_val)
    prob_short_val = prob_val[:,0]
    prob_long_val = prob_val[:,2]

    try:
        iso_long = IsotonicRegression(out_of_bounds="clip")
        iso_long.fit(prob_long_val, (y_val_raw==1).astype(int))
    except:
        iso_long = None

    try:
        iso_short = IsotonicRegression(out_of_bounds="clip")
        iso_short.fit(prob_short_val, (y_val_raw==-1).astype(int))
    except:
        iso_short = None

    prob_long_val_cal = iso_long.predict(prob_long_val) if iso_long else prob_long_val
    prob_short_val_cal = iso_short.predict(prob_short_val) if iso_short else prob_short_val

    def compute_ef(prob_cal, future_ret):
        bins = np.linspace(0,1,N_BINS+1)
        bin_idx = np.digitize(prob_cal, bins)-1
        centers = (bins[:-1]+bins[1:])/2.0
        ef = np.array([future_ret[bin_idx==i].mean() if (bin_idx==i).sum()>0 else 0 for i in range(len(centers))])
        return centers, ef

    centers_long, ef_long = compute_ef(prob_long_val_cal, future_ret_val)
    centers_short, ef_short = compute_ef(prob_short_val_cal, future_ret_val)

    prob_test = model.predict_proba(X_test)
    prob_long_t_cal = iso_long.predict(prob_test[:,2]) if iso_long else prob_test[:,2]
    prob_short_t_cal = iso_short.predict(prob_test[:,0]) if iso_short else prob_test[:,0]

    ef_long_test = np.interp(prob_long_t_cal, centers_long, ef_long)
    ef_short_test = np.interp(prob_short_t_cal, centers_short, ef_short)

    ev_long = prob_long_t_cal * ef_long_test
    ev_short = prob_short_t_cal * (-ef_short_test)
    sigs = np.where((ev_long>TRANSACTION_COST+MIN_EV) & (ev_long>ev_short),1,
                    np.where((ev_short>TRANSACTION_COST+MIN_EV) & (ev_short>ev_long),-1,0))

    for ts, s, pl, ps, plc, psc, efl, efs in zip(X_test.index, sigs, prob_test[:,2], prob_test[:,0],
                                                  prob_long_t_cal, prob_short_t_cal, ef_long_test, ef_short_test):
        pred_rows.append({
            "timestamp": ts,
            "signal": int(s),
            "prob_long_raw": float(pl),
            "prob_short_raw": float(ps),
            "prob_long_cal": float(plc),
            "prob_short_cal": float(psc),
            "ef_long": float(efl),
            "ef_short": float(efs),
            "future_ret": float(df.at[ts,"future_ret"]),
            "true": int(df.at[ts,"Target"])
        })

    summary_rows.append({
        "day": str(day),
        "train_bars": len(X_train_all),
        "val_bars": len(X_val),
        "test_bars": len(X_test),
        "n_trades": int((sigs!=0).sum())
    })

pred_df = pd.DataFrame(pred_rows).set_index("timestamp").sort_index()
pred_df["prev_signal"] = pred_df["signal"].shift(1).fillna(0)
pred_df["trade"] = ((pred_df["signal"] != pred_df["prev_signal"]) & (pred_df["signal"] != 0)).astype(int)
pred_df["tcost"] = pred_df["trade"] * TRANSACTION_COST
pred_df["strategy_ret"] = pred_df["signal"] * pred_df["future_ret"] - pred_df["tcost"]
pred_df["equity"] = (1 + pred_df["strategy_ret"]).cumprod()

total_return = pred_df["equity"].iloc[-1] - 1
mean_r = pred_df["strategy_ret"].mean()
std_r = pred_df["strategy_ret"].std()
sharpe = (mean_r / std_r) * np.sqrt(HOURS_PER_YEAR) if std_r!=0 else np.nan

try:
    y_true_up = (pred_df["true"]==1).astype(int)
    y_prob_long = pred_df["prob_long_cal"]
    acc = accuracy_score(y_true_up, (y_prob_long>0.5).astype(int))
    auc = roc_auc_score(y_true_up, y_prob_long)
except:
    acc = auc = np.nan

print(f"\nPredicted bars: {len(pred_df)}")
print(f"Total return: {total_return*100:.2f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Accuracy (up moves): {acc:.4f}")
print(f"AUC (up moves): {auc:.4f}")
print("Signal counts:", pred_df["signal"].value_counts().to_dict())

pred_df.to_csv(OUT_CSV)
pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

plt.figure(figsize=(12,5))
plt.plot(pred_df["equity"], label="Equity (EV-filtered)")
plt.title("BTC EV-filtered Strategy (calibrated probabilities)")
plt.xlabel("Time")
plt.ylabel("Equity (start=1)")
plt.grid(True)
plt.legend()
plt.show()

'''
FINAL RESULTS

Total return: 5.43%
Sharpe: 9.00
Accuracy (up moves): 0.5720
AUC (up moves): 0.5117
Signal counts: {0: 467, 1: 13, -1: 6}
'''
