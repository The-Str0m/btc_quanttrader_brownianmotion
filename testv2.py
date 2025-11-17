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
HOURS_PER_YEAR = 365 * 24
RANDOM_STATE = 42
VERBOSE = True
OUT_CSV = "btc_ev_calibrated_backtest.csv"
SUMMARY_CSV = "btc_ev_calibrated_summary.csv"

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
    df["macd"] = macd.iloc[:,0]
    if macd.shape[1] > 1: df["macd_sig"] = macd.iloc[:,1]
    if macd.shape[1] > 2: df["macd_hist"] = macd.iloc[:,2]

df["rsi_14"] = ta.rsi(df["Close"], length=14)
stoch = ta.stoch(df["High"], df["Low"], df["Close"])
if stoch is not None and not stoch.empty:
    cols = [c.lower() for c in stoch.columns]
    k_i = next((i for i,c in enumerate(cols) if "k" in c), 0)
    d_i = next((i for i,c in enumerate(cols) if "d" in c and i!=k_i), (1 if stoch.shape[1] > 1 else 0))
    df["stoch_k"] = stoch.iloc[:,k_i]
    if stoch.shape[1] > 1: df["stoch_d"] = stoch.iloc[:,d_i]

df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
bb = ta.bbands(df["Close"], length=20, std=2)
if bb is not None and not bb.empty:
    df["bb_upper"] = bb.iloc[:,0]
    df["bb_mid"] = bb.iloc[:,1]
    df["bb_lower"] = bb.iloc[:,2]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-12)

df["obv"] = ta.obv(df["Close"], df["Volume"])
df["vol_20"] = df["ret_1h"].rolling(20).std()
df["vol_z"] = (df["Volume"] - df["Volume"].rolling(50).mean()) / (df["Volume"].rolling(50).std() + 1e-12)
df["body"] = df["Close"] - df["Open"]
df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]
df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-12)
df["ret_2h"] = df["Close"].pct_change(2)
df["ret_3h"] = df["Close"].pct_change(3)
df["ret_6h"] = df["Close"].pct_change(6)
df["zscore_20"] = (df["Close"] - df["Close"].rolling(20).mean()) / (df["Close"].rolling(20).std() + 1e-12)

df = df.dropna().sort_index()

def to_tern(x, thr):
    if x > thr: return 1
    if x < -thr: return -1
    return 0

df["Target"] = df["future_ret"].apply(lambda x: to_tern(x, RET_THRESHOLD))
df["Target_m"] = df["Target"].map({-1:0,0:1,1:2})

df = df.dropna()
features = [
    "VWAP","ema_diff_9_21","sma_20","dist_close_sma20","macd","macd_sig","macd_hist",
    "rsi_14","stoch_k","stoch_d","atr_14","vol_20","bb_width","obv","vol_z","body",
    "upper_wick","lower_wick","range_pct","ret_1h","ret_2h","ret_3h","ret_6h","zscore_20"
]
features = [f for f in features if f in df.columns]

unique_dates = sorted({t.date() for t in df.index})
test_days = unique_dates[-TEST_DAYS:]

pred_rows = []
summary_rows = []

model_params = dict(
    n_estimators=300,max_depth=5,learning_rate=0.03,
    subsample=0.85,colsample_bytree=0.85,
    objective="multi:softprob",num_class=3,
    use_label_encoder=False,eval_metric="mlogloss",
    n_jobs=-1,random_state=RANDOM_STATE
)

for day in test_days:
    ds = pd.Timestamp(day).tz_localize("UTC")
    ts0 = ds - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    mask_train = (df.index >= ts0) & (df.index.date < day)
    mask_test = df.index.date == day
    X_train_all = df.loc[mask_train, features]
    y_train_all = df.loc[mask_train, "Target_m"]
    X_test = df.loc[mask_test, features]
    y_test_raw = df.loc[mask_test, "Target"]
    fr_test = df.loc[mask_test, "future_ret"]
    if len(X_train_all) < MIN_TRAIN_BARS or X_test.empty:
        continue
    train_dates = sorted({t.date() for t in df.loc[mask_train].index})
    if len(train_dates) > VAL_DAYS:
        vc = train_dates[-VAL_DAYS]
        vts = pd.Timestamp(vc).tz_localize("UTC")
        mask_val = (df.index >= vts) & (df.index.date < day)
        mask_train2 = (df.index >= ts0) & (df.index < vts)
    else:
        N = min(800, max(200,int(len(X_train_all)*0.2)))
        idx_all = df.loc[mask_train].index
        mask_train2 = df.index.isin(idx_all[:-N]) if len(idx_all)>N else df.index.isin([])
        mask_val = df.index.isin(idx_all[-N:]) if len(idx_all)>N else df.index.isin([])
    X_train = df.loc[mask_train2, features]
    y_train = df.loc[mask_train2, "Target_m"]
    X_val = df.loc[mask_val, features]
    y_val_raw = df.loc[mask_val, "Target"]
    fr_val = df.loc[mask_val, "future_ret"]

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    prob_val = model.predict_proba(X_val)
    ps_val = prob_val[:,0]
    pf_val = prob_val[:,1]
    pl_val = prob_val[:,2]

    try:
        iso_l = IsotonicRegression(out_of_bounds="clip")
        iso_l.fit(pl_val, (y_val_raw==1).astype(int))
    except:
        iso_l = None
    try:
        iso_s = IsotonicRegression(out_of_bounds="clip")
        iso_s.fit(ps_val, (y_val_raw==-1).astype(int))
    except:
        iso_s = None

    pl_val_cal = iso_l.predict(pl_val) if iso_l is not None else pl_val
    ps_val_cal = iso_s.predict(ps_val) if iso_s is not None else ps_val

    if len(pl_val_cal) >= N_BINS:
        bins = np.linspace(0,1,N_BINS+1)
        idx = np.digitize(pl_val_cal,bins)-1
        centers = (bins[:-1]+bins[1:])/2
        ef_l = np.array([fr_val[idx==i].mean() if (idx==i).sum()>0 else 0 for i in range(len(centers))])
    else:
        ef_l = None
        centers = None

    if len(ps_val_cal) >= N_BINS:
        bins2 = np.linspace(0,1,N_BINS+1)
        idx2 = np.digitize(ps_val_cal,bins2)-1
        centers2 = (bins2[:-1]+bins2[1:])/2
        ef_s = np.array([fr_val[idx2==i].mean() if (idx2==i).sum()>0 else 0 for i in range(len(centers2))])
    else:
        ef_s = None
        centers2 = None

    prob_test = model.predict_proba(X_test)
    ps_t = prob_test[:,0]
    pl_t = prob_test[:,2]
    pl_t_cal = iso_l.predict(pl_t) if iso_l is not None else pl_t
    ps_t_cal = iso_s.predict(ps_t) if iso_s is not None else ps_t

    def map_ef(p, c, e):
        if e is None:
            m = fr_val.mean() if len(fr_val)>0 else 0
            return m*(p-0.5)*2
        return np.interp(p, c, e)

    ef_l_t = map_ef(pl_t_cal, centers, ef_l)
    ef_s_t = map_ef(ps_t_cal, centers2, ef_s)

    ev_l = pl_t_cal * ef_l_t
    ev_s = ps_t_cal * (-ef_s_t)

    sig = np.zeros_like(ev_l, dtype=int)
    for i in range(len(sig)):
        if ev_l[i] > TRANSACTION_COST+MIN_EV and ev_l[i] > ev_s[i]:
            sig[i] = 1
        elif ev_s[i] > TRANSACTION_COST+MIN_EV and ev_s[i] > ev_l[i]:
            sig[i] = -1

    for ts, s, plr, psr, plc, psc, el, es in zip(
        X_test.index, sig, pl_t, ps_t, pl_t_cal, ps_t_cal, ef_l_t, ef_s_t
    ):
        pred_rows.append({
            "timestamp": ts,
            "signal": int(s),
            "prob_long_raw": float(plr),
            "prob_short_raw": float(psr),
            "prob_long_cal": float(plc),
            "prob_short_cal": float(psc),
            "ef_long": float(el),
            "ef_short": float(es),
            "future_ret": float(df.at[ts,"future_ret"]),
            "true": int(df.at[ts,"Target"])
        })

    summary_rows.append({
        "day": str(day),
        "train_bars": len(X_train_all),
        "val_bars": len(X_val),
        "test_bars": len(X_test),
        "chosen_min_ev": MIN_EV,
        "avg_ev_long": float(np.nanmean(ev_l)) if len(ev_l)>0 else np.nan,
        "avg_ev_short": float(np.nanmean(ev_s)) if len(ev_s)>0 else np.nan,
        "n_trades": int((sig!=0).sum())
    })

pred_df = pd.DataFrame(pred_rows).set_index("timestamp").sort_index()
pred_df["future_ret_aligned"] = pred_df["future_ret"]
pred_df["prev_signal"] = pred_df["signal"].shift(1).fillna(0)
pred_df["trade"] = ((pred_df["signal"]!=pred_df["prev_signal"]) & (pred_df["signal"]!=0)).astype(int)
pred_df["tcost"] = pred_df["trade"] * TRANSACTION_COST
pred_df["strategy_ret_raw"] = pred_df["signal"] * pred_df["future_ret_aligned"]
pred_df["strategy_ret"] = pred_df["strategy_ret_raw"] - pred_df["tcost"]
pred_df["equity"] = (1 + pred_df["strategy_ret"]).cumprod()

total_return = pred_df["equity"].iloc[-1] - 1
m = pred_df["strategy_ret"].mean()
s = pred_df["strategy_ret"].std()
sharpe = (m / s) * np.sqrt(HOURS_PER_YEAR) if s!=0 else np.nan
y_true_up = (pred_df["true"]==1).astype(int)
y_prob = pred_df["prob_long_cal"]
acc = accuracy_score(y_true_up, (y_prob>0.5).astype(int))
auc = roc_auc_score(y_true_up, y_prob)

print("Predicted bars:", len(pred_df))
print("Total return:", total_return*100)
print("Annualized Sharpe (24/7):", sharpe)
print("Accuracy:", acc)
print("AUC:", auc)
print("Signal counts:", pred_df["signal"].value_counts().to_dict())

pred_df.to_csv(OUT_CSV)
pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

plt.figure(figsize=(12,5))
plt.plot(pred_df["equity"])
plt.title("BTC EV-filtered Strategy")
plt.grid(True)
plt.show()

'''
FINAL RESULTS 

Predicted bars: 486
Total return: 0.07%
Annualized Sharpe (24/7): 0.20
Accuracy (raw>0.5): 0.5782
AUC (prob_long calibrated): 0.5109
Signal counts: {0: 471, -1: 8, 1: 7}
'''
