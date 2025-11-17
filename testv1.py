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

TICKER = "BTC-USD"
PERIOD = "729d"
INTERVAL = "1h"
TEST_DAYS = 7
FUTURE_HORIZ_HOURS = 3
RET_THRESHOLD = 0.0015
MIN_TRAIN_BARS = 500
VAL_DAYS = 14
PROB_GRID = np.linspace(0.51, 0.9, 40)
TRANSACTION_COST = 0.0005
HOURS_PER_YEAR = 365 * 24
RANDOM_STATE = 42

df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
df.index = pd.to_datetime(df.index)
df = df.dropna()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
else:
    df.columns = list(df.columns)

df["ret_1h"] = df["Close"].pct_change()
df["future_ret"] = df["Close"].shift(-FUTURE_HORIZ_HOURS) / df["Close"] - 1

df["typ_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
groups = df.index.normalize()
df["vwap_num"] = (df["typ_price"] * df["Volume"]).groupby(groups).cumsum()
df["vwap_den"] = df["Volume"].groupby(groups).cumsum()
df["VWAP"] = df["vwap_num"] / (df["vwap_den"] + 1e-12)

df["ema_9"] = ta.ema(df["Close"], length=9)
df["ema_21"] = ta.ema(df["Close"], length=21)
df["ema_diff_9_21"] = df["ema_9"] - df["ema_21"]
df["sma_20"] = ta.sma(df["Close"], length=20)
df["dist_close_sma20"] = (df["Close"] - df["sma_20"]) / (df["sma_20"] + 1e-12)

macd = ta.macd(df["Close"])
df["macd"] = macd.iloc[:, 0]
df["macd_sig"] = macd.iloc[:, 1]
df["macd_hist"] = macd.iloc[:, 2]

df["rsi_14"] = ta.rsi(df["Close"], length=14)
stoch = ta.stoch(df["High"], df["Low"], df["Close"])
df["stoch_k"] = stoch.iloc[:, 0]
df["stoch_d"] = stoch.iloc[:, 1]

df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
df["vol_20"] = df["ret_1h"].rolling(20).std()

bb = ta.bbands(df["Close"], length=20, std=2)
df["bb_upper"] = bb.iloc[:, 0]
df["bb_mid"] = bb.iloc[:, 1]
df["bb_lower"] = bb.iloc[:, 2]
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-12)

df["obv"] = ta.obv(df["Close"], df["Volume"])
df["vol_z"] = (df["Volume"] - df["Volume"].rolling(50).mean()) / (df["Volume"].rolling(50).std() + 1e-12)

df["body"] = (df["Close"] - df["Open"]).abs()
df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-12)

df["ret_2h"] = df["Close"].pct_change(2)
df["ret_3h"] = df["Close"].pct_change(3)
df["ret_6h"] = df["Close"].pct_change(6)
df["zscore_20"] = (df["Close"] - df["Close"].rolling(20).mean()) / (df["Close"].rolling(20).std() + 1e-12)

df = df.dropna().sort_index()

def make_ternary(x, thr):
    if x > thr: return 1
    if x < -thr: return -1
    return 0

df["Target_raw"] = df["future_ret"]
df["Target"] = df["Target_raw"].apply(lambda x: make_ternary(x, RET_THRESHOLD))
df["Target_m"] = df["Target"].map({-1: 0, 0: 1, 1: 2})
df = df.dropna()

features = [
    "VWAP","ema_diff_9_21","sma_20","dist_close_sma20","macd","macd_sig","macd_hist",
    "rsi_14","stoch_k","stoch_d","atr_14","vol_20","bb_width","obv","vol_z",
    "body","upper_wick","lower_wick","range_pct","ret_1h","ret_2h","ret_3h",
    "ret_6h","zscore_20"
]

unique_dates = sorted(df.index.date)
test_days = unique_dates[-TEST_DAYS:]
pred_rows = []

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
    train_mask = df.index.date < day
    test_mask = df.index.date == day

    X_train_all = df.loc[train_mask, features]
    y_train_all = df.loc[train_mask, "Target_m"]

    X_test = df.loc[test_mask, features]
    y_test_true = df.loc[test_mask, "Target"]
    future_ret_test = df.loc[test_mask, "future_ret"]

    if len(X_train_all) < MIN_TRAIN_BARS or X_test.empty:
        continue

    train_dates = sorted(df.loc[train_mask].index.date)
    val_cut = train_dates[-VAL_DAYS] if len(train_dates) > VAL_DAYS else train_dates[0]

    val_mask = (df.index.date >= val_cut) & (df.index.date < day)
    train_mask2 = (df.index.date < val_cut)

    X_train = df.loc[train_mask2, features]
    y_train = df.loc[train_mask2, "Target_m"]
    X_val = df.loc[val_mask, features]
    y_val_true = df.loc[val_mask, "Target"]
    future_ret_val = df.loc[val_mask, "future_ret"]

    if X_val.shape[0] < 50:
        last_n = 200
        X_train = df.loc[train_mask, features].iloc[:-last_n]
        y_train = df.loc[train_mask, "Target_m"].iloc[:-last_n]
        X_val = df.loc[train_mask, features].iloc[-last_n:]
        y_val_true = df.loc[train_mask, "Target"].iloc[-last_n:]
        future_ret_val = df.loc[train_mask, "future_ret"].iloc[-last_n:]

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    prob_val = model.predict_proba(X_val)
    best_u, best_sharpe = None, -np.inf

    for u in PROB_GRID:
        prob_short = prob_val[:, 0]
        prob_long = prob_val[:, 2]

        sig = np.zeros(len(prob_val), dtype=int)
        for i in range(len(sig)):
            ps, pl = prob_short[i], prob_long[i]
            if pl > u and pl >= ps: sig[i] = 1
            elif ps > u and ps > pl: sig[i] = -1

        ret_series = sig * future_ret_val.values
        std = ret_series.std()
        if std == 0 or np.isnan(std):
            continue

        sh = (ret_series.mean() / std) * np.sqrt(HOURS_PER_YEAR)
        if sh > best_sharpe:
            best_sharpe, best_u = sh, u

    if best_u is None:
        best_u = 0.6

    prob_test = model.predict_proba(X_test)
    prob_short_t = prob_test[:, 0]
    prob_long_t = prob_test[:, 2]

    sig_test = np.zeros(len(prob_test), dtype=int)
    for i in range(len(sig_test)):
        ps, pl = prob_short_t[i], prob_long_t[i]
        if pl > best_u and pl >= ps: sig_test[i] = 1
        elif ps > best_u and ps > pl: sig_test[i] = -1

    for ts, s, p_up, p_short, true_lbl, fr in zip(
        X_test.index, sig_test, prob_long_t, prob_short_t, y_test_true, future_ret_test
    ):
        pred_rows.append({
            "timestamp": ts,
            "signal": int(s),
            "prob_long": float(p_up),
            "prob_short": float(p_short),
            "true": int(true_lbl),
            "future_ret": float(fr),
            "tuned_u": float(best_u),
        })

pred_df = pd.DataFrame(pred_rows).set_index("timestamp").sort_index()

backtest = pred_df.merge(df[["future_ret"]], left_index=True, right_index=True, how="left")
backtest["future_ret"] = backtest["future_ret_x"].fillna(backtest["future_ret_y"])

backtest["prev_signal"] = backtest["signal"].shift(1).fillna(0)
backtest["trade"] = ((backtest["signal"] != backtest["prev_signal"]) & (backtest["signal"] != 0)).astype(int)
backtest["tcost"] = backtest["trade"] * TRANSACTION_COST

backtest["strategy_ret_raw"] = backtest["signal"] * backtest["future_ret"]
backtest["strategy_ret"] = backtest["strategy_ret_raw"] - backtest["tcost"]
backtest["equity"] = (1 + backtest["strategy_ret"]).cumprod()

total_return = backtest["equity"].iloc[-1] - 1
mean_ret = backtest["strategy_ret"].mean()
std_ret = backtest["strategy_ret"].std()
sharpe = (mean_ret / std_ret) * np.sqrt(HOURS_PER_YEAR) if std_ret != 0 else np.nan

raw_label = (backtest["prob_long"] > 0.5).astype(int)
true_up = (backtest["true"] == 1).astype(int)
accuracy = (raw_label == true_up).mean()
auc = roc_auc_score(true_up, backtest["prob_long"])

print(f"Predicted bars: {len(backtest)}")
print(f"Total return: {total_return*100:.2f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Accuracy (up moves): {accuracy:.4f}")
print(f"AUC (up moves): {auc:.4f}")
print("Signal counts:", backtest["signal"].value_counts().to_dict())

plt.figure(figsize=(12,5))
plt.plot(backtest["equity"], label="Strategy Equity")
plt.title("BTC-USD Hourly | Daily Retrain | Ternary Target | Tuned Threshold")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.grid(True)
plt.legend()
plt.show()

backtest.to_csv("btc_daily_retrain_ternary_tuned_backtest.csv")
