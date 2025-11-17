note - v4 is the submitted model, v5-v7 are GBM based



ML Prediction of BTC Price Movements and Their Connection to Brownian Motion

By Arjun and Pranav (11K)





Abstract

This project aims to investigate whether short-term Bitcoin price movements can be predicted using a machine learning classifier combined with probability calibration and an expected value (EV) filter. An XGBoost multiclass classifier is used to predict short, long and flat outcomes one hour prior using two years of 1-hour historical data, pulled from Yahoo Finance. 

To relate this to physics, we show that Bitcoin prices behave similar to a stochastic Brownian particle, where price evolution comes from random noise combined with a small structural drift. The ML model aims to learn deviations from pure randomness, similar to detecting microscopic drift inside Brownian Motion.

By yielding positive results from a backtest of the previous 21 days, we have been able to successfully show how statistical structure can be extracted from otherwise noisy/stochastic behaviour. 





Introduction

Financial markets behave unpredictably, especially at short timescales. Bitcoin, as a highly liquid and decentralized digital asset, displays rapid fluctuations often modeled as stochastic processes, similar to random thermal motion observed in physics.

We aim to answer whether an ML model can detect small predictive signals inside what otherwise seems to be Brownian Motion.

To test this, we have built an ML pipeline that - 
Downloads 729 days of hourly BTC data (from Yahoo Finance)
Calculates technical and statistical features
Trains an XGBoost classifier daily
Calibrates the predicted probabilities (using isotonic regression)
Converts probabilities into Expected Returns
Uses EV filtering to only make trades when profitable
Backtests the strategy for the previous 21 days
This allows us to apply Brownian Motion theory to Quant Trading.





Data & Classification

We pull 729 days of BTC OHLCV data using Yahoo Finance’s Python module.

We define short-term future return as - 
( Price (next hour) / Price (current hour) )  - 1

If this is above 0.08% (baseline transaction cost), we classify it as long. If it is below 0.08%, we classify it as short. Otherwise, we classify it as hold.

These are internally mapped for the XGBoost model.





Features

We’re using more than 25 features to train our XGBoost Classifier model. These features represent trend, momentum, volatility, price actions, and volume dynamics.

The features include - 
Moving Averages
Momentum Indicators
Volatility Indicators
Volume Based
Price Action
Returns & Z-Scores
Volume Weighted Average Price
These features are commonly used in quant finance and can help detect structural signals hidden inside noisy price movements.





Model

The parameters we use for our XGBoost Classifier model include - 
n_estimators = 300
max_depth = 5
learning_rate = 0.03
subsample = 0.85
colsample_bytree = 0.85
objective = "multi:softprob"

XGBoost was chosen because - 
It handles non-linearity
Works well on time-series data
Produces class probabilities (for short, hold, long classifications)
Supports rolling retraining

The rolling retraining method works by training on the previous 729 days of data on every test day. It reserves the last 14 days for validation, and calibrates the model probabilities. It then computes the expected returns and gives its predictions for that day.

This simulates a real trading system that updates each day with new information.





Expected Value Modelling 


While the XGBoost Classifier model provides a calibrated probability for each class, it does not determine whether such a trade is financially feasible or not. EV modelling incorporates both the likelihood and magnitude of the associated outcome, which helps us take a more risk-aware trade.

To do this, we compute the mean future return for each probability, so that each probability in the test set is associated with an expected return. This ensures that EV represents the probability-weighted average return under each action. The model uses this information to enter a position, if the EV is greater than the transaction cost, the model classifies it as a long position, or a short position for the opposite, ensuring that it remains profitable.

The advantages of this approach are - 
EV directly measures expected profit
EV filtering suppresses weak signals and highlights only statistically meaningful events
EV helps avoid reacting erratically to minimal price movements





Results

The results of the calibrated, EV-filtered XGBoost Classifier model was evaluated over a 21-day test period after being trained on 729 days of hourly financial data. The results are summarised below - 
Predicted Bars -  484
Total Return -  5.43%
Sharpe Ratio -  9.02
Directional Accuracy -  0.5723
AUC - 0.5116
Signal Counts - {0: 465, 1: 13, –1: 6}

The model produced only 19 trades (out of 484 total hourly bars. This confirms that the EV filter is highly selective, entering a position only when the expected return, adjusted for transaction costs, exceeds the threshold. The dominance of the “flat” flat position demonstrates that the model avoids low-quality predictions and restricts trading to situations with meaningful statistical edge.
The directional accuracy of 57.23% and AUC of 0.5116 indicate that the classifier has weak but non-random ability to predict positive returns. Importantly, even modest predictive power can produce profitable results when combined with probability calibration and EV filtering. The AUC near 0.51 suggests that raw model discrimination is limited, however, this is consistent with the stochastic microstructure of short time-period cryptocurrency returns, which often approach noise-dominated behavior similar to Brownian motion.





Brownian Motion and Market Prices

Brownian motion describes the random movement of particles suspended in a fluid.
Mathematically - 
dx = μdt + σdWt
Where -
μ = drift
σ = volatility
Wt = Wiener process (pure randomness)
Financial price dynamics are modelled using a similar technique. The XGBoost Classifier model aims to detect the drift component inside of the volatile noise. Most price movements are random, however the ML model aims to detect those movements which reflect structural effects. EV filtering helps select trajectories where drift outweighs noise. 





Conclusion

This project demonstrates that - 
BTC prices exhibit stochastic Brownian behaviour
ML models can detect small drift patterns using calibration and EV filtering
The final strategy trades rarely but is able to identify profitable trades. 
The physics connection shows why it is impossible to time the market, noise mostly dominates drift, similar to Brownian Motion

Future enhancements include - 
Using stochastic differential equations (SDE)
Multi-horizon forecasting (2h, 4h, 12h)
Advanced Calibration techniques (Platt Scaling, Beta Calibration)
Accurate Expected Value Estimation (Bayesian)
