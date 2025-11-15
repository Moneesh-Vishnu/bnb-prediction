# predict_bnb.py
# Minimal BNB/USDT prediction pipeline.
# - fetches klines from Binance (public)
# - computes EMA, RSI, returns
# - trains a small XGBoost classifier
# - prints probability next candle UP
# Optional: send Telegram message if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars set.

import os, json, time
import ccxt
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import requests
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env if present

# Config
SYMBOL = "BNB/USDT"
TF = "1h"
LIMIT = 1000
MODEL_PARAMS = {"n_estimators": 200, "max_depth": 4, "use_label_encoder": False, "eval_metric": "logloss"}

def fetch_ohlcv(symbol=SYMBOL, timeframe=TF, limit=LIMIT):
    exchange = ccxt.binance({"enableRateLimit": True})
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","vol"])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def featurize(df):
    df = df.copy()
    df['ema10'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['rsi14'] = RSIIndicator(df['close'], window=14).rsi()
    df['ret1'] = df['close'].pct_change(1)
    df['vol_ret1'] = df['vol'].pct_change(1)
    df.dropna(inplace=True)
    return df

def train_and_predict(df):
    df = df.copy()
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    features = ['ema10','ema50','rsi14','ret1','vol_ret1']
    X = df[features].values
    y = df['target'].values

    # time-aware split (train on first 80%)
    split = int(len(df)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(np.unique(y_train)) < 2:
        # fallback: not enough variety in labels
        return {"error": "not enough label variety to train"}

    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    latest = df[features].iloc[-1].values.reshape(1,-1)
    prob_up = float(model.predict_proba(latest)[0,1])

    out = {
        "symbol": SYMBOL,
        "timeframe": TF,
        "timestamp": df.index[-1].isoformat(),
        "prob_up_next": prob_up,
        "test_accuracy": acc,
        "features_latest": dict(zip(features, map(float, latest.flatten().tolist())))
    }
    return out

def maybe_send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"sent": False, "reason": "no_telegram_env"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=15)
        return {"sent": r.ok, "status_code": r.status_code, "resp": r.text}
    except Exception as e:
        return {"sent": False, "error": str(e)}

def main():
    try:
        df = fetch_ohlcv()
        df = featurize(df)
        result = train_and_predict(df)
    except Exception as e:
        result = {"error": "exception", "exception": str(e)}

    # save to file
    with open("prediction.json", "w") as f:
        json.dump(result, f, indent=2)

    # print summary
    print(json.dumps(result, indent=2))

    # optional telegram
    if isinstance(result, dict) and "prob_up_next" in result:
        prob = result["prob_up_next"]
        text = f"BNB/USDT ({TF}) → Prob next up: {prob:.3f} (acc={result.get('test_accuracy'):.3f})"
        tg = maybe_send_telegram(text)
        print("telegram:", tg)
    else:
        print("No valid result to send to Telegram.")

if __name__ == "__main__":
    main()
