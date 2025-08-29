#!/usr/bin/env python3
"""
stock_telegram_bot.py
A single-file, end-to-end stock prediction + Telegram alert bot.

Features:
- Data download via yfinance (no API key required)
- Indicators: SMA, RSI (+ simple MACD)
- Baseline ML: Linear Regression (scikit-learn)
- Optional Deep Learning: LSTM (if TensorFlow is installed; script degrades gracefully if not)
- Backtesting for SMA crossover + RSI filter
- Buy/Sell signal generation with thresholds
- Telegram alerts (via requests to Bot API)
- CLI subcommands to train, predict, backtest, and run the bot loop
- Auto-trains if a saved model is missing

Requirements (install first):
    pip install yfinance pandas numpy scikit-learn joblib requests
    # Optional for LSTM:
    pip install tensorflow

Quickstart examples:
    # Download data and train a Linear Regression model for AAPL
    python stock_telegram_bot.py train --ticker AAPL --model lr --period 2y

    # Predict next 5 days
    python stock_telegram_bot.py predict --ticker AAPL --model lr --horizon 5

    # Backtest SMA crossover with RSI filter
    python stock_telegram_bot.py backtest --ticker AAPL --period 2y --sma_short 20 --sma_long 50 --rsi_buy 30 --rsi_sell 70

    # Run alert bot every 60 minutes for AAPL, sending Telegram messages when signals trigger
    # (Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as env vars first)
    python stock_telegram_bot.py run-bot --ticker AAPL --interval 60 --model lr

Environment variables:
    TELEGRAM_BOT_TOKEN   - Your Telegram bot token
    TELEGRAM_CHAT_ID     - Your Telegram chat id (or a group/channel id where the bot is a member)
"""

import argparse
import datetime as dt
import io
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Optional TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ----------------------------
# Data & Indicators
# ----------------------------

def download_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker or network.")
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["SMA_5"] = sma(df["Close"], 5)
    df["SMA_20"] = sma(df["Close"], 20)
    df["SMA_50"] = sma(df["Close"], 50)
    df["RSI_14"] = rsi(df["Close"], 14)
    macd_line, signal_line = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["Target_1d"] = df["Close"].shift(-1)  # next day's close (regression)
    df.dropna(inplace=True)
    return df

# ----------------------------
# Baseline ML: Linear Regression
# ----------------------------

@dataclass
class LRModel:
    scaler: StandardScaler
    model: LinearRegression
    feature_cols: list

def train_lr(df_feat: pd.DataFrame, feature_cols: Optional[list] = None) -> LRModel:
    if feature_cols is None:
        feature_cols = ["Close","Return_1d","SMA_5","SMA_20","SMA_50","RSI_14","MACD","MACD_Signal","Volume"]
    avail = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[avail].values
    y = df_feat["Target_1d"].values
    split = int(len(df_feat) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    log(f"LR eval -> MAE: {mae:.4f}, R2: {r2:.4f}")
    return LRModel(scaler, model, avail)

def predict_lr(lr: LRModel, row: pd.Series) -> float:
    X = row[lr.feature_cols].values.reshape(1, -1)
    Xs = lr.scaler.transform(X)
    return float(lr.model.predict(Xs)[0])

def save_lr(lr: LRModel, ticker: str):
    path = MODEL_DIR / f"lr_{ticker}.joblib"
    dump(lr, path)
    log(f"Saved LR model to {path}")

def load_lr(ticker: str) -> Optional[LRModel]:
    path = MODEL_DIR / f"lr_{ticker}.joblib"
    if path.exists():
        return load(path)
    return None

# ----------------------------
# Optional LSTM
# ----------------------------

@dataclass
class LSTMModel:
    scaler: StandardScaler
    model: "keras.Model"
    feature_cols: list
    lookback: int

def make_lstm_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 30):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_lstm(df_feat: pd.DataFrame, feature_cols: Optional[list] = None, lookback: int = 30, epochs: int = 5) -> Optional[LSTMModel]:
    if not TF_AVAILABLE:
        log("TensorFlow is not available. Skipping LSTM training.")
        return None
    if feature_cols is None:
        feature_cols = ["Close","Return_1d","SMA_5","SMA_20","SMA_50","RSI_14","MACD","MACD_Signal","Volume"]
    avail = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[avail].values
    y = df_feat["Target_1d"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    split = int(len(df_feat) * 0.8)
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = y[:split], y[split:]
    X_train_seq, y_train_seq = make_lstm_sequences(X_train, y_train, lookback=lookback)
    X_test_seq, y_test_seq = make_lstm_sequences(X_test, y_test, lookback=lookback)
    model = keras.Sequential([
        layers.Input(shape=(lookback, len(avail))),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mae")
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=32, validation_data=(X_test_seq, y_test_seq), verbose=1)
    # Simple evaluation
    test_pred = model.predict(X_test_seq, verbose=0).ravel()
    mae = mean_absolute_error(y_test_seq, test_pred)
    log(f"LSTM eval -> MAE: {mae:.4f}")
    return LSTMModel(scaler, model, avail, lookback)

def predict_lstm(lstm: LSTMModel, df_feat: pd.DataFrame) -> float:
    # Predict next day's close using the latest lookback window
    tail = df_feat[lstm.feature_cols].values
    Xs = lstm.scaler.transform(tail)
    if len(Xs) < lstm.lookback:
        raise ValueError("Not enough data for LSTM lookback window.")
    window = Xs[-lstm.lookback:]
    window = np.expand_dims(window, axis=0)
    pred = lstm.model.predict(window, verbose=0)[0,0]
    return float(pred)

def save_lstm(lstm: LSTMModel, ticker: str):
    if not TF_AVAILABLE:
        return
    path = MODEL_DIR / f"lstm_{ticker}.keras"
    lstm.model.save(path)
    meta = {
        "feature_cols": lstm.feature_cols,
        "lookback": lstm.lookback
    }
    with open(MODEL_DIR / f"lstm_{ticker}.meta.json", "w") as f:
        json.dump(meta, f)
    dump(lstm.scaler, MODEL_DIR / f"lstm_{ticker}.scaler.joblib")
    log(f"Saved LSTM artifacts to {MODEL_DIR}")

def load_lstm(ticker: str) -> Optional[LSTMModel]:
    if not TF_AVAILABLE:
        return None
    path = MODEL_DIR / f"lstm_{ticker}.keras"
    meta_path = MODEL_DIR / f"lstm_{ticker}.meta.json"
    scaler_path = MODEL_DIR / f"lstm_{ticker}.scaler.joblib"
    if path.exists() and meta_path.exists() and scaler_path.exists():
        model = keras.models.load_model(path)
        with open(meta_path) as f:
            meta = json.load(f)
        scaler = load(scaler_path)
        return LSTMModel(scaler, model, meta["feature_cols"], meta["lookback"])
    return None

# ----------------------------
# Signals & Backtest
# ----------------------------

def generate_signals(df: pd.DataFrame, sma_short: int = 20, sma_long: int = 50, rsi_buy: int = 35, rsi_sell: int = 65) -> pd.DataFrame:
    df = df.copy()
    df["SMA_S"] = sma(df["Close"], sma_short)
    df["SMA_L"] = sma(df["Close"], sma_long)
    df["RSI"] = rsi(df["Close"], 14)
    df["Signal"] = 0  # 1=buy, -1=sell
    # Buy when short SMA crosses above long SMA and RSI below threshold
    cross_up = (df["SMA_S"] > df["SMA_L"]) & (df["SMA_S"].shift(1) <= df["SMA_L"].shift(1))
    cross_down = (df["SMA_S"] < df["SMA_L"]) & (df["SMA_S"].shift(1) >= df["SMA_L"].shift(1))
    df.loc[cross_up & (df["RSI"] <= rsi_buy), "Signal"] = 1
    df.loc[cross_down & (df["RSI"] >= rsi_sell), "Signal"] = -1
    return df

def backtest(df: pd.DataFrame, sma_short: int, sma_long: int, rsi_buy: int, rsi_sell: int, fee_bps: float = 5.0) -> dict:
    sig = generate_signals(df, sma_short, sma_long, rsi_buy, rsi_sell)
    sig["Position"] = sig["Signal"].replace(0, np.nan).ffill().fillna(0)  # hold last signal
    sig["MarketRet"] = sig["Close"].pct_change().fillna(0.0)
    # Apply basic trading rule: long when position==1, flat when 0, short when -1
    gross = sig["Position"].shift(1).fillna(0) * sig["MarketRet"]
    # Transaction cost when signal changes (per trade)
    trades = sig["Position"].diff().abs().fillna(0)
    cost = trades * (fee_bps / 1e4)
    net = gross - cost
    equity = (1 + net).cumprod()
    total_return = equity.iloc[-1] - 1
    ann_factor = 252  # daily
    sharpe = (net.mean() * ann_factor) / (net.std(ddof=0) * math.sqrt(ann_factor) + 1e-9)
    max_dd = (equity / equity.cummax() - 1).min()
    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "equity_last": float(equity.iloc[-1]),
    }

# ----------------------------
# Telegram
# ----------------------------

def send_telegram_message(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; skipping Telegram send.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=15)
        ok = r.status_code == 200 and r.json().get("ok", False)
        if not ok:
            log(f"Telegram send failed: {r.text}")
        return ok
    except Exception as e:
        log(f"Telegram error: {e}")
        return False

# ----------------------------
# CLI Workflows
# ----------------------------

def cmd_train(args):
    df = download_data(args.ticker, period=args.period, interval=args.interval)
    df_feat = add_features(df)
    if args.model == "lr":
        lr = train_lr(df_feat)
        save_lr(lr, args.ticker)
    elif args.model == "lstm":
        lstm = train_lstm(df_feat, lookback=args.lookback, epochs=args.epochs)
        if lstm:
            save_lstm(lstm, args.ticker)
        else:
            log("LSTM not trained (TensorFlow missing).")
    else:
        log("Unknown model type. Use 'lr' or 'lstm'.")

def cmd_predict(args):
    df = download_data(args.ticker, period=args.period, interval=args.interval)
    df_feat = add_features(df)
    if args.model == "lr":
        lr = load_lr(args.ticker)
        if lr is None:
            log("No saved LR model; training now...")
            lr = train_lr(df_feat)
            save_lr(lr, args.ticker)
        last_row = df_feat.iloc[-1]
        pred = predict_lr(lr, last_row)
        print(f"Predicted next close for {args.ticker}: {pred:.2f}")
    else:
        if not TF_AVAILABLE:
            log("TensorFlow not available; cannot run LSTM prediction.")
            return
        lstm = load_lstm(args.ticker)
        if lstm is None:
            log("No saved LSTM; training now...")
            lstm = train_lstm(df_feat, lookback=args.lookback, epochs=args.epochs)
            if lstm:
                save_lstm(lstm, args.ticker)
            else:
                return
        pred = predict_lstm(lstm, df_feat)
        print(f"Predicted next close for {args.ticker}: {pred:.2f}")

def cmd_backtest(args):
    df = download_data(args.ticker, period=args.period, interval=args.interval)
    res = backtest(df, args.sma_short, args.sma_long, args.rsi_buy, args.rsi_sell, args.fee_bps)
    print(json.dumps(res, indent=2))

def cmd_run_bot(args):
    ticker = args.ticker
    interval_min = args.interval
    model_type = args.model
    sma_short, sma_long = args.sma_short, args.sma_long
    rsi_buy, rsi_sell = args.rsi_buy, args.rsi_sell
    period, interval = args.period, args.interval_tf

    log(f"Starting bot for {ticker} every {interval_min} minutes using {model_type.upper()}...")

    while True:
        try:
            df = download_data(ticker, period=period, interval=interval)
            df_feat = add_features(df)
            last_close = df_feat["Close"].iloc[-1]
            # Signals
            sig = generate_signals(df, sma_short, sma_long, rsi_buy, rsi_sell).iloc[-1]
            signal_txt = "HOLD"
            if sig["Signal"] == 1:
                signal_txt = "BUY"
            elif sig["Signal"] == -1:
                signal_txt = "SELL"

            # Prediction
            if model_type == "lr":
                lr = load_lr(ticker)
                if lr is None:
                    log("Training LR on the fly...")
                    lr = train_lr(df_feat)
                    save_lr(lr, ticker)
                pred = predict_lr(lr, df_feat.iloc[-1])
            else:
                pred = None
                if TF_AVAILABLE:
                    lstm = load_lstm(ticker)
                    if lstm is None:
                        log("Training LSTM on the fly...")
                        lstm = train_lstm(df_feat, lookback=args.lookback, epochs=args.epochs)
                        if lstm:
                            save_lstm(lstm, ticker)
                    if lstm:
                        pred = predict_lstm(lstm, df_feat)
                else:
                    log("TensorFlow not available; skipping LSTM prediction.")

            msg = (
                f"{ticker} @ {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Last Close: {last_close:.2f}\n"
                f"Signal: {signal_txt} (SMA{args.sma_short}/{args.sma_long}, RSI buy<= {rsi_buy} sell>= {rsi_sell})\n"
            )
            if pred is not None:
                msg += f"Predicted next close: {pred:.2f}\n"
                delta = ((pred - last_close) / last_close) * 100
                msg += f"Implied move: {delta:+.2f}%\n"

            ok = send_telegram_message(msg)
            if ok:
                log("Telegram sent.")
            else:
                log("Telegram not sent (details above).")

        except Exception as e:
            log(f"Run error: {e}")

        time.sleep(max(60, int(interval_min * 60)))

def build_parser():
    p = argparse.ArgumentParser(description="Stock prediction + Telegram bot (single-file).")
    sub = p.add_subparsers(dest="cmd")

    # Train
    t = sub.add_parser("train", help="Train a model and save it.")
    t.add_argument("--ticker", required=True)
    t.add_argument("--period", default="2y")
    t.add_argument("--interval", default="1d")
    t.add_argument("--model", choices=["lr","lstm"], default="lr")
    t.add_argument("--lookback", type=int, default=30)
    t.add_argument("--epochs", type=int, default=5)
    t.set_defaults(func=cmd_train)

    # Predict
    pr = sub.add_parser("predict", help="Predict next close using saved (or on-the-fly) model.")
    pr.add_argument("--ticker", required=True)
    pr.add_argument("--period", default="2y")
    pr.add_argument("--interval", default="1d")
    pr.add_argument("--model", choices=["lr","lstm"], default="lr")
    pr.add_argument("--lookback", type=int, default=30)
    pr.add_argument("--epochs", type=int, default=3)
    pr.set_defaults(func=cmd_predict)

    # Backtest
    b = sub.add_parser("backtest", help="Backtest SMA crossover + RSI filter.")
    b.add_argument("--ticker", required=True)
    b.add_argument("--period", default="2y")
    b.add_argument("--interval", default="1d")
    b.add_argument("--sma_short", type=int, default=20)
    b.add_argument("--sma_long", type=int, default=50)
    b.add_argument("--rsi_buy", type=int, default=35)
    b.add_argument("--rsi_sell", type=int, default=65)
    b.add_argument("--fee_bps", type=float, default=5.0, help="Per-trade fee in basis points (5=0.05%).")
    b.set_defaults(func=cmd_backtest)

    # Run bot
    rb = sub.add_parser("run-bot", help="Run periodic alerts to Telegram.")
    rb.add_argument("--ticker", required=True)
    rb.add_argument("--interval", type=int, default=60, help="Minutes between checks.")
    rb.add_argument("--model", choices=["lr","lstm"], default="lr")
    rb.add_argument("--period", default="6mo")
    rb.add_argument("--interval_tf", default="1h", help="Data interval (e.g., 1h, 1d).")
    rb.add_argument("--lookback", type=int, default=30)
    rb.add_argument("--epochs", type=int, default=3)
    rb.add_argument("--sma_short", type=int, default=20)
    rb.add_argument("--sma_long", type=int, default=50)
    rb.add_argument("--rsi_buy", type=int, default=35)
    rb.add_argument("--rsi_sell", type=int, default=65)
    rb.set_defaults(func=cmd_run_bot)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(0)
    args.func(args)

if __name__ == "__main__":
    main()
