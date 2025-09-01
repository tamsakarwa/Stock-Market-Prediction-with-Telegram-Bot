
# Stock Market Prediction with Telegram Bot

A complete end-to-end stock prediction and Telegram alert bot in a **single Python file**.

## Features
- 📈 Data download via [yfinance](https://pypi.org/project/yfinance/)
- 📊 Technical indicators: SMA, RSI, MACD
- 🤖 Models:
  - Linear Regression (scikit-learn)
  - LSTM (TensorFlow, optional)
- 📉 Backtesting: SMA crossover + RSI filter with Sharpe ratio & max drawdown
- 🔔 Telegram alerts for Buy/Sell signals & price predictions
- 🛠 CLI commands: `train`, `predict`, `backtest`, `run-bot`
- 💾 Auto-saves trained models (`models/` directory)

## Installation

```bash
git clone <[predictionbot](https://github.com/tamsakarwa/Stock-Market-Prediction-with-Telegram-Bot)>
cd <Stock-Market-Prediction-with-Telegram-Bot>

# Install dependencies
pip install -r requirements.txt
```

Optional (for deep learning LSTM model):
```bash
pip install tensorflow
```

## Usage

### 1. Train a model
```bash
python stock_telegram_bot.py train --ticker AAPL --model lr --period 2y
```

### 2. Predict next close
```bash
python stock_telegram_bot.py predict --ticker AAPL --model lr --period 2y
```

### 3. Backtest strategy
```bash
python stock_telegram_bot.py backtest --ticker AAPL --period 2y --sma_short 20 --sma_long 50 --rsi_buy 30 --rsi_sell 70
```

### 4. Run Telegram bot
Set environment variables first:
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

Run the bot loop (every 60 min):
```bash
python stock_telegram_bot.py run-bot --ticker AAPL --interval 60 --model lr
```

## Project Structure
```
stock_telegram_bot.py    # Single-file project with everything
models/                  # Saved models (created after training)
README.md
requirements.txt
```

## Requirements
- Python 3.8+
- See `requirements.txt`

## License
MIT License

