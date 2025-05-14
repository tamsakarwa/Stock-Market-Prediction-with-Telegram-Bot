# Stock Market Prediction with Telegram Bot

## Project Overview
The system is designed to:

- Load and preprocess historical stock market data
- Train a machine learning model (Linear Regression) to predict future stock prices
- Notify users of predictions using a Telegram bot
- Allow simple daily updates without manual intervention

This project demonstrates a practical application of ML for financial forecasting, with emphasis on usability and automation.

## Directory Structure
The **Stock Market Prediction Telegram Bot** project is structured into several directories and files, each serving a specific purpose to ensure modularity and clarity. The `data/` folder contains historical stock data in CSV format, essential for training and evaluation. The `notebooks/` directory holds Jupyter notebooks used for exploratory data analysis (EDA). The core functionality resides in the `src/` directory, which includes multiple Python scripts: `data_preprocessing.py` handles data cleaning and scaling, `model_training.py` manages model training and saves the trained model, `prediction_service.py` is responsible for making predictions using the trained model, and `telegram_bot.py` sends real-time stock alerts via Telegram. The trained model is stored in the `model/` directory. Project dependencies are listed in the `requirements.txt` file, while the `README.md` provides comprehensive documentation. The `.gitignore` file ensures that unnecessary or sensitive files are excluded from version control.

## How to Use

### 1. Clone the repository

bash
git clone https://github.com/TamsaKarwa/Stock-Market-Prediction-Telegram-Bot.git
cd Stock-Market-Prediction-Telegram-Bot
`
### 2. Install dependencies

Make sure Python 3 is installed, then run:

bash
pip install -r requirements.txt

### 3. Prepare your dataset

By default, a sample dataset is included. If you want to use your own data, you can download it using [Yahoo Finance](https://finance.yahoo.com) or the `yfinance` package:

python
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2023-12-31")
df.to_csv("data/historical_stock_data.csv")

### 4. Train the model

This will save a trained model file into the `model/` directory:

bash
python src/model_training.py

### 5. Configure the Telegram bot

* Create a bot using [@BotFather](https://t.me/BotFather) on Telegram
* Get your **Bot Token** and **Chat ID**
* Edit the following lines in `src/telegram_bot.py`:

python
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"

### 6. Start the Telegram bot

bash
python src/telegram_bot.py


It will send daily predictions to your Telegram based on the latest model.

## Technologies Used

* Python
* pandas, numpy
* scikit-learn
* matplotlib (for EDA)
* yfinance (optional, for data collection)
* python-telegram-bot

## Future Enhancements

* Switch to time-series models like LSTM or ARIMA
* Add web interface for input/output
* Monitor multiple stock tickers
* Use job scheduling or cloud hosting for deployment

## License

This project is open source 
