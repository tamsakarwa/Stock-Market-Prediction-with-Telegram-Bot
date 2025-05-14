# Stock Market Prediction with Telegram Bot

## Project Overview
The system is designed to:

- Load and preprocess historical stock market data
- Train a machine learning model (Linear Regression) to predict future stock prices
- Notify users of predictions using a Telegram bot
- Allow simple daily updates without manual intervention

This project demonstrates a practical application of ML for financial forecasting, with emphasis on usability and automation.

## Directory Structure
Stock-Market-Prediction-Telegram-Bot/
│
├── data/                        # Historical stock data (CSV)
├── notebooks/                  # Jupyter notebook for EDA
├── src/                        # Source code
│   ├── data\_preprocessing.py   # Data cleaning and scaling
│   ├── model\_training.py       # Model training and saving
│   ├── prediction\_service.py   # Prediction using trained model
│   └── telegram\_bot.py         # Sends alerts through Telegram
├── model/                      # Trained model is saved here
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
└── .gitignore                  # Prevents versioning of unnecessary files

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

## About the Developer

This project was created by **Tamsa Karwa**, a software developer with a focus on machine learning and automation. It was developed as part of a larger interest in using AI to simplify decision-making.

You can reach out via GitHub or connect on LinkedIn.

## Future Enhancements

* Switch to time-series models like LSTM or ARIMA
* Add web interface for input/output
* Monitor multiple stock tickers
* Use job scheduling or cloud hosting for deployment

## License

This project is open source 
