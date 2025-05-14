"""
Author: Tamsa Karwa
Module: Telegram Bot

Sends stock prediction messages to a Telegram user.
"""

import telegram
import time
from src.data_preprocessing import load_and_preprocess
from src.prediction_service import predict_next

# Replace these with your actual bot token and chat ID
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"

bot = telegram.Bot(token=TOKEN)

def notify_user(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

def run_bot():
    df, scaler = load_and_preprocess('data/historical_stock_data.csv')
    prediction = predict_next(df['Scaled'].values.tolist())
    predicted_price = scaler.inverse_transform([[prediction]])[0][0]
    notify_user(f"📈 Hello from Tamsa Karwa's bot! Predicted next closing price: ${predicted_price:.2f}")

if __name__ == "__main__":
    while True:
        run_bot()
        time.sleep(86400)  # Sleep for 1 day
