"""
Author: Tamsa Karwa
Module: Prediction Service

Uses trained model to make predictions based on recent stock prices.
"""

from joblib import load
import numpy as np

def predict_next(prices, window=5):
    model = load('model/stock_model.joblib')
    input_data = np.array(prices[-window:]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]
