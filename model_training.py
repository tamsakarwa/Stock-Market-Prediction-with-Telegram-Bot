"""
Author: Tamsa Karwa
Module: Model Training

Trains a Linear Regression model on stock data and saves it.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump
import os

def create_dataset(data, window=5):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

def train_model(df):
    data = df['Scaled'].values
    X, y = create_dataset(data)
    model = LinearRegression()
    model.fit(X, y)
    os.makedirs('model', exist_ok=True)
    dump(model, 'model/stock_model.joblib')
    print("✅ Model trained and saved as 'model/stock_model.joblib'")
