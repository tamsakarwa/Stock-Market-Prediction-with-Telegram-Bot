"""
Author: Tamsa Karwa
Module: Data Preprocessing

This script loads and scales historical stock market data.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df = df[['Close']]
    scaler = MinMaxScaler()
    df['Scaled'] = scaler.fit_transform(df[['Close']])
    return df, scaler
