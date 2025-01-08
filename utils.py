import pandas as pd
import numpy as np
from ta.momentum import KAMAIndicator

def load_data(data_files):
    """Load datasets and preprocess them."""
    spx = pd.read_csv(data_files['spx'], parse_dates=['Date'], index_col=0).sort_index().astype(float)
    vix = pd.read_csv(data_files['vix'], parse_dates=['Date'], index_col=0).sort_index().astype(float)
    features_5d = pd.read_csv(data_files['features_5d'], parse_dates=['date'], index_col=0).sort_index()
    features_ta = pd.read_csv(data_files['features_ta'], index_col=0, parse_dates=True).sort_index()

    # Preprocessing
    spx.rename(columns={'Last Price': 'price'}, inplace=True)
    spx.drop(['Volume'], axis=1, inplace=True)
    vix.rename(columns={'VIX': 'vix'}, inplace=True)
    vix.drop(['First', 'Last'], axis=1, inplace=True)
    vix = vix.groupby(vix.index).mean()  # Eliminate duplicates

    features_5d.drop(columns=['date', 'underlying', 'price'], inplace=True)

    raw = pd.concat([spx, vix, features_5d, features_ta], axis=1)
    return raw

def kama_signals(data, fast, slow, signal_window):
    """Calculate KAMA signals."""
    kama = KAMAIndicator(close=data, window_slow=slow, window_fast=fast, window=signal_window).kama()
    return kama

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE: {mse}, MAE: {mae}, R^2: {r2}")
