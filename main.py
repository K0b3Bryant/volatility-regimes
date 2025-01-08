# -*- coding: utf-8 -*-

from utils import load_data, kama_signals, evaluate_model
from config import data_files, random_state
from features_engineering import generate_features
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    print("Starting KAMA Model Script")

    # Load data
    raw = load_data(data_files)

    # Generate features
    features, features_fd = generate_features(raw)
    print("Features generated successfully.")

    # Example: Generate KAMA signals
    fast, slow, signal_window = 10, 30, 5
    kama = kama_signals(raw['price'], fast, slow, signal_window)

    # Plot KAMA vs. Price
    plt.figure(figsize=(10, 6))
    plt.plot(raw['price'], label='Price')
    plt.plot(kama, label=f'KAMA ({fast}, {slow}, {signal_window})')
    plt.title('Price and KAMA Indicator')
    plt.legend()
    plt.show()

    # Model pipeline (example)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=random_state))
    ])

    # Example evaluation (placeholders for real train/test data)
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    evaluate_model(y_true, y_pred)
