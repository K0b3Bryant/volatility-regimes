import pandas as pd
import numpy as np
import warnings
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame, impute, roll_time_series
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
from tsfracdiff import FractionalDifferentiator

warnings.filterwarnings("ignore")

def generate_features(price_data, window_ideal=5, window_min=5):
    """Generate time series features for the given price data."""

    # Base feature
    feature_base = pd.DataFrame()
    feature_base['feature'] = price_data['price_scaled']

    # TA features
    features_ta = pd.DataFrame()
    features_ta['RSI'] = RSIIndicator(feature_base['feature'], window=14).rsi()
    features_ta['ROC_6M'] = ROCIndicator(feature_base['feature'], window=125).roc()
    features_ta['ROC_12M'] = ROCIndicator(feature_base['feature'], window=250).roc()
    features_ta['BB'] = BollingerBands(feature_base['feature'], window=20, window_dev=2).bollinger_hband()

    # Custom CMO function
    def cmo(close_prices, window=14):
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        sum_gain = gain.rolling(window=window).sum()
        sum_loss = loss.rolling(window=window).sum()
        return ((sum_gain - sum_loss) / (sum_gain + sum_loss)).abs() * 100

    features_ta['CMO'] = cmo(feature_base['feature'], window=14)
    features_ta = features_ta.fillna(method='bfill')

    # tsfresh features
    price_rolled = pd.DataFrame(feature_base['feature'], columns=['feature'])
    price_rolled['date'] = price_rolled.index
    price_rolled['underlying'] = 'spx'

    price_rolled = roll_time_series(
        price_rolled,
        column_id='underlying',
        column_sort='date',
        max_timeshift=window_ideal,
        min_timeshift=window_min
    ).drop('underlying', axis=1)

    features_tsfresh = extract_features(
        price_rolled,
        column_id='id',
        column_sort='date',
        column_value='feature',
        impute_function=impute,
        show_warnings=False
    )

    features_tsfresh.index = features_tsfresh.index.map(lambda x: x[1])
    features_tsfresh.index.name = 'last_date'
    features_tsfresh = features_tsfresh.fillna(method='bfill')

    # Merge features
    features_merge = pd.merge(features_tsfresh, features_ta, left_index=True, right_index=True, how='outer').fillna(method='bfill')

    # Apply fractional differencing
    frac_diff = FractionalDifferentiator()
    features_fd = frac_diff.FitTransform(features_merge)

    return features_merge, features_fd
