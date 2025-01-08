# KAMA and Regime Trading Strategy Research & Backtest

This project implements a trading strategy that combines a technical indicator (KAMA - Kaufman’s Adaptive Moving Average) with a regime signal derived from Markov Models. The script evaluates the strategy through extensive backtesting and optimization.

## Features
- **Technical Indicator**: Optimized KAMA signals with varying parameters.
- **Regime Modeling**: Hidden Markov Models and state classification.
- **Feature Engineering**: Time-series features using `tsfresh`, `ta`, and custom transformations.
- **Backtesting**: Transaction cost modeling and strategy evaluation.
- **Visualization**: Performance plots and feature importance.

## Prerequisites
The following Python packages are required:
- `numpy`
- `pandas`
- `matplotlib`
- `optuna`
- `ta`
- `hmmlearn`
- `scipy`
- `sklearn`
- `statsmodels`
- `pywavelets`
- `tsfresh`
- `tsfracdiff`

To install all dependencies, use:
```bash
pip install -r requirements.txt
