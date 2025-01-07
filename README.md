# KAMA and Regime Trading Strategy Research & Backtest

This project implements a trading strategy that combines a technical indicator (KAMA - Kaufman’s Adaptive Moving Average) with a regime signal derived from Markov Models. The script evaluates the strategy through extensive backtesting and optimization.

## Features
- **Technical Indicator**: Optimized KAMA signals with varying parameters.
- **Regime Modeling**: Hidden Markov Models and state classification.
- **Feature Engineering**: Standardization, rolling statistics, and more.
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

To install all dependencies, use:
```bash
pip install -r requirements.txt
```

## Usage

### Clone the Repository
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/kama-regime-strategy
   cd kama-regime-strategy
   ```

### Prepare Data
2. Place your data files (e.g., `SPX.csv`, `VIX.csv`) in the appropriate directory. Update the file paths in the script if necessary.

### Run the Script
3. Execute the script to perform the backtest:
   ```bash
   python kama_model.py
   ```

## File Structure
- `kama_model.py`: The main script for loading data, feature engineering, model training, and evaluation.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Results
The script generates:
- Optimized parameters for KAMA-based trading strategies.
- Evaluation metrics (e.g., Sharpe ratio, compounded returns).
- Plots for strategy performance and feature importance.

