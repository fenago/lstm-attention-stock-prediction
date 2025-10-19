# LSTM with Attention Mechanism for Stock Prediction - CORRECTED VERSION

## Overview

This is a **corrected and improved** implementation of the LSTM with Attention mechanism for stock price prediction, based on the article from drlee.io. The original article contained several critical errors that have been fixed in this version.

## Critical Fixes Applied

### 1. ✅ Fixed Broken Attention Mechanism

**Original Problem:**
```python
# This code DOES NOT WORK in Sequential models
attention = AdditiveAttention(name='attention_weight')
model.add(Permute((2, 1)))
attention_result = attention([model.output, model.output])  # ❌ Invalid
```

**Fixed Solution:**
```python
# Using Functional API properly
inputs = Input(shape=(sequence_length, n_features))
lstm_out = LSTM(units, return_sequences=True)(inputs)
attention_out = AdditiveAttention()([lstm_out, lstm_out])  # ✅ Works!
```

### 2. ✅ Fixed Scaler Inconsistency (Data Leakage)

**Original Problem:**
- Training used one scaler fitted on 2020-2024 data
- Prediction created a NEW scaler fitted on different data
- This causes **wrong predictions** due to different min/max values

**Fixed Solution:**
- Scaler fitted ONCE on training data
- Scaler saved and reused for all predictions
- Methods to save/load scaler with model

### 3. ✅ Prevented Look-Ahead Bias

**Original Problem:**
```python
# Scaler sees ALL data (including test)
scaler.fit_transform(all_data)
# THEN split train/test
train_test_split()
```

**Fixed Solution:**
```python
# Split FIRST
train, test = split_data()
# Fit scaler on training data ONLY
scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
```

### 4. ✅ Proper Test Data Preparation

**Original Problem:**
- `X_test` and `y_test` were never converted to numpy arrays or reshaped
- Code would crash when trying to evaluate

**Fixed Solution:**
- All data properly converted and reshaped
- Proper 3D shape `[samples, sequence_length, features]` for LSTM

### 5. ✅ Added Walk-Forward Validation

**Original Problem:**
- No proper backtesting mechanism
- Can't validate predictions on historical data
- Confusion between predicting unknown future vs validating on past

**Fixed Solution:**
- `backtest_predictions()` method for walk-forward validation
- Test predictions on historical data we held out
- Proper temporal validation

### 6. ✅ Multi-Feature Support

**Original Problem:**
- Only used Close price
- Missed valuable information from OHLCV data

**Fixed Solution:**
- Support for multiple features: Open, High, Low, Close, Volume
- Easy to add technical indicators
- Predicts all features simultaneously

### 7. ✅ Fixed Date Handling

**Original Problem:**
- No proper date shifting for validation
- Couldn't validate "future" predictions

**Fixed Solution:**
- Proper date indexing for test set
- Backtest function works with historical dates
- Can validate predictions against known outcomes

## Installation

```bash
# Install required packages
pip install tensorflow keras yfinance numpy pandas matplotlib scikit-learn
```

## Quick Start

```python
from lstm_attention_stock_prediction import StockPredictorLSTMAttention

# Initialize predictor
predictor = StockPredictorLSTMAttention(
    sequence_length=60,
    prediction_days=4,
    features=['Open', 'High', 'Low', 'Close', 'Volume']
)

# Fetch data
data = predictor.fetch_data('AAPL', '2020-01-01', '2024-01-01')

# Prepare data (proper splitting!)
X_train, y_train, X_test, y_test, test_dates = predictor.prepare_data(data)

# Build model (working attention!)
model = predictor.build_model(lstm_units=[64, 32])

# Train
history = predictor.train(X_train, y_train, epochs=100)

# Evaluate
metrics, predictions, actuals = predictor.evaluate(X_test, y_test)

# Predict future (using saved scaler!)
future_predictions = predictor.predict_next_n_days(data, n_days=4)

# Save model and scaler
predictor.save_model('model.h5', 'scaler.pkl')
```

## Running the Complete Example

```bash
python lstm_attention_stock_prediction.py
```

This will:
1. Download AAPL data
2. Train the model with proper validation
3. Evaluate on test set
4. Generate prediction plots
5. Perform walk-forward backtesting
6. Save model and scaler

## Key Features

### Proper Data Pipeline
- ✅ No look-ahead bias
- ✅ Scaler preservation
- ✅ Proper train/val/test split
- ✅ Multi-feature support

### Working Attention Mechanism
- ✅ Built with Functional API
- ✅ Self-attention over LSTM outputs
- ✅ Actually integrated into model

### Robust Evaluation
- ✅ Multiple metrics (MAE, RMSE, R²)
- ✅ Walk-forward validation
- ✅ Backtesting on historical data
- ✅ Visualization tools

### Production Ready
- ✅ Model and scaler saving/loading
- ✅ Proper error handling
- ✅ Configurable parameters
- ✅ Reproducible results

## Model Architecture

```
Input (60, n_features)
    ↓
LSTM Layer 1 (64 units, return_sequences=True)
    ↓
Dropout + BatchNorm
    ↓
LSTM Layer 2 (32 units, return_sequences=True)
    ↓
Dropout + BatchNorm
    ↓
Attention Mechanism (Self-Attention)
    ↓
Concatenate [LSTM + Attention]
    ↓
Global Average Pooling
    ↓
Dense (32 units, ReLU)
    ↓
Output (n_features)
```

## Understanding the Fixes

### Why the Original Code Didn't Work

1. **Sequential API Limitation**: You can't use `model.output` in Sequential models mid-construction
2. **Scaler Mismatch**: New data scaled differently than training data
3. **Data Leakage**: Test data statistics leaked into training
4. **Missing Validation**: No way to validate predictions properly

### Why This Version Works

1. **Functional API**: Proper layer connections and attention integration
2. **Single Scaler**: Same transformation for all data
3. **Proper Splitting**: No information leakage
4. **Backtesting**: Validate on historical data

## Performance Expectations

With proper implementation, you should see:
- **MAE**: 2-5% of stock price (e.g., $3-8 for AAPL at $150)
- **R²**: 0.85-0.95 on test set
- **RMSE**: 3-6% of stock price

Note: The original article showed unrealistic metrics due to data leakage.

## Customization

### Use Different Features

```python
# Just Close price (like original)
features = ['Close']

# OHLCV (recommended)
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Add technical indicators
data['SMA_20'] = data['Close'].rolling(20).mean()
data['RSI'] = calculate_rsi(data['Close'])
features = ['Close', 'Volume', 'SMA_20', 'RSI']
```

### Adjust Model Architecture

```python
model = predictor.build_model(
    lstm_units=[128, 64, 32],  # 3 LSTM layers
    dropout_rate=0.3           # Higher dropout
)
```

### Change Prediction Horizon

```python
predictor = StockPredictorLSTMAttention(
    sequence_length=90,    # Look back 90 days
    prediction_days=10     # Predict 10 days ahead
)
```

## Backtesting Example

```python
# Test predictions on historical data
backtest_results = predictor.backtest_predictions(
    data=data,
    start_date='2023-06-01',
    end_date='2023-12-01',
    step_days=5  # Make prediction every 5 days
)

# Analyze results
for result in backtest_results:
    print(f"Predicted on: {result['prediction_date']}")
    print(f"Predictions: {result['predictions']}")
    print(f"Actuals: {result['actuals']}")
```

## Important Notes

### What This Model Can and Cannot Do

✅ **CAN**:
- Learn temporal patterns from historical data
- Capture short-term trends and momentum
- Provide probabilistic estimates of future prices
- Help with risk assessment

❌ **CANNOT**:
- Predict market crashes or black swan events
- Account for news and external events
- Guarantee profits (markets are not fully predictable)
- Replace fundamental analysis

### Limitations

1. **Past performance ≠ future results**: Historical patterns may not repeat
2. **External factors**: News, earnings, macro events not captured
3. **Iterative predictions**: Errors compound when predicting multiple days
4. **Market efficiency**: If patterns were perfectly predictable, they'd be arbitraged away

## Comparison: Original vs Corrected

| Aspect | Original Article | This Version |
|--------|-----------------|--------------|
| Attention mechanism | ❌ Broken | ✅ Working |
| Scaler handling | ❌ Creates new scaler | ✅ Reuses saved scaler |
| Data leakage | ❌ Yes (test in scaler) | ✅ No leakage |
| Test data prep | ❌ Missing reshape | ✅ Properly prepared |
| Validation | ❌ No backtesting | ✅ Walk-forward validation |
| Features | ❌ Close only | ✅ Multi-feature support |
| Production ready | ❌ No | ✅ Yes (save/load) |

## References

- Original Article: [drlee.io](https://drlee.io/advanced-stock-pattern-prediction-using-lstm-with-the-attention-mechanism-in-tensorflow-a-step-by-143a2e8b0e95)
- TensorFlow Functional API: [tensorflow.org/guide/keras/functional](https://www.tensorflow.org/guide/keras/functional)
- Time Series Validation: [scikit-learn.org/stable/modules/cross_validation.html#time-series-split](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

## Contributing

Feel free to improve this implementation by:
- Adding more technical indicators
- Implementing ensemble methods
- Adding confidence intervals
- Improving the attention mechanism
- Adding more evaluation metrics

## License

MIT License - Feel free to use and modify for your projects.

## Disclaimer

**This code is for educational purposes only. Stock market prediction is inherently uncertain. Do not use this for actual trading without proper risk management and professional financial advice.**
