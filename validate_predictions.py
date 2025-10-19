"""
Validate LSTM Predictions on Recent Data
=========================================

This script:
1. Fetches data up to TODAY (October 19, 2025)
2. Makes predictions for last few days
3. Compares predictions vs actual prices
4. Shows real-world accuracy

This demonstrates how well the model actually works!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lstm_attention_stock_prediction import StockPredictorLSTMAttention

# Today's date
TODAY = datetime(2025, 10, 19)
print(f"Validation Date: {TODAY.strftime('%Y-%m-%d')} (Sunday)")
print("="*80)

# Configuration
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = TODAY.strftime('%Y-%m-%d')
VALIDATION_DAYS = 10  # Validate last 10 trading days

print(f"\nFetching {TICKER} data up to {END_DATE}...")

# Initialize predictor
predictor = StockPredictorLSTMAttention(
    sequence_length=60,
    prediction_days=1,  # Predict 1 day ahead for accuracy
    features=['Close']
)

# Fetch ALL available data
data = predictor.fetch_data(TICKER, START_DATE, END_DATE)

print(f"Total trading days: {len(data)}")
print(f"Latest data: {data.index[-1].strftime('%Y-%m-%d')}")
print(f"Latest price: ${float(data['Close'].iloc[-1]):.2f}")

# Prepare data (80/20 split)
X_train, y_train, X_test, y_test, test_dates = predictor.prepare_data(data, train_split=0.8)

# Split for validation
val_split = 0.2
val_idx = int(len(X_train) * (1 - val_split))
X_train_final = X_train[:val_idx]
y_train_final = y_train[:val_idx]
X_val = X_train[val_idx:]
y_val = y_train[val_idx:]

# Build and train model
print("\nTraining model...")
model = predictor.build_model(lstm_units=[64, 32], dropout_rate=0.2)

history = predictor.train(
    X_train_final, y_train_final,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    verbose=0  # Silent training
)

print("Training complete!")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")

# Now validate on RECENT days
print("\n" + "="*80)
print("VALIDATING ON RECENT DAYS")
print("="*80)

# Get last N trading days for validation
validation_results = []

for days_ago in range(VALIDATION_DAYS, 0, -1):
    # Get data up to N days ago
    cutoff_idx = len(data) - days_ago
    historical_data = data.iloc[:cutoff_idx]

    if len(historical_data) < predictor.sequence_length + 1:
        continue

    # Predict next day
    prediction = predictor.predict_next_n_days(historical_data, n_days=1)
    predicted_price = prediction[0][0]

    # Get actual next day price
    actual_date = data.index[cutoff_idx]
    actual_price = float(data['Close'].iloc[cutoff_idx])

    # Get previous day for direction
    prev_price = float(data['Close'].iloc[cutoff_idx - 1])

    # Calculate metrics
    error = predicted_price - actual_price
    abs_error = abs(error)
    pct_error = (error / actual_price) * 100

    # Direction accuracy
    predicted_direction = "UP" if predicted_price > prev_price else "DOWN"
    actual_direction = "UP" if actual_price > prev_price else "DOWN"
    direction_correct = predicted_direction == actual_direction

    validation_results.append({
        'date': actual_date,
        'prev_price': prev_price,
        'actual_price': actual_price,
        'predicted_price': predicted_price,
        'error': error,
        'abs_error': abs_error,
        'pct_error': pct_error,
        'predicted_direction': predicted_direction,
        'actual_direction': actual_direction,
        'direction_correct': direction_correct
    })

# Create DataFrame
results_df = pd.DataFrame(validation_results)

# Print results
print("\nRecent Predictions vs Actual Prices:\n")
print(f"{'Date':<12} {'Prev':<8} {'Actual':<8} {'Predicted':<10} {'Error':<10} {'% Error':<10} {'Direction'}")
print("-" * 85)

for idx, row in results_df.iterrows():
    direction_symbol = "✅" if row['direction_correct'] else "❌"
    print(f"{row['date'].strftime('%Y-%m-%d'):<12} "
          f"${row['prev_price']:<7.2f} "
          f"${row['actual_price']:<7.2f} "
          f"${row['predicted_price']:<9.2f} "
          f"${row['error']:+8.2f} "
          f"{row['pct_error']:+8.2f}% "
          f"{direction_symbol} {row['predicted_direction']} (actual: {row['actual_direction']})")

# Calculate overall metrics
print("\n" + "="*80)
print("OVERALL METRICS")
print("="*80)

mae = results_df['abs_error'].mean()
rmse = np.sqrt((results_df['error'] ** 2).mean())
mape = results_df['pct_error'].abs().mean()
direction_accuracy = (results_df['direction_correct'].sum() / len(results_df)) * 100

print(f"\nPrice Prediction:")
print(f"  Mean Absolute Error:  ${mae:.2f}")
print(f"  Root Mean Squared Error: ${rmse:.2f}")
print(f"  Mean Absolute % Error: {mape:.2f}%")

print(f"\nDirection Prediction:")
print(f"  Direction Accuracy: {direction_accuracy:.1f}%")
print(f"  Correct: {results_df['direction_correct'].sum()}/{len(results_df)}")

# Plot recent predictions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Prices
ax1.plot(results_df['date'], results_df['actual_price'],
         'o-', color='green', label='Actual Price', linewidth=2, markersize=6)
ax1.plot(results_df['date'], results_df['predicted_price'],
         'x--', color='red', label='Predicted Price', linewidth=2, markersize=8)
ax1.set_title(f'{TICKER} - Recent Predictions vs Actual (Last {VALIDATION_DAYS} Trading Days)',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Plot 2: Errors
ax2.bar(results_df['date'], results_df['pct_error'],
        color=['red' if x < 0 else 'green' for x in results_df['pct_error']],
        alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_title('Prediction Errors (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Error (%)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('validation_recent_predictions.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization: validation_recent_predictions.png")

# Predict FUTURE (next 4 days from today)
print("\n" + "="*80)
print("FUTURE PREDICTIONS (Next 4 Trading Days)")
print("="*80)

future_predictions = predictor.predict_next_n_days(data, n_days=4)
last_price = float(data['Close'].iloc[-1])
last_date = data.index[-1]

print(f"\nPredicting from: {last_date.strftime('%Y-%m-%d')}")
print(f"Last known price: ${last_price:.2f}\n")

print(f"{'Day':<5} {'Date':<12} {'Predicted Price':<16} {'Change':<12} {'% Change':<10} {'Direction'}")
print("-" * 75)

for i, pred in enumerate(future_predictions, 1):
    pred_price = pred[0]
    change = pred_price - last_price
    pct_change = (change / last_price) * 100
    direction = "UP ⬆️" if change > 0 else "DOWN ⬇️"

    # Estimate next trading day (skip weekends)
    next_date = last_date + timedelta(days=i)
    while next_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
        next_date += timedelta(days=1)

    print(f"{i:<5} {next_date.strftime('%Y-%m-%d'):<12} ${pred_price:<15.2f} "
          f"${change:+10.2f} {pct_change:+8.2f}%  {direction}")

    # Update last_price for next iteration
    last_price = pred_price

print("\n" + "="*80)
print("VALIDATION COMPLETE!")
print("="*80)

# Save results
results_df.to_csv('validation_results.csv', index=False)
print("\nResults saved to: validation_results.csv")

# Save model
predictor.save_model('validated_model.h5', 'validated_scaler.pkl')
print("Model saved: validated_model.h5, validated_scaler.pkl")
