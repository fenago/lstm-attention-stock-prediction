"""
Run LSTM with Attention for Article Generation
This script runs the corrected implementation and captures all results
"""

import sys
import os
import json
from datetime import datetime
from lstm_attention_stock_prediction import StockPredictorLSTMAttention
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configuration
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
SEQUENCE_LENGTH = 60
PREDICTION_DAYS = 4
FEATURES = ['Close']  # Start with Close only for comparison

# Store results
results = {
    'config': {
        'ticker': TICKER,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_days': PREDICTION_DAYS,
        'features': FEATURES
    },
    'data_info': {},
    'metrics': {},
    'predictions': {},
    'training_info': {}
}

print("="*80)
print("LSTM with Attention - Stock Price Prediction")
print("Article Generation Run")
print("="*80)

# Initialize predictor
predictor = StockPredictorLSTMAttention(
    sequence_length=SEQUENCE_LENGTH,
    prediction_days=PREDICTION_DAYS,
    features=FEATURES
)

# Fetch data
print(f"\n1. Fetching {TICKER} data...")
data = predictor.fetch_data(TICKER, START_DATE, END_DATE)
results['data_info']['total_samples'] = len(data)
results['data_info']['date_range'] = f"{data.index[0].date()} to {data.index[-1].date()}"
print(f"   Total trading days: {len(data)}")
print(f"   Date range: {results['data_info']['date_range']}")

# Prepare data
print("\n2. Preparing data...")
X_train, y_train, X_test, y_test, test_dates = predictor.prepare_data(data, train_split=0.8)

results['data_info']['train_samples'] = X_train.shape[0]
results['data_info']['test_samples'] = X_test.shape[0]
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# Split for validation
val_split = 0.2
val_idx = int(len(X_train) * (1 - val_split))
X_train_final = X_train[:val_idx]
y_train_final = y_train[:val_idx]
X_val = X_train[val_idx:]
y_val = y_train[val_idx:]

print(f"   Validation samples: {X_val.shape[0]}")

# Build model
print("\n3. Building model...")
model = predictor.build_model(lstm_units=[64, 32], dropout_rate=0.2)
total_params = model.count_params()
results['model_info'] = {
    'total_parameters': int(total_params),
    'lstm_units': [64, 32],
    'dropout_rate': 0.2
}
print(f"   Total parameters: {total_params:,}")

# Train model
print("\n4. Training model...")
history = predictor.train(
    X_train_final, y_train_final,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    verbose=1
)

# Store training history
results['training_info']['epochs_trained'] = len(history.history['loss'])
results['training_info']['final_train_loss'] = float(history.history['loss'][-1])
results['training_info']['final_val_loss'] = float(history.history['val_loss'][-1])
results['training_info']['final_train_mae'] = float(history.history['mae'][-1])
results['training_info']['final_val_mae'] = float(history.history['val_mae'][-1])

print(f"\n   Training completed:")
print(f"   Final train loss: {results['training_info']['final_train_loss']:.6f}")
print(f"   Final val loss: {results['training_info']['final_val_loss']:.6f}")

# Plot training history
print("\n5. Generating training history plot...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_title('Model MAE Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('article_training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: article_training_history.png")

# Evaluate on test set
print("\n6. Evaluating on test set...")
metrics, predictions, actuals = predictor.evaluate(X_test, y_test)

# Store metrics
for feature, metric_dict in metrics.items():
    results['metrics'][feature] = {
        'MAE': float(metric_dict['MAE']),
        'RMSE': float(metric_dict['RMSE']),
        'R2': float(metric_dict['R2'])
    }

print(f"\n   Test Set Results:")
print(f"   MAE:  {metrics['Close']['MAE']:.4f}")
print(f"   RMSE: {metrics['Close']['RMSE']:.4f}")
print(f"   R²:   {metrics['Close']['R2']:.4f}")

# Plot predictions
print("\n7. Generating predictions plot...")
print(f"   test_dates length: {len(test_dates)}")
print(f"   predictions length: {len(predictions)}")
print(f"   actuals length: {len(actuals)}")

# Fix: Use only the matching lengths
num_to_plot = min(len(test_dates), len(predictions), len(actuals))
test_dates_plot = test_dates[:num_to_plot]
predictions_plot = predictions[:num_to_plot]
actuals_plot = actuals[:num_to_plot]

plt.figure(figsize=(16, 7))

# Plot historical data (last 300 days for context)
plt.plot(data.index[-300:], data['Close'].values[-300:],
         label='Historical Price', color='#2E86AB', alpha=0.8, linewidth=2)

# Plot actual test values
plt.plot(test_dates_plot, actuals_plot[:, 0],
         label='Actual (Test Set)', color='#06A77D', marker='o',
         markersize=4, linewidth=2, alpha=0.9)

# Plot predictions
plt.plot(test_dates_plot, predictions_plot[:, 0],
         label='Predicted (Test Set)', color='#D62828', marker='x',
         markersize=5, linewidth=2, alpha=0.9)

# Add vertical line for train/test split
split_date = test_dates_plot[0]
plt.axvline(x=split_date, color='black', linestyle='--', linewidth=2,
            label='Train/Test Split', alpha=0.6)

plt.title(f'{TICKER} Stock Price Prediction - LSTM with Attention',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Price ($)', fontsize=13)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('article_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: article_predictions.png")

# Zoomed-in plot of predictions
print("\n8. Generating zoomed prediction plot...")
plt.figure(figsize=(16, 7))

# Plot last 60 days of historical + all test
context_days = 60
context_start = len(data) - len(test_dates_plot) - context_days
context_dates = data.index[context_start:]
context_prices = data['Close'].values[context_start:]

plt.plot(context_dates[:context_days], context_prices[:context_days],
         label='Historical Price', color='#2E86AB', linewidth=2.5, alpha=0.8)

plt.plot(test_dates_plot, actuals_plot[:, 0],
         label='Actual (Test Set)', color='#06A77D', marker='o',
         markersize=5, linewidth=2.5, alpha=0.9)

plt.plot(test_dates_plot, predictions_plot[:, 0],
         label='Predicted (Test Set)', color='#D62828', marker='x',
         markersize=6, linewidth=2.5, alpha=0.9)

plt.axvline(x=split_date, color='black', linestyle='--', linewidth=2,
            label='Train/Test Split', alpha=0.6)

plt.title(f'{TICKER} Stock Price - Detailed View of Predictions',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Price ($)', fontsize=13)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('article_predictions_zoomed.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: article_predictions_zoomed.png")

# Error distribution plot
print("\n9. Generating error distribution plot...")
errors = actuals_plot[:, 0] - predictions_plot[:, 0]
percentage_errors = (errors / actuals_plot[:, 0]) * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Absolute errors
axes[0].hist(errors, bins=50, color='#D62828', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[0].set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Error ($)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Percentage errors
axes[1].hist(percentage_errors, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[1].set_title('Distribution of Percentage Errors', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Error (%)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('article_error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: article_error_distribution.png")

results['error_stats'] = {
    'mean_error': float(errors.mean()),
    'std_error': float(errors.std()),
    'mean_percentage_error': float(percentage_errors.mean()),
    'std_percentage_error': float(percentage_errors.std())
}

# Predict next 4 days
print("\n10. Predicting next 4 trading days...")
future_predictions = predictor.predict_next_n_days(data, n_days=4)

last_date = data.index[-1]
last_price = data['Close'].iloc[-1]

print(f"\n   Last known date: {last_date.date()}")
print(f"   Last known price: ${last_price:.2f}")
print(f"\n   Future Predictions:")

future_pred_list = []
for i, pred in enumerate(future_predictions, 1):
    pred_price = pred[0]
    change = pred_price - last_price
    pct_change = (change / last_price) * 100
    print(f"   Day {i}: ${pred_price:.2f} (change: {change:+.2f}, {pct_change:+.2f}%)")

    future_pred_list.append({
        'day': i,
        'price': float(pred_price),
        'change': float(change),
        'pct_change': float(pct_change)
    })

results['predictions']['last_known_price'] = float(last_price)
results['predictions']['last_known_date'] = str(last_date.date())
results['predictions']['future_predictions'] = future_pred_list

# Save model
print("\n11. Saving model and scaler...")
predictor.save_model('article_model.h5', 'article_scaler.pkl')

# Save results to JSON
print("\n12. Saving results to JSON...")
with open('article_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("   Saved: article_results.json")

print("\n" + "="*80)
print("ARTICLE GENERATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - article_training_history.png")
print("  - article_predictions.png")
print("  - article_predictions_zoomed.png")
print("  - article_error_distribution.png")
print("  - article_model.h5")
print("  - article_scaler.pkl")
print("  - article_results.json")
print("\n" + "="*80)
