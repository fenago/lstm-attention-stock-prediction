"""
Test Production Model on Oct 2025 Data
=======================================

This script trains and validates the improved model to see if we can
beat 50% accuracy and get to 55-60%+

Author: Dr. Ernesto Lee | drlee.io
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from production_lstm_predictor import ProductionStockPredictor

print("="*80)
print("TESTING PRODUCTION MODEL")
print("="*80)
print("\nGoal: Beat 50% direction accuracy")
print("Target: 55-60%+ for credible performance")
print("="*80)

# Configuration - USE RECENT DATA ONLY (same regime)
TICKER = 'AAPL'
START_DATE = '2023-01-01'  # Recent regime only (~18 months in same price range)
END_DATE = datetime(2025, 10, 19).strftime('%Y-%m-%d')

print(f"\n⚠️  IMPORTANT: Training on recent data only (2023-2025)")
print(f"   This ensures train and test data are in the SAME price regime")
print(f"   More honest evaluation than training on 2020 data!\n")

# Initialize predictor
predictor = ProductionStockPredictor(
    sequence_length=60,
    use_ensemble=True,
    n_models=3  # Start with 3 models (faster training)
)

# Fetch and prepare data
data = predictor.fetch_and_prepare_data(TICKER, START_DATE, END_DATE)

# Prepare sequences for DIRECTION prediction
# Use 85/15 split since we have less data (2023-2025 only)
X_train, y_train, X_test, y_test, test_dates, scaler = predictor.prepare_sequences(
    data, train_split=0.85, predict_direction=True
)

# Store scaler
predictor.scalers.append(scaler)

# Split training data for validation
val_split = 0.2
val_idx = int(len(X_train) * (1 - val_split))
X_train_final = X_train[:val_idx]
y_train_final = y_train[:val_idx]
X_val = X_train[val_idx:]
y_val = y_train[val_idx:]

print(f"\nDataset split:")
print(f"  Training: {len(X_train_final)} sequences")
print(f"  Validation: {len(X_val)} sequences")
print(f"  Test: {len(X_test)} sequences")

# Train ensemble
print("\n" + "="*80)
print("TRAINING ENSEMBLE")
print("="*80)

predictor.train_ensemble(
    X_train_final, y_train_final,
    X_val, y_val,
    epochs=100
)

# Evaluate on test set
print("\n" + "="*80)
print("EVALUATING ON TEST SET (Oct 2025 data)")
print("="*80)

results = predictor.evaluate(X_test, y_test)

# Detailed analysis
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

predictions = results['predictions']
probabilities = results['probabilities'].flatten()

# Get actual dates and prices for analysis
test_data = data.iloc[int(len(data)*0.8) + predictor.sequence_length:]
actual_returns = test_data['Returns'].values

print(f"\nConfidence Analysis:")
high_conf = probabilities[(probabilities > 0.6) | (probabilities < 0.4)]
high_conf_preds = predictions[(probabilities > 0.6) | (probabilities < 0.4)]
high_conf_actuals = y_test[(probabilities > 0.6) | (probabilities < 0.4)]
if len(high_conf) > 0:
    high_conf_acc = (high_conf_preds == high_conf_actuals).sum() / len(high_conf_preds)
    print(f"  High confidence predictions (>60% or <40%): {len(high_conf)}")
    print(f"  High confidence accuracy: {high_conf_acc*100:.2f}%")

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Predictions over time
ax1 = axes[0]
correct = predictions == y_test
# Ensure dates match predictions length
min_len = min(len(test_dates), len(predictions))
dates = test_dates[:min_len]
correct = correct[:min_len]

# Use indices within valid date range
correct_indices = np.where(correct)[0]
wrong_indices = np.where(~correct)[0]

# Only plot points where we have dates
correct_indices = correct_indices[correct_indices < len(dates)]
wrong_indices = wrong_indices[wrong_indices < len(dates)]

ax1.scatter(dates.iloc[correct_indices], correct_indices,
           color='green', alpha=0.6, s=50, label='Correct Prediction')
ax1.scatter(dates.iloc[wrong_indices], wrong_indices,
           color='red', alpha=0.6, s=50, label='Wrong Prediction')
ax1.set_title(f'Prediction Accuracy Over Time - {TICKER}', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Prediction Index')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Plot 2: Confidence distribution
ax2 = axes[1]
ax2.hist(probabilities, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted Probability (UP)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Actual vs Predicted Direction
ax3 = axes[2]
window = min(50, len(dates))  # Last 50 predictions
dates_window = dates[-window:]
actual_window = y_test[-window:]
pred_window = predictions[-window:]

x = range(len(dates_window))
ax3.plot(x, actual_window, 'go-', label='Actual Direction', markersize=8, linewidth=2, alpha=0.7)
ax3.plot(x, pred_window, 'rx--', label='Predicted Direction', markersize=8, linewidth=2, alpha=0.7)
ax3.set_title(f'Last {window} Predictions: Actual vs Predicted Direction', fontsize=14, fontweight='bold')
ax3.set_xlabel('Trading Day')
ax3.set_ylabel('Direction (0=DOWN, 1=UP)')
ax3.set_xticks(range(0, len(dates_window), max(1, len(dates_window)//10)))
ax3.set_xticklabels([dates_window[i].strftime('%m/%d') for i in range(0, len(dates_window), max(1, len(dates_window)//10))], rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('production_model_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved visualization: production_model_results.png")

# Save the model
predictor.save_ensemble('production')
print("✅ Saved production ensemble models")

# Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

accuracy = results['accuracy'] * 100

if accuracy >= 60:
    verdict = "🎉 EXCELLENT! Ready for real use!"
    status = "✅ CREDIBLE"
elif accuracy >= 55:
    verdict = "✅ GOOD! Better than random, potentially profitable"
    status = "✅ ACCEPTABLE"
elif accuracy >= 52:
    verdict = "⚠️  MARGINAL. Slightly better than random"
    status = "⚠️  WEAK"
else:
    verdict = "❌ FAILED. No better than random guessing"
    status = "❌ NOT CREDIBLE"

print(f"\nDirection Accuracy: {accuracy:.2f}%")
print(f"Status: {status}")
print(f"Verdict: {verdict}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
