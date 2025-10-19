"""
Generate Visualization from Saved Production Models
====================================================

Loads the trained production ensemble and generates visualization.

Author: Dr. Ernesto Lee | drlee.io
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from production_lstm_predictor import ProductionStockPredictor

print("="*80)
print("GENERATING PRODUCTION MODEL VISUALIZATION")
print("="*80)

# Configuration
TICKER = 'AAPL'
START_DATE = '2023-01-01'
END_DATE = datetime(2025, 10, 19).strftime('%Y-%m-%d')

# Initialize predictor
predictor = ProductionStockPredictor(
    sequence_length=60,
    use_ensemble=True,
    n_models=3
)

# Load saved models
print("\nLoading saved production ensemble...")
predictor.load_ensemble('production')
print("✅ Models loaded successfully")

# Fetch and prepare data
print("\nFetching data...")
data = predictor.fetch_and_prepare_data(TICKER, START_DATE, END_DATE)

# Prepare sequences
X_train, y_train, X_test, y_test, test_dates, scaler = predictor.prepare_sequences(
    data, train_split=0.85, predict_direction=True
)

# Evaluate
print("\nEvaluating on test set...")
results = predictor.evaluate(X_test, y_test)

predictions = results['predictions']
probabilities = results['probabilities'].flatten()

# Generate visualization
print("\nGenerating visualization...")
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Predictions over time
ax1 = axes[0]
correct = predictions == y_test
dates = test_dates[:len(predictions)]

# Use numpy arrays to ensure proper indexing
correct_indices = np.where(correct)[0]
wrong_indices = np.where(~correct)[0]

ax1.scatter(dates[correct_indices], correct_indices,
           color='green', alpha=0.6, s=50, label='Correct Prediction')
ax1.scatter(dates[wrong_indices], wrong_indices,
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
window = min(50, len(dates))
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

# Print results
print("\n" + "="*80)
print("FINAL RESULTS")
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
print("VISUALIZATION COMPLETE")
print("="*80)
