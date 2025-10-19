"""
Quick Example - LSTM with Attention for Stock Prediction
=========================================================

This demonstrates the simplest usage of the corrected LSTM implementation.
"""

from lstm_attention_stock_prediction import StockPredictorLSTMAttention

# 1. Initialize predictor with configuration
predictor = StockPredictorLSTMAttention(
    sequence_length=60,              # Look back 60 days
    prediction_days=4,               # Predict 4 days ahead
    features=['Close']               # Start simple with just Close price
    # features=['Open', 'High', 'Low', 'Close', 'Volume']  # Or use all features
)

# 2. Fetch historical data
print("Fetching data...")
data = predictor.fetch_data(
    ticker='AAPL',
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# 3. Prepare data with PROPER splitting (no data leakage!)
print("Preparing data...")
X_train, y_train, X_test, y_test, test_dates = predictor.prepare_data(
    data,
    train_split=0.8  # 80% train, 20% test
)

# 4. Build model with working attention mechanism
print("Building model...")
model = predictor.build_model(
    lstm_units=[64, 32],   # Two LSTM layers
    dropout_rate=0.2       # 20% dropout
)
print(model.summary())

# 5. Train the model
print("\nTraining model...")
history = predictor.train(
    X_train, y_train,
    epochs=50,              # Reduced for quick demo
    batch_size=32,
    verbose=1
)

# 6. Visualize training history
predictor.plot_training_history(history)

# 7. Evaluate on test set
print("\nEvaluating on test set...")
metrics, predictions, actuals = predictor.evaluate(X_test, y_test)

# 8. Plot predictions vs actual
predictor.plot_predictions(
    data, test_dates, predictions, actuals,
    feature_idx=0,  # Close price
    save_path='quick_example_predictions.png'
)

# 9. Predict next 4 days (using SAVED scaler - no data leakage!)
print("\nPredicting next 4 trading days...")
future_predictions = predictor.predict_next_n_days(data, n_days=4)

last_date = data.index[-1]
last_price = data['Close'].iloc[-1]

print(f"\nLast known date: {last_date.date()}")
print(f"Last known price: ${last_price:.2f}")
print("\nPredictions:")
for i, pred in enumerate(future_predictions, 1):
    print(f"  Day {i}: ${pred[0]:.2f}")

# 10. Save model and scaler for later use
print("\nSaving model and scaler...")
predictor.save_model(
    model_path='aapl_lstm_model.h5',
    scaler_path='aapl_scaler.pkl'
)

print("\nDone! Model saved successfully.")

# Optional: Load and use saved model
# predictor_new = StockPredictorLSTMAttention(sequence_length=60, prediction_days=4, features=['Close'])
# predictor_new.load_model('aapl_lstm_model.h5', 'aapl_scaler.pkl')
# new_predictions = predictor_new.predict_next_n_days(data, n_days=4)
