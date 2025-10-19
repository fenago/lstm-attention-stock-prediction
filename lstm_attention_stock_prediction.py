"""
LSTM with Attention Mechanism for Stock Price Prediction (CORRECTED VERSION)
============================================================================

This corrected implementation fixes critical issues from the original article:
1. Proper attention mechanism using Functional API
2. Scaler preservation and reuse
3. Proper backtesting with walk-forward validation
4. Multi-feature inputs (OHLCV)
5. Correct data preparation for train/test
6. Look-ahead bias prevention
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from keras.layers import AdditiveAttention, Permute, Reshape, Multiply, Concatenate, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class StockPredictorLSTMAttention:
    """
    LSTM with Attention Mechanism for Stock Price Prediction
    """

    def __init__(self, sequence_length=60, prediction_days=4, features=['Close']):
        """
        Initialize the predictor

        Args:
            sequence_length: Number of days to look back
            prediction_days: Number of days to predict forward
            features: List of features to use (e.g., ['Open', 'High', 'Low', 'Close', 'Volume'])
        """
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.features = features
        self.n_features = len(features)
        self.scaler = None
        self.model = None

    def fetch_data(self, ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
        """
        Fetch stock data from Yahoo Finance
        """
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            raise ValueError("No data fetched. Check ticker symbol and dates.")

        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        print(f"Fetched {len(data)} trading days")
        return data

    def prepare_data(self, data, train_split=0.8):
        """
        Prepare data with PROPER train/test split and NO look-ahead bias

        Key improvements:
        - Scaler fitted on training data ONLY
        - Proper sequence creation
        - Test data properly prepared
        """
        # Extract features
        feature_data = data[self.features].values

        # Split BEFORE scaling (critical!)
        split_idx = int(len(feature_data) * train_split)
        train_data = feature_data[:split_idx]
        test_data = feature_data[split_idx:]

        # Fit scaler on training data ONLY
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)

        # Create sequences for training
        X_train, y_train = self._create_sequences(train_scaled)

        # Create sequences for testing
        # For test, we need to include some training data for the first sequences
        combined_data = np.concatenate([train_scaled[-self.sequence_length:], test_scaled], axis=0)
        X_test, y_test = self._create_sequences(combined_data)

        print(f"Training sequences: {X_train.shape}")
        print(f"Testing sequences: {X_test.shape}")
        print(f"Features used: {self.features}")

        return X_train, y_train, X_test, y_test, data.index[split_idx + self.sequence_length:]

    def _create_sequences(self, data):
        """
        Create sequences for LSTM training

        Args:
            data: Scaled data array

        Returns:
            X: Input sequences [samples, sequence_length, n_features]
            y: Target values [samples, n_features] (for next day)
        """
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i, :])  # Predict all features for next day

        return np.array(X), np.array(y)

    def build_model(self, lstm_units=[64, 32], dropout_rate=0.2):
        """
        Build LSTM model with WORKING attention mechanism using Functional API

        This fixes the broken Sequential model from the original article
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))

        # First LSTM layer
        lstm_out1 = LSTM(lstm_units[0], return_sequences=True, name='lstm_1')(inputs)
        lstm_out1 = Dropout(dropout_rate)(lstm_out1)
        lstm_out1 = BatchNormalization()(lstm_out1)

        # Second LSTM layer
        lstm_out2 = LSTM(lstm_units[1], return_sequences=True, name='lstm_2')(lstm_out1)
        lstm_out2 = Dropout(dropout_rate)(lstm_out2)
        lstm_out2 = BatchNormalization()(lstm_out2)

        # Attention mechanism (PROPERLY IMPLEMENTED)
        # Query and value are the same (self-attention)
        attention_out = AdditiveAttention(name='attention')([lstm_out2, lstm_out2])

        # Combine attention output with LSTM output
        concat = Concatenate()([lstm_out2, attention_out])

        # Global pooling to get fixed-size output
        pooled = Lambda(lambda x: tf.reduce_mean(x, axis=1))(concat)

        # Dense layers
        dense1 = Dense(32, activation='relu')(pooled)
        dense1 = Dropout(dropout_rate)(dense1)

        # Output layer (predict all features)
        outputs = Dense(self.n_features, name='output')(dense1)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='LSTM_Attention')

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, verbose=1):
        """
        Train the model with proper callbacks
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance with multiple metrics
        """
        predictions = self.model.predict(X_test)

        # Inverse transform predictions and actual values
        predictions_original = self.scaler.inverse_transform(predictions)
        y_test_original = self.scaler.inverse_transform(y_test)

        # Calculate metrics for each feature
        metrics = {}
        for i, feature in enumerate(self.features):
            mae = mean_absolute_error(y_test_original[:, i], predictions_original[:, i])
            rmse = np.sqrt(mean_squared_error(y_test_original[:, i], predictions_original[:, i]))
            r2 = r2_score(y_test_original[:, i], predictions_original[:, i])

            metrics[feature] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }

            print(f"\n{feature}:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")

        return metrics, predictions_original, y_test_original

    def predict_next_n_days(self, data, n_days=4):
        """
        Predict next N days using the SAVED scaler (CRITICAL FIX)

        This fixes the scaler inconsistency issue
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        if self.scaler is None:
            raise ValueError("Scaler not fitted yet.")

        # Get last sequence_length days
        last_sequence = data[self.features].values[-self.sequence_length:]

        # Scale using the SAVED scaler (not a new one!)
        last_sequence_scaled = self.scaler.transform(last_sequence)

        # Predict iteratively
        predictions = []
        current_sequence = last_sequence_scaled.copy()

        for _ in range(n_days):
            # Reshape for prediction
            current_batch = current_sequence.reshape(1, self.sequence_length, self.n_features)

            # Predict next day
            next_pred = self.model.predict(current_batch, verbose=0)

            # Store prediction
            predictions.append(next_pred[0])

            # Update sequence (sliding window)
            current_sequence = np.vstack([current_sequence[1:], next_pred[0]])

        # Inverse transform predictions
        predictions = np.array(predictions)
        predictions_original = self.scaler.inverse_transform(predictions)

        return predictions_original

    def backtest_predictions(self, data, start_date, end_date, step_days=5):
        """
        Backtest predictions on historical data with walk-forward validation

        This allows us to validate predictions on KNOWN historical data
        """
        results = []
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        while current_date < end_date:
            # Get data up to current_date
            historical_data = data[:current_date]

            if len(historical_data) < self.sequence_length + 1:
                current_date += timedelta(days=step_days)
                continue

            # Predict next 4 days
            try:
                predictions = self.predict_next_n_days(historical_data, n_days=self.prediction_days)

                # Get actual values for comparison
                future_dates = pd.date_range(
                    current_date + timedelta(days=1),
                    periods=self.prediction_days
                )

                actuals = []
                for future_date in future_dates:
                    if future_date in data.index:
                        actuals.append(data.loc[future_date, self.features].values)
                    else:
                        actuals.append(None)

                results.append({
                    'prediction_date': current_date,
                    'predictions': predictions,
                    'actuals': actuals,
                    'future_dates': future_dates
                })

            except Exception as e:
                print(f"Error at {current_date}: {e}")

            current_date += timedelta(days=step_days)

        return results

    def plot_predictions(self, data, test_dates, predictions, actuals,
                        feature_idx=0, save_path=None):
        """
        Visualize predictions vs actual values
        """
        plt.figure(figsize=(15, 6))

        # Plot historical data
        plt.plot(data.index[-200:], data[self.features[feature_idx]].values[-200:],
                label='Historical', color='blue', alpha=0.7)

        # Plot test predictions
        plt.plot(test_dates, actuals[:, feature_idx],
                label='Actual (Test)', color='green', marker='o', markersize=3)

        plt.plot(test_dates, predictions[:, feature_idx],
                label='Predicted (Test)', color='red', marker='x', markersize=3)

        plt.title(f'{self.features[feature_idx]} Price: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(history.history['mae'], label='Train MAE')
        if 'val_mae' in history.history:
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_model(self, model_path='lstm_attention_model.h5',
                   scaler_path='scaler.pkl'):
        """
        Save model and scaler (CRITICAL for production use)
        """
        if self.model is None:
            raise ValueError("No model to save.")

        self.model.save(model_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path='lstm_attention_model.h5',
                   scaler_path='scaler.pkl'):
        """
        Load saved model and scaler
        """
        self.model = keras.models.load_model(model_path)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

def main():
    """
    Main execution function demonstrating proper usage
    """
    print("="*80)
    print("LSTM with Attention - Stock Price Prediction (CORRECTED VERSION)")
    print("="*80)

    # Configuration
    TICKER = 'AAPL'
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    SEQUENCE_LENGTH = 60
    PREDICTION_DAYS = 4

    # Use multiple features for better predictions
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Or use just Close for comparison with original
    # FEATURES = ['Close']

    # Initialize predictor
    predictor = StockPredictorLSTMAttention(
        sequence_length=SEQUENCE_LENGTH,
        prediction_days=PREDICTION_DAYS,
        features=FEATURES
    )

    # Fetch data
    data = predictor.fetch_data(TICKER, START_DATE, END_DATE)

    # Prepare data with PROPER splitting
    X_train, y_train, X_test, y_test, test_dates = predictor.prepare_data(
        data, train_split=0.8
    )

    # Split training data for validation
    val_split = 0.2
    val_idx = int(len(X_train) * (1 - val_split))
    X_train_final = X_train[:val_idx]
    y_train_final = y_train[:val_idx]
    X_val = X_train[val_idx:]
    y_val = y_train[val_idx:]

    print(f"\nFinal split:")
    print(f"  Training: {X_train_final.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Build model
    print("\nBuilding model...")
    model = predictor.build_model(lstm_units=[64, 32], dropout_rate=0.2)
    print(model.summary())

    # Train model
    print("\nTraining model...")
    history = predictor.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=100,
        batch_size=32,
        verbose=1
    )

    # Plot training history
    predictor.plot_training_history(history)

    # Evaluate on test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    metrics, predictions, actuals = predictor.evaluate(X_test, y_test)

    # Plot predictions
    close_idx = FEATURES.index('Close') if 'Close' in FEATURES else 0
    predictor.plot_predictions(
        data, test_dates, predictions, actuals,
        feature_idx=close_idx,
        save_path='predictions_vs_actual.png'
    )

    # Predict next 4 days from TODAY
    print("\n" + "="*80)
    print("FUTURE PREDICTION (Next 4 Trading Days)")
    print("="*80)
    future_predictions = predictor.predict_next_n_days(data, n_days=4)

    last_date = data.index[-1]
    for i, pred in enumerate(future_predictions, 1):
        print(f"\nDay {i} prediction:")
        for j, feature in enumerate(FEATURES):
            print(f"  {feature}: ${pred[j]:.2f}")

    # Save model and scaler
    predictor.save_model('lstm_attention_model.h5', 'scaler.pkl')

    # Demonstrate backtesting (optional, for validation)
    print("\n" + "="*80)
    print("BACKTESTING (Walk-Forward Validation)")
    print("="*80)
    print("Performing walk-forward validation on historical data...")

    backtest_start = pd.to_datetime('2023-06-01')
    backtest_end = pd.to_datetime('2023-12-01')

    backtest_results = predictor.backtest_predictions(
        data, backtest_start, backtest_end, step_days=10
    )

    print(f"Completed {len(backtest_results)} backtest iterations")

    # Calculate backtest accuracy
    backtest_errors = []
    for result in backtest_results:
        preds = result['predictions']
        actuals = result['actuals']

        for i, actual in enumerate(actuals):
            if actual is not None:
                error = np.abs(preds[i] - actual)
                backtest_errors.append(error)

    if backtest_errors:
        backtest_errors = np.array(backtest_errors)
        print(f"\nBacktest MAE per feature:")
        for i, feature in enumerate(FEATURES):
            print(f"  {feature}: {np.mean(backtest_errors[:, i]):.4f}")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
