"""
Advanced LSTM with Attention and Technical Indicators
=====================================================

This enhanced version adds:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Multi-feature training
- Directional accuracy metrics
- Feature importance analysis
- Better prediction accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from keras.layers import AdditiveAttention, Concatenate, Lambda
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)


def add_technical_indicators(data):
    """
    Add comprehensive technical indicators to stock data

    Indicators added:
    - SMA (Simple Moving Averages): 5, 10, 20, 50 days
    - EMA (Exponential Moving Averages): 12, 26 days
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - ATR (Average True Range)
    - OBV (On-Balance Volume)
    - Rate of Change (ROC)
    """
    df = data.copy()

    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Simple Moving Averages
    for period in [5, 10, 20, 50]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'Close_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}'] - 1

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Rate of Change
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # Drop NaN values
    df = df.dropna()

    return df


class AdvancedStockPredictor:
    """
    Advanced Stock Predictor with Technical Indicators
    """

    def __init__(self, sequence_length=60, prediction_days=1):
        """
        Initialize with default technical indicators

        Args:
            sequence_length: Number of days to look back
            prediction_days: Number of days to predict (default=1 for better accuracy)
        """
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = None
        self.model = None
        self.feature_columns = None

    def fetch_and_prepare_data(self, ticker='AAPL', start_date='2018-01-01', end_date='2024-01-01'):
        """Fetch data and add technical indicators"""
        print(f"Fetching {ticker} data...")
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data.fillna(method='ffill').fillna(method='bfill')

        print("Adding technical indicators...")
        data = add_technical_indicators(data)

        # Select features (all numerical columns except target)
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'Log_Returns',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'Close_to_SMA_5', 'Close_to_SMA_10', 'Close_to_SMA_20', 'Close_to_SMA_50',
            'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'ATR', 'OBV', 'ROC',
            'Volume_SMA_20', 'Volume_Ratio',
            'Momentum',
            'Stoch_K', 'Stoch_D'
        ]

        print(f"Total features: {len(self.feature_columns)}")
        print(f"Total samples: {len(data)}")

        return data

    def prepare_data(self, data, train_split=0.8, predict_direction=False):
        """
        Prepare data for training

        Args:
            predict_direction: If True, also prepare direction labels
        """
        feature_data = data[self.feature_columns].values

        # Split BEFORE scaling
        split_idx = int(len(feature_data) * train_split)
        train_data = feature_data[:split_idx]
        test_data = feature_data[split_idx:]

        # Fit scaler on training data ONLY
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)

        # Create sequences
        X_train, y_train = self._create_sequences(train_scaled, data[:split_idx])

        # For test
        combined_data = np.concatenate([train_scaled[-self.sequence_length:], test_scaled])
        X_test, y_test = self._create_sequences(combined_data, data[split_idx:])

        # Also get direction labels if requested
        direction_train = direction_test = None
        if predict_direction:
            direction_train = self._get_direction_labels(data[:split_idx], len(y_train))
            direction_test = self._get_direction_labels(data[split_idx:], len(y_test))

        test_dates = data.index[split_idx + self.sequence_length:]

        print(f"\nData split:")
        print(f"  Training sequences: {X_train.shape}")
        print(f"  Testing sequences: {X_test.shape}")
        print(f"  Features per sequence: {X_train.shape[2]}")

        return X_train, y_train, X_test, y_test, test_dates, direction_train, direction_test

    def _create_sequences(self, data, original_data):
        """Create sequences for LSTM training"""
        X, y = [], []

        # Get Close price index
        close_idx = self.feature_columns.index('Close')

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])

            # Target: next day's close price
            y.append(data[i, close_idx])

        return np.array(X), np.array(y)

    def _get_direction_labels(self, data, n_samples):
        """Get direction labels (1 for up, 0 for down)"""
        close_prices = data['Close'].values
        directions = []

        start_idx = self.sequence_length
        for i in range(start_idx, start_idx + n_samples):
            if i < len(close_prices) - 1:
                direction = 1 if close_prices[i + 1] > close_prices[i] else 0
                directions.append(direction)
            else:
                directions.append(directions[-1] if directions else 0)

        return np.array(directions)

    def build_model(self, lstm_units=[128, 64, 32], dropout_rate=0.3):
        """Build advanced LSTM model"""
        n_features = len(self.feature_columns)

        # Input layer
        inputs = Input(shape=(self.sequence_length, n_features))

        # First LSTM layer
        lstm_out1 = LSTM(lstm_units[0], return_sequences=True)(inputs)
        lstm_out1 = Dropout(dropout_rate)(lstm_out1)
        lstm_out1 = BatchNormalization()(lstm_out1)

        # Second LSTM layer
        lstm_out2 = LSTM(lstm_units[1], return_sequences=True)(lstm_out1)
        lstm_out2 = Dropout(dropout_rate)(lstm_out2)
        lstm_out2 = BatchNormalization()(lstm_out2)

        # Third LSTM layer
        lstm_out3 = LSTM(lstm_units[2], return_sequences=True)(lstm_out2)
        lstm_out3 = Dropout(dropout_rate)(lstm_out3)
        lstm_out3 = BatchNormalization()(lstm_out3)

        # Attention mechanism
        attention_out = AdditiveAttention()([lstm_out3, lstm_out3])

        # Combine
        concat = Concatenate()([lstm_out3, attention_out])

        # Global pooling
        pooled = Lambda(lambda x: tf.reduce_mean(x, axis=1))(concat)

        # Dense layers
        dense1 = Dense(64, activation='relu')(pooled)
        dense1 = Dropout(dropout_rate)(dense1)
        dense1 = BatchNormalization()(dense1)

        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(dropout_rate / 2)(dense2)

        # Output: single value (scaled close price)
        outputs = Dense(1, name='price_output')(dense2)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=150, batch_size=64, verbose=1):
        """Train the model"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def evaluate(self, X_test, y_test, data, test_dates):
        """Evaluate with price and direction metrics"""
        predictions = self.model.predict(X_test)

        # Get Close price index for inverse transform
        close_idx = self.feature_columns.index('Close')

        # Create dummy array for inverse transform
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        dummy[:, close_idx] = predictions.flatten()
        predictions_original = self.scaler.inverse_transform(dummy)[:, close_idx]

        dummy_y = np.zeros((len(y_test), len(self.feature_columns)))
        dummy_y[:, close_idx] = y_test
        y_test_original = self.scaler.inverse_transform(dummy_y)[:, close_idx]

        # Price metrics
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
        r2 = r2_score(y_test_original, predictions_original)
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100

        print("\n" + "="*60)
        print("PRICE PREDICTION METRICS")
        print("="*60)
        print(f"MAE:   ${mae:.2f}")
        print(f"RMSE:  ${rmse:.2f}")
        print(f"R²:    {r2:.4f}")
        print(f"MAPE:  {mape:.2f}%")

        # Direction metrics
        actual_directions = []
        predicted_directions = []

        for i in range(len(y_test_original) - 1):
            actual_dir = 1 if y_test_original[i+1] > y_test_original[i] else 0
            pred_dir = 1 if predictions_original[i+1] > predictions_original[i] else 0
            actual_directions.append(actual_dir)
            predicted_directions.append(pred_dir)

        direction_accuracy = accuracy_score(actual_directions, predicted_directions) * 100

        print("\n" + "="*60)
        print("DIRECTIONAL PREDICTION METRICS")
        print("="*60)
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        print(f"Correct directions: {sum(np.array(actual_directions) == np.array(predicted_directions))}/{len(actual_directions)}")

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'predictions': predictions_original,
            'actuals': y_test_original
        }

    def save_model(self, model_path='advanced_lstm.h5', scaler_path='advanced_scaler.pkl',
                   features_path='features.pkl'):
        """Save model, scaler, and feature list"""
        self.model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"Model, scaler, and features saved!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("ADVANCED LSTM with Technical Indicators - Stock Prediction")
    print("="*80)

    # Initialize
    predictor = AdvancedStockPredictor(
        sequence_length=60,
        prediction_days=1
    )

    # Fetch and prepare data
    data = predictor.fetch_and_prepare_data(
        ticker='AAPL',
        start_date='2018-01-01',
        end_date='2024-01-01'
    )

    # Prepare data
    X_train, y_train, X_test, y_test, test_dates, dir_train, dir_test = predictor.prepare_data(
        data,
        train_split=0.8,
        predict_direction=True
    )

    # Split for validation
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
    print("\nBuilding advanced model...")
    model = predictor.build_model(lstm_units=[128, 64, 32], dropout_rate=0.3)
    print(f"Total parameters: {model.count_params():,}")

    # Train
    print("\nTraining model...")
    history = predictor.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=150,
        batch_size=64
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = predictor.evaluate(X_test, y_test, data, test_dates)

    # Save
    predictor.save_model(
        'advanced_lstm_model.h5',
        'advanced_scaler.pkl',
        'advanced_features.pkl'
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

    return predictor, results, history


if __name__ == "__main__":
    predictor, results, history = main()
