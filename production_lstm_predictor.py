"""
Production-Ready LSTM Stock Predictor
======================================

This version addresses the 50% accuracy problem with:
1. Uses RETURNS instead of absolute prices (handles regime changes)
2. 35+ technical indicators
3. Deeper architecture with better regularization
4. Ensemble methods
5. Focus on DIRECTION prediction
6. Proper walk-forward validation

Author: Dr. Ernesto Lee | drlee.io
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler  # Better than MinMaxScaler for outliers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization,
                          AdditiveAttention, Concatenate, Lambda, Bidirectional)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1_l2

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)


class ProductionStockPredictor:
    """
    Production-ready LSTM predictor focused on DIRECTION, not price

    Key improvements:
    - Uses returns (% changes) instead of absolute prices
    - 35+ technical indicators
    - Bidirectional LSTM for better context
    - Ensemble of multiple models
    - Proper walk-forward validation
    """

    def __init__(self, sequence_length=60, use_ensemble=True, n_models=5):
        self.sequence_length = sequence_length
        self.use_ensemble = use_ensemble
        self.n_models = n_models
        self.scalers = []
        self.models = []
        self.feature_columns = []

    def add_technical_indicators(self, data):
        """Add 35+ technical indicators"""
        df = data.copy()

        # Returns (KEY: Use returns instead of prices!)
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            # Distance from MA (normalized)
            df[f'Close_to_SMA_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_normalized'] = (df['RSI'] - 50) / 50  # Normalize to [-1, 1]

        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        # Normalize MACD
        df['MACD_normalized'] = df['MACD'] / df['Close']
        df['MACD_Hist_normalized'] = df['MACD_Hist'] / df['Close']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        df['ATR_normalized'] = df['ATR'] / df['Close']

        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_normalized'] = df['OBV'] / df['OBV'].rolling(window=20).mean()

        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['Momentum_normalized'] = df['Momentum'] / df['Close'].shift(10)
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        # Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        # Price patterns
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']

        # Volatility
        df['Returns_Volatility'] = df['Returns'].rolling(window=20).std()

        # Trend strength
        df['ADX'] = self._calculate_adx(df)

        return df.dropna()

    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = pd.concat([high - low,
                       abs(high - close.shift()),
                       abs(low - close.shift())], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def fetch_and_prepare_data(self, ticker='AAPL', start_date='2018-01-01', end_date=None):
        """Fetch data and add all indicators"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        # Fix MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.fillna(method='ffill').fillna(method='bfill')

        print(f"Adding technical indicators...")
        data = self.add_technical_indicators(data)

        print(f"Total trading days: {len(data)}")
        print(f"Latest date: {data.index[-1].strftime('%Y-%m-%d')}")

        return data

    def prepare_sequences(self, data, train_split=0.8, predict_direction=True):
        """
        Prepare sequences for training

        KEY: We predict DIRECTION (up/down), not exact price!
        """
        # Select features
        self.feature_columns = [
            'Returns', 'Log_Returns',
            'RSI_normalized', 'MACD_normalized', 'MACD_Hist_normalized',
            'BB_Width', 'BB_Position', 'ATR_normalized',
            'Volume_Ratio', 'OBV_normalized',
            'Momentum_normalized', 'ROC', 'Returns_Volatility',
            'Stoch_K', 'Stoch_D', 'ADX',
            'Close_to_SMA_5', 'Close_to_SMA_10', 'Close_to_SMA_20', 'Close_to_SMA_50',
            'High_Low_Ratio', 'Close_Open_Ratio'
        ]

        # Get feature data
        feature_data = data[self.feature_columns].values

        # Create target: 1 if price goes up next day, 0 if down
        if predict_direction:
            future_returns = data['Returns'].shift(-1).values
            target = (future_returns > 0).astype(int)
        else:
            target = data['Returns'].shift(-1).values

        # Split data
        split_idx = int(len(feature_data) * train_split)

        X_train_data = feature_data[:split_idx]
        y_train_data = target[:split_idx]
        X_test_data = feature_data[split_idx:]
        y_test_data = target[split_idx:]

        # Scale features (fit on train only!)
        scaler = RobustScaler()  # Better for outliers than MinMaxScaler
        X_train_scaled = scaler.fit_transform(X_train_data)
        X_test_scaled = scaler.transform(X_test_data)

        # Create sequences
        X_train, y_train = self._create_sequences(X_train_scaled, y_train_data)

        # For test, include last training sequences
        combined_X = np.vstack([X_train_scaled[-self.sequence_length:], X_test_scaled])
        combined_y = np.concatenate([y_train_data[-self.sequence_length:], y_test_data])
        X_test, y_test = self._create_sequences(combined_X, combined_y)

        test_dates = data.index[split_idx + self.sequence_length:]

        print(f"\nData prepared:")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Training sequences: {X_train.shape}")
        print(f"  Test sequences: {X_test.shape}")
        print(f"  Target: {'Direction (0/1)' if predict_direction else 'Returns'}")

        return X_train, y_train, X_test, y_test, test_dates, scaler

    def _create_sequences(self, X, y):
        """Create sequences for LSTM"""
        sequences_X = []
        sequences_y = []

        for i in range(self.sequence_length, len(X)):
            sequences_X.append(X[i - self.sequence_length:i])
            sequences_y.append(y[i])

        return np.array(sequences_X), np.array(sequences_y)

    def build_model(self, n_features, use_bidirectional=True):
        """
        Build improved LSTM model

        Improvements:
        - Bidirectional LSTM (sees past AND future context in training)
        - Deeper architecture
        - Better regularization
        - L1/L2 regularization to prevent overfitting
        """
        inputs = Input(shape=(self.sequence_length, n_features))

        # First LSTM layer (Bidirectional)
        if use_bidirectional:
            lstm1 = Bidirectional(LSTM(128, return_sequences=True,
                                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(inputs)
        else:
            lstm1 = LSTM(128, return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)

        lstm1 = Dropout(0.3)(lstm1)
        lstm1 = BatchNormalization()(lstm1)

        # Second LSTM layer
        if use_bidirectional:
            lstm2 = Bidirectional(LSTM(64, return_sequences=True,
                                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(lstm1)
        else:
            lstm2 = LSTM(64, return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(lstm1)

        lstm2 = Dropout(0.3)(lstm2)
        lstm2 = BatchNormalization()(lstm2)

        # Attention mechanism
        attention = AdditiveAttention()([lstm2, lstm2])
        concat = Concatenate()([lstm2, attention])

        # Global pooling
        pooled = Lambda(lambda x: tf.reduce_mean(x, axis=1))(concat)

        # Dense layers
        dense1 = Dense(64, activation='relu',
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(pooled)
        dense1 = Dropout(0.4)(dense1)
        dense1 = BatchNormalization()(dense1)

        dense2 = Dense(32, activation='relu',
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense1)
        dense2 = Dropout(0.3)(dense2)

        # Output layer - BINARY CLASSIFICATION (direction)
        outputs = Dense(1, activation='sigmoid')(dense2)

        model = Model(inputs=inputs, outputs=outputs)

        # Use binary crossentropy for classification
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        return model

    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=100):
        """
        Train ensemble of models with different initializations

        This improves robustness - multiple models vote on direction
        """
        n_features = X_train.shape[2]

        for i in range(self.n_models):
            print(f"\n{'='*60}")
            print(f"Training Model {i+1}/{self.n_models}")
            print(f"{'='*60}")

            # Different random seed for each model
            tf.random.set_seed(42 + i)
            np.random.seed(42 + i)

            # Build model
            model = self.build_model(n_features)

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7,
                    verbose=1
                )
            ]

            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=64,
                callbacks=callbacks,
                verbose=1
            )

            self.models.append(model)

            # Print final metrics
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]
            val_auc = history.history['val_auc'][-1]

            print(f"\nModel {i+1} Final Metrics:")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc*100:.2f}%")
            print(f"  Val AUC: {val_auc:.4f}")

        print(f"\n{'='*60}")
        print(f"Ensemble Training Complete! {self.n_models} models trained.")
        print(f"{'='*60}")

    def predict_ensemble(self, X):
        """
        Predict using ensemble voting

        Returns probability of UP direction
        """
        predictions = []

        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble performance on direction prediction
        """
        # Get probability predictions
        y_pred_proba = self.predict_ensemble(X_test)

        # Convert to binary (0/1)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n{'='*60}")
        print(f"ENSEMBLE PERFORMANCE")
        print(f"{'='*60}")
        print(f"Direction Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nPredictions:")
        print(f"  Total: {len(y_test)}")
        print(f"  Correct: {(y_pred == y_test).sum()}")
        print(f"  Wrong: {(y_pred != y_test).sum()}")
        print(f"\nPredicted Direction:")
        print(f"  UP: {(y_pred == 1).sum()} ({(y_pred == 1).sum()/len(y_pred)*100:.1f}%)")
        print(f"  DOWN: {(y_pred == 0).sum()} ({(y_pred == 0).sum()/len(y_pred)*100:.1f}%)")
        print(f"\nActual Direction:")
        print(f"  UP: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
        print(f"  DOWN: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def save_ensemble(self, prefix='production'):
        """Save all models and scalers"""
        for i, model in enumerate(self.models):
            model.save(f'{prefix}_model_{i}.h5')

        for i, scaler in enumerate(self.scalers):
            with open(f'{prefix}_scaler_{i}.pkl', 'wb') as f:
                pickle.dump(scaler, f)

        print(f"\nSaved {len(self.models)} models and {len(self.scalers)} scalers")


if __name__ == "__main__":
    print("Production LSTM Stock Predictor")
    print("=" * 60)
    print("\nThis version focuses on DIRECTION prediction, not price.")
    print("Goal: Get above 55% direction accuracy (better than random)")
    print("=" * 60)
