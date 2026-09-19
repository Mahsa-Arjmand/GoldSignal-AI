import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Attention, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer: 
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.history = None
        self.training_metrics = {}
        self.feature_columns = []
        
        os.makedirs(model_dir, exist_ok=True)
        
    def create_lstm_model(self, input_shape, layers=[100, 100, 50], dropout=0.2, learning_rate=0.001):
     
        model = Sequential()
        
        if len(layers) > 1:
            model.add(LSTM(layers[0], return_sequences=True, input_shape=input_shape))
        else:
            model.add(LSTM(layers[0], input_shape=input_shape))
        
        model.add(Dropout(dropout))
        
        for i, units in enumerate(layers[1:-1]):
            return_seq = (i < len(layers) - 2)
            model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(dropout))
        
        if len(layers) > 1:
            model.add(LSTM(layers[-1], return_sequences=False))
            model.add(Dropout(dropout))
        
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        
        self.model = model
        return model
    
    def create_bidirectional_lstm(self, input_shape, units=[128, 64], dropout=0.2):
    
        model = Sequential()
        
        model.add(Bidirectional(LSTM(units[0], return_sequences=True), input_shape=input_shape))
        model.add(Dropout(dropout))
        
        if len(units) > 1:
            model.add(Bidirectional(LSTM(units[1], return_sequences=False)))
            model.add(Dropout(dropout))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def cross_validate(self, X, y, n_splits=5, epochs=20, batch_size=32, sequence_length=60):
   
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {
            'rmse': [],
            'mae': [],
            'mape': []
        }
        
        print(f" Performing {n_splits}-fold TimeSeries Cross-Validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            input_shape = (X.shape[1], X.shape[2])
            fold_model = self.create_lstm_model(input_shape)
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = fold_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            predictions = fold_model.predict(X_val, verbose=0).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            mae = mean_absolute_error(y_val, predictions)
            mape = mean_absolute_percentage_error(y_val, predictions) * 100
            
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['mape'].append(mape)
            
            print(f"  Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        cv_summary = {}
        for metric, scores in cv_scores.items():
            cv_summary[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        cv_summary['overall_mean_rmse'] = cv_summary['rmse']['mean']
        cv_summary['overall_std_rmse'] = cv_summary['rmse']['std']
        
        print(f"\n CV Summary - Mean RMSE: {cv_summary['rmse']['mean']:.4f} (+/- {cv_summary['rmse']['std']:.4f})")
        
        return cv_summary
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, patience=15):

        if self.model is None:
            raise ValueError("Model not created. Call create_lstm_model() first.")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        print(f" Training model...")
        print(f"   - Input shape: {X_train.shape}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2
        )
        
        self.history = history
        return history
    
    def evaluate(self, X_test, y_test, scaler=None):

        if self.model is None:
            raise ValueError("No trained model found.")
        
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
        if scaler is not None:
            n_features = scaler.scale_.shape[0]
            dummy_pred = np.zeros((len(predictions), n_features))
            dummy_pred[:, 0] = predictions
            predictions = scaler.inverse_transform(dummy_pred)[:, 0]
            
            dummy_true = np.zeros((len(y_test), n_features))
            dummy_true[:, 0] = y_test
            y_test = scaler.inverse_transform(dummy_true)[:, 0]
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'mape': mean_absolute_percentage_error(y_test, predictions) * 100,
            'r2': r2_score(y_test, predictions)
        }
        
        if len(predictions) > 1:
            actual_direction = np.diff(y_test) > 0
            pred_direction = np.diff(predictions) > 0
            metrics['direction_accuracy'] = np.mean(actual_direction == pred_direction) * 100
        
        self.training_metrics = metrics
        
        print("\n Model Evaluation:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return metrics, predictions, y_test
    
    def predict_future(self, last_sequence, days=7, scaler=None, sentiment_factor=0):

        if self.model is None:
            raise ValueError("No trained model found.")
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for day in range(days):
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            
            sentiment_weight = sentiment_factor * (1 - day * 0.1)
            next_pred_adjusted = next_pred * (1 + sentiment_weight)
            
            future_predictions.append(next_pred_adjusted)
            
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        if scaler is not None:
            n_features = scaler.scale_.shape[0]
            dummy = np.zeros((len(future_predictions), n_features))
            dummy[:, 0] = future_predictions
            future_predictions = scaler.inverse_transform(dummy)[:, 0]
        
        return np.array(future_predictions)
    
    def save_model(self, name='lstm_model'):
  
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_path = os.path.join(self.model_dir, f'{name}.h5')
        self.model.save(model_path)
        
        if self.scaler is not None:
            scaler_path = os.path.join(self.model_dir, f'{name}_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        
        metadata = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'model_type': 'LSTM',
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in self.training_metrics.items()},
            'feature_columns': self.feature_columns
        }
        
        metadata_path = os.path.join(self.model_dir, f'{name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Model saved to {model_path}")
    
    def load_model(self, name='lstm_model'):
     
        model_path = os.path.join(self.model_dir, f'{name}.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = load_model(model_path)
        print(f" Model loaded from {model_path}")
        
        scaler_path = os.path.join(self.model_dir, f'{name}_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        metadata_path = os.path.join(self.model_dir, f'{name}_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.training_metrics = metadata.get('metrics', {})
                self.feature_columns = metadata.get('feature_columns', [])