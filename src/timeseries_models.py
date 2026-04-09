"""
Time-series forecasting models for cloud workload prediction.

Models: ARIMA (via statsmodels) and LSTM (via TensorFlow/Keras).
Operates on hourly-aggregated CPU utilization for tractability.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple

warnings.filterwarnings("ignore")


def _hourly_cpu(path: str = "data/cloud_workload.csv") -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    hourly = df.set_index("timestamp")["cpu_utilization"].resample("h").mean()
    return hourly


def evaluate_ts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


# ===================== ARIMA =====================

def train_arima(series: pd.Series, train_ratio: float = 0.8) -> Tuple[dict, np.ndarray, np.ndarray]:
    from statsmodels.tsa.arima.model import ARIMA

    split = int(len(series) * train_ratio)
    train, test = series.iloc[:split], series.iloc[split:]

    print("[TimeSeries] Fitting ARIMA(5,1,2)...")
    model = ARIMA(train, order=(5, 1, 2))
    fitted = model.fit(method_kwargs={"maxiter": 500})

    forecast = fitted.forecast(steps=len(test))
    forecast = np.clip(forecast.values, 1, 100)
    metrics = evaluate_ts(test.values, forecast)
    print(f"  -> ARIMA: {metrics}")
    return metrics, test.values, forecast


# ===================== LSTM =====================

def train_lstm(series: pd.Series, train_ratio: float = 0.8,
               lookback: int = 48, epochs: int = 30, batch_size: int = 64
               ) -> Tuple[dict, np.ndarray, np.ndarray]:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(42)

    values = series.values.reshape(-1, 1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    def create_sequences(data, lb):
        X, y = [], []
        for i in range(lb, len(data)):
            X.append(data[i - lb:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, lookback)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    print("[TimeSeries] Training LSTM...")
    model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size, verbose=0,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = np.clip(y_pred, 1, 100)

    metrics = evaluate_ts(y_actual, y_pred)
    print(f"  -> LSTM: {metrics}")
    return metrics, y_actual, y_pred


def run_timeseries_models() -> Dict[str, dict]:
    series = _hourly_cpu()
    arima_metrics, arima_true, arima_pred = train_arima(series)
    lstm_metrics, lstm_true, lstm_pred = train_lstm(series)
    return {
        "ARIMA": arima_metrics,
        "LSTM": lstm_metrics,
    }, {
        "ARIMA": (arima_true, arima_pred),
        "LSTM": (lstm_true, lstm_pred),
    }


if __name__ == "__main__":
    results, _ = run_timeseries_models()
    print("\n=== Time-Series Results ===")
    print(pd.DataFrame(results).T)
