"""
Supervised learning models for cloud resource demand forecasting.

Models: Linear Regression, Random Forest, Gradient Boosting (XGBoost-style via
scikit-learn's HistGradientBoostingRegressor for speed).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple

from src.preprocessing import load_data, add_temporal_features, add_lag_features

FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "minute_of_day",
    "memory_utilization", "disk_io_mbps", "network_mbps", "request_count",
    "cpu_utilization_lag_1", "cpu_utilization_lag_3", "cpu_utilization_lag_6",
    "cpu_utilization_lag_12", "cpu_utilization_lag_288",
    "cpu_utilization_rolling_12", "cpu_utilization_rolling_288",
]
TARGET = "cpu_utilization"


def prepare_dataset() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_data()
    df = add_temporal_features(df)
    df = add_lag_features(df, target=TARGET)
    X = df[FEATURE_COLS].values
    y = df[TARGET].values
    return df, X, y


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def train_and_evaluate() -> Dict[str, dict]:
    df, X, y = prepare_dataset()

    # Temporal 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, max_depth=8, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        print(f"[Supervised] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        results[name] = metrics
        predictions[name] = y_pred
        print(f"  -> {metrics}")

    return results, predictions, y_test, df.iloc[split_idx:].reset_index(drop=True)


if __name__ == "__main__":
    results, *_ = train_and_evaluate()
    print("\n=== Supervised Model Results ===")
    print(pd.DataFrame(results).T)
