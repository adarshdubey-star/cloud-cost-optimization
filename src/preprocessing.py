"""
Data preprocessing and Exploratory Data Analysis (EDA) module.

Handles loading, cleaning, feature engineering, and statistical summaries
for the cloud workload dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(path: str = "data/cloud_workload.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["week_number"] = df["timestamp"].dt.isocalendar().week.astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "cpu_utilization", lags: list = None) -> pd.DataFrame:
    df = df.copy()
    if lags is None:
        lags = [1, 3, 6, 12, 288]  # 5min, 15min, 30min, 1hr, 1day (at 5-min intervals)
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    df[f"{target}_rolling_12"] = df[target].rolling(12).mean()
    df[f"{target}_rolling_288"] = df[target].rolling(288).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def generate_eda_plots(df: pd.DataFrame, output_dir: str = "results/figures"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Time-series overview
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    sample = df.iloc[:288 * 7]  # first week
    axes[0].plot(sample["timestamp"], sample["cpu_utilization"], linewidth=0.7, color="#2196F3")
    axes[0].set_ylabel("CPU (%)")
    axes[0].set_title("Cloud Workload — First Week")
    axes[1].plot(sample["timestamp"], sample["memory_utilization"], linewidth=0.7, color="#4CAF50")
    axes[1].set_ylabel("Memory (%)")
    axes[2].plot(sample["timestamp"], sample["network_mbps"], linewidth=0.7, color="#FF9800")
    axes[2].set_ylabel("Network (Mbps)")
    axes[2].set_xlabel("Time")
    plt.tight_layout()
    fig.savefig(out / "timeseries_overview.png", dpi=150)
    plt.close(fig)

    # 2. Correlation heatmap
    numeric = df[["cpu_utilization", "memory_utilization", "disk_io_mbps",
                   "network_mbps", "request_count"]].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Hourly CPU distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly = df.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    sns.boxplot(data=hourly, x="hour", y="cpu_utilization", ax=ax,
                palette="Blues", fliersize=1)
    ax.set_title("CPU Utilization by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("CPU (%)")
    plt.tight_layout()
    fig.savefig(out / "cpu_hourly_boxplot.png", dpi=150)
    plt.close(fig)

    # 4. Cost over time (daily aggregation)
    daily = df.set_index("timestamp").resample("D").agg({
        "cost_threshold": "sum",
        "cpu_utilization": "mean",
    }).reset_index()
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(daily["timestamp"], daily["cost_threshold"], color="#EF5350", alpha=0.7, label="Daily Cost ($)")
    ax1.set_ylabel("Cost ($)", color="#EF5350")
    ax2 = ax1.twinx()
    ax2.plot(daily["timestamp"], daily["cpu_utilization"], color="#1565C0", linewidth=1.5, label="Avg CPU %")
    ax2.set_ylabel("Avg CPU (%)", color="#1565C0")
    ax1.set_title("Daily Cloud Cost vs Average CPU Utilization")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    plt.tight_layout()
    fig.savefig(out / "daily_cost_vs_cpu.png", dpi=150)
    plt.close(fig)

    print(f"[EDA] Saved plots to {out}/")


if __name__ == "__main__":
    df = load_data()
    generate_eda_plots(df)
