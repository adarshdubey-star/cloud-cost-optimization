"""
Synthetic cloud workload data generator.

Produces realistic time-series data mimicking production cloud environments with
diurnal patterns, weekly seasonality, random spikes, and gradual trend drift.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_cloud_workload_data(
    days: int = 90,
    interval_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    n_points = days * 24 * 60 // interval_minutes
    timestamps = pd.date_range("2025-01-01", periods=n_points, freq=f"{interval_minutes}min")

    hours = timestamps.hour + timestamps.minute / 60.0
    day_of_week = timestamps.dayofweek

    # ----- CPU utilization (%) -----
    cpu_diurnal = 30 + 25 * np.sin(2 * np.pi * (hours - 6) / 24)
    cpu_weekly = 5 * np.where(day_of_week < 5, 1, -1).astype(float)
    cpu_trend = np.linspace(0, 8, n_points)
    cpu_noise = np.random.normal(0, 4, n_points)
    cpu_spikes = np.random.choice([0, 30], size=n_points, p=[0.98, 0.02])
    cpu = np.clip(cpu_diurnal + cpu_weekly + cpu_trend + cpu_noise + cpu_spikes, 1, 100)

    # ----- Memory utilization (%) -----
    mem_base = 40 + 15 * np.sin(2 * np.pi * (hours - 8) / 24)
    mem_noise = np.random.normal(0, 3, n_points)
    mem_trend = np.linspace(0, 6, n_points)
    memory = np.clip(mem_base + mem_noise + mem_trend, 5, 100)

    # ----- Disk I/O (MB/s) -----
    disk_base = 50 + 20 * np.sin(2 * np.pi * (hours - 10) / 24)
    disk_noise = np.random.normal(0, 8, n_points)
    disk_io = np.clip(disk_base + disk_noise, 0, 200)

    # ----- Network traffic (Mbps) -----
    net_base = 100 + 60 * np.sin(2 * np.pi * (hours - 9) / 24)
    net_weekly = 15 * np.where(day_of_week < 5, 1, -0.5).astype(float)
    net_noise = np.random.normal(0, 12, n_points)
    net_spikes = np.random.choice([0, 80], size=n_points, p=[0.97, 0.03])
    network = np.clip(net_base + net_weekly + net_noise + net_spikes, 0, 500)

    # ----- Request count (per interval) -----
    req_base = 500 + 300 * np.sin(2 * np.pi * (hours - 9) / 24)
    req_weekly = 80 * np.where(day_of_week < 5, 1, -0.3).astype(float)
    req_noise = np.random.normal(0, 40, n_points)
    requests = np.clip(req_base + req_weekly + req_noise, 10, 2000).astype(int)

    # ----- Active instances (simulated current allocation) -----
    # Traditional threshold-based: scale when CPU > 70%
    instances_threshold = np.clip(np.ceil(cpu / 25), 1, 20).astype(int)

    # ----- Cost computation -----
    cost_per_instance_per_interval = 0.05  # ~$0.60/hr for a medium VM
    cost_threshold = instances_threshold * cost_per_instance_per_interval

    df = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_utilization": np.round(cpu, 2),
        "memory_utilization": np.round(memory, 2),
        "disk_io_mbps": np.round(disk_io, 2),
        "network_mbps": np.round(network, 2),
        "request_count": requests,
        "instances_threshold": instances_threshold,
        "cost_threshold": np.round(cost_threshold, 4),
    })
    return df


def save_dataset(output_dir: str = "data") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = generate_cloud_workload_data()
    path = out / "cloud_workload.csv"
    df.to_csv(path, index=False)
    print(f"[DataGen] Saved {len(df)} records to {path}")
    return path


if __name__ == "__main__":
    save_dataset()
