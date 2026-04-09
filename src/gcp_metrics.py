"""
GCP Cloud Monitoring metrics collector.

Pulls real CPU, memory, disk, and network metrics from Google Cloud Monitoring
for Compute Engine instances in a Managed Instance Group.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import monitoring_v3
from google.protobuf.timestamp_pb2 import Timestamp


PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")

METRIC_MAP = {
    "cpu_utilization": "compute.googleapis.com/instance/cpu/utilization",
    "memory_utilization": "agent.googleapis.com/memory/percent_used",
    "disk_io_read": "compute.googleapis.com/instance/disk/read_bytes_count",
    "disk_io_write": "compute.googleapis.com/instance/disk/write_bytes_count",
    "network_received": "compute.googleapis.com/instance/network/received_bytes_count",
    "network_sent": "compute.googleapis.com/instance/network/sent_bytes_count",
}


def get_monitoring_client():
    return monitoring_v3.MetricServiceClient()


def fetch_metric(
    client,
    project_id: str,
    metric_type: str,
    hours_back: int = 24,
    alignment_period_seconds: int = 300,
) -> pd.DataFrame:
    """Fetch a single metric from Cloud Monitoring for all GCE instances."""
    project_name = f"projects/{project_id}"

    now = time.time()
    start = now - (hours_back * 3600)

    interval = monitoring_v3.TimeInterval()
    interval.end_time = Timestamp(seconds=int(now))
    interval.start_time = Timestamp(seconds=int(start))

    aggregation = monitoring_v3.Aggregation()
    aggregation.alignment_period = {"seconds": alignment_period_seconds}
    aggregation.per_series_aligner = monitoring_v3.Aggregation.Aligner.ALIGN_MEAN

    request = monitoring_v3.ListTimeSeriesRequest(
        name=project_name,
        filter=f'metric.type = "{metric_type}" AND resource.type = "gce_instance"',
        interval=interval,
        aggregation=aggregation,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    )

    results = client.list_time_series(request=request)

    records = []
    for ts in results:
        instance_id = ts.resource.labels.get("instance_id", "unknown")
        for point in ts.points:
            records.append({
                "timestamp": point.interval.end_time.timestamp(),
                "instance_id": instance_id,
                "value": point.value.double_value,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def collect_all_metrics(
    project_id: str = None,
    hours_back: int = 24,
    output_path: str = "data/gcp_metrics.csv",
) -> pd.DataFrame:
    """Collect all metrics and merge into a single DataFrame."""
    project_id = project_id or PROJECT_ID
    if not project_id:
        raise ValueError("Set GCP_PROJECT_ID environment variable or pass project_id")

    client = get_monitoring_client()
    print(f"[GCP Metrics] Collecting {hours_back}h of metrics from project '{project_id}'...")

    all_dfs = {}
    for name, metric_type in METRIC_MAP.items():
        print(f"  Fetching {name}...")
        try:
            df = fetch_metric(client, project_id, metric_type, hours_back)
            if not df.empty:
                agg = df.groupby("timestamp")["value"].mean().reset_index()
                agg.rename(columns={"value": name}, inplace=True)
                all_dfs[name] = agg
                print(f"    -> {len(agg)} data points")
            else:
                print(f"    -> No data (metric may not be enabled)")
        except Exception as e:
            print(f"    -> Error: {e}")

    if not all_dfs:
        print("[GCP Metrics] No metrics collected. Ensure Ops Agent is installed on VMs.")
        return pd.DataFrame()

    merged = None
    for name, df in all_dfs.items():
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="timestamp", how="outer")

    merged.sort_values("timestamp", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # CPU comes as 0-1 fraction from GCP, convert to percentage
    if "cpu_utilization" in merged.columns:
        merged["cpu_utilization"] = merged["cpu_utilization"] * 100

    # Convert network bytes to Mbps (bytes per 300s -> Mbps)
    for col in ["network_received", "network_sent"]:
        if col in merged.columns:
            merged[col] = merged[col] * 8 / (300 * 1e6)  # bytes -> Mbps

    # Combine network into single column
    if "network_received" in merged.columns and "network_sent" in merged.columns:
        merged["network_mbps"] = merged["network_received"] + merged["network_sent"]
        merged.drop(columns=["network_received", "network_sent"], inplace=True)

    # Combine disk I/O into single column (MB/s)
    if "disk_io_read" in merged.columns and "disk_io_write" in merged.columns:
        merged["disk_io_mbps"] = (merged["disk_io_read"] + merged["disk_io_write"]) / (300 * 1e6)
        merged.drop(columns=["disk_io_read", "disk_io_write"], inplace=True)

    merged.to_csv(output_path, index=False)
    print(f"[GCP Metrics] Saved {len(merged)} records to {output_path}")
    return merged


def get_instance_count(project_id: str, zone: str, mig_name: str) -> int:
    """Get current number of running instances in a MIG."""
    from google.cloud import compute_v1

    client = compute_v1.InstanceGroupManagersClient()
    mig = client.get(project=project_id, zone=zone, instance_group_manager=mig_name)
    return mig.target_size


def get_current_cost_rate(instance_count: int, machine_type: str = "e2-medium") -> float:
    """Estimate current hourly cost based on instance count."""
    # GCP e2-medium pricing (us-central1, on-demand, approx.)
    PRICING = {
        "e2-micro": 0.0084,
        "e2-small": 0.0168,
        "e2-medium": 0.0336,
        "e2-standard-2": 0.0672,
        "e2-standard-4": 0.1344,
        "n1-standard-1": 0.0475,
    }
    rate = PRICING.get(machine_type, 0.0336)
    return instance_count * rate


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else None
    df = collect_all_metrics(project_id=project, hours_back=6)
    if not df.empty:
        print(df.head())
        print(f"\nColumns: {df.columns.tolist()}")
