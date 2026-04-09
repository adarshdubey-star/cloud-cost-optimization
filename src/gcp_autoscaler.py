"""
GCP Managed Instance Group autoscaler — applies ML/RL scaling decisions.

Connects to the Compute Engine API to resize the MIG based on:
  1. ML-predicted demand (proactive scaling)
  2. RL agent recommended action (dynamic optimization)
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from google.cloud import compute_v1

from src.gcp_metrics import collect_all_metrics, get_instance_count, get_current_cost_rate


PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
ZONE = os.environ.get("GCP_ZONE", "us-central1-a")
MIG_NAME = os.environ.get("GCP_MIG_NAME", "workload-mig")
INSTANCE_CAPACITY_CPU = 25.0


def resize_mig(project_id: str, zone: str, mig_name: str, target_size: int) -> dict:
    """Resize the Managed Instance Group to the target number of instances."""
    client = compute_v1.InstanceGroupManagersClient()

    current = get_instance_count(project_id, zone, mig_name)
    target_size = max(1, min(target_size, 10))  # safety bounds

    if current == target_size:
        action = "no_change"
    elif target_size > current:
        action = "scale_up"
    else:
        action = "scale_down"

    print(f"[GCP Autoscaler] {action}: {current} -> {target_size} instances")

    if current != target_size:
        request = compute_v1.ResizeInstanceGroupManagerRequest(
            project=project_id,
            zone=zone,
            instance_group_manager=mig_name,
            size=target_size,
        )
        operation = client.resize(request=request)
        print(f"  Operation: {operation.name}")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "previous_size": current,
        "target_size": target_size,
        "action": action,
        "cost_rate_hourly": get_current_cost_rate(target_size),
    }


def ml_scaling_decision(cpu_predicted: float, headroom: float = 1.10) -> int:
    """Determine instance count from ML-predicted CPU demand."""
    demand = cpu_predicted * headroom
    return max(1, min(int(np.ceil(demand / INSTANCE_CAPACITY_CPU)), 10))


def rl_scaling_decision(agent, cpu_current: float, current_instances: int) -> int:
    """Use the trained RL agent to decide scaling action."""
    cpu_bucket = min(int(cpu_current // 10), 9)
    inst_bucket = min(current_instances - 1, 9)
    state = cpu_bucket * 10 + inst_bucket

    action = int(np.argmax(agent.q_table[state]))  # greedy

    if action == 0 and current_instances > 1:
        return current_instances - 1
    elif action == 2 and current_instances < 10:
        return current_instances + 1
    return current_instances


def run_live_optimization(
    project_id: str = None,
    zone: str = None,
    mig_name: str = None,
    strategy: str = "ml",
    duration_minutes: int = 30,
    interval_seconds: int = 60,
    output_dir: str = "results",
):
    """
    Run live autoscaling optimization on GCP.

    strategy: "ml" for ML-predicted, "rl" for RL agent, "threshold" for baseline
    """
    project_id = project_id or PROJECT_ID
    zone = zone or ZONE
    mig_name = mig_name or MIG_NAME
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Live GCP Optimization — Strategy: {strategy.upper()}")
    print(f"  Project: {project_id} | Zone: {zone} | MIG: {mig_name}")
    print(f"  Duration: {duration_minutes}min | Interval: {interval_seconds}s")
    print(f"{'='*60}\n")

    rl_agent = None
    if strategy == "rl":
        from src.rl_autoscaler import QLearningAgent
        agent_path = out / "rl_qtable.npy"
        if agent_path.exists():
            rl_agent = QLearningAgent()
            rl_agent.q_table = np.load(agent_path)
            rl_agent.epsilon = 0.0
            print("[GCP] Loaded trained RL agent")
        else:
            print("[GCP] WARNING: No trained RL agent found. Run main.py first.")
            return

    log = []
    n_steps = duration_minutes * 60 // interval_seconds

    for step in range(n_steps):
        # 1. Collect current metrics
        metrics_df = collect_all_metrics(project_id, hours_back=1)
        if metrics_df.empty or "cpu_utilization" not in metrics_df.columns:
            print(f"  Step {step+1}: No metrics available, skipping...")
            time.sleep(interval_seconds)
            continue

        current_cpu = metrics_df["cpu_utilization"].iloc[-1]
        current_instances = get_instance_count(project_id, zone, mig_name)

        # 2. Decide target size
        if strategy == "ml":
            target = ml_scaling_decision(current_cpu)
        elif strategy == "rl":
            target = rl_scaling_decision(rl_agent, current_cpu, current_instances)
        else:
            target = max(1, min(int(np.ceil(current_cpu * 1.5 / INSTANCE_CAPACITY_CPU)), 10))

        # 3. Apply scaling
        result = resize_mig(project_id, zone, mig_name, target)
        result["cpu_utilization"] = round(current_cpu, 2)
        result["step"] = step + 1
        log.append(result)

        print(f"  Step {step+1}/{n_steps} | CPU: {current_cpu:.1f}% | "
              f"Instances: {result['previous_size']}->{result['target_size']} | "
              f"Cost: ${result['cost_rate_hourly']:.4f}/hr")

        if step < n_steps - 1:
            time.sleep(interval_seconds)

    # Save log
    log_df = pd.DataFrame(log)
    log_path = out / f"gcp_live_{strategy}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[GCP] Optimization log saved to {log_path}")

    total_cost = log_df["cost_rate_hourly"].sum() * (interval_seconds / 3600)
    avg_instances = log_df["target_size"].mean()
    print(f"[GCP] Total estimated cost: ${total_cost:.4f}")
    print(f"[GCP] Avg instances: {avg_instances:.1f}")

    return log_df


if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "ml"
    run_live_optimization(strategy=strategy, duration_minutes=10, interval_seconds=60)
