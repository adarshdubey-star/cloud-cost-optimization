"""
Cost optimization engine.

Compares three resource allocation strategies:
  1. Threshold-based (static rule: scale when CPU > threshold)
  2. ML-predicted (use Gradient Boosting forecast to right-size)
  3. RL-optimized (Q-Learning dynamic autoscaler)

Computes total cost, waste, and SLA metrics for each.
"""

import numpy as np
import pandas as pd
from typing import Dict

COST_PER_INSTANCE_PER_STEP = 0.05
INSTANCE_CAPACITY_CPU = 25.0


def compute_cost_metrics(
    cpu_actual: np.ndarray,
    instances: np.ndarray,
) -> Dict[str, float]:
    total_cost = float(np.sum(instances) * COST_PER_INSTANCE_PER_STEP)
    capacity = instances * INSTANCE_CAPACITY_CPU
    over_provisioned = np.clip(capacity - cpu_actual, 0, None)
    under_provisioned = np.clip(cpu_actual - capacity, 0, None)
    waste_pct = float(np.sum(over_provisioned) / np.sum(capacity) * 100) if np.sum(capacity) > 0 else 0
    sla_violations = int(np.sum(under_provisioned > 0))
    sla_rate = sla_violations / len(cpu_actual) * 100

    return {
        "total_cost": round(total_cost, 2),
        "avg_instances": round(float(np.mean(instances)), 2),
        "waste_pct": round(waste_pct, 2),
        "sla_violations": sla_violations,
        "sla_violation_rate_pct": round(sla_rate, 2),
    }


def threshold_strategy(cpu_actual: np.ndarray, safety_margin: float = 1.5) -> np.ndarray:
    """Classic reactive approach with a conservative safety margin (typical ops behavior)."""
    return np.clip(np.ceil(cpu_actual * safety_margin / INSTANCE_CAPACITY_CPU), 1, 20).astype(int)


def ml_predicted_strategy(cpu_predicted: np.ndarray, headroom: float = 1.10) -> np.ndarray:
    """Proactive scaling based on ML forecasted demand with a modest safety headroom."""
    demand_with_headroom = cpu_predicted * headroom
    return np.clip(np.ceil(demand_with_headroom / INSTANCE_CAPACITY_CPU), 1, 20).astype(int)


def compare_strategies(
    cpu_actual: np.ndarray,
    cpu_predicted: np.ndarray,
    rl_instances: np.ndarray,
) -> pd.DataFrame:
    strategies = {
        "Threshold-Based": threshold_strategy(cpu_actual),
        "ML-Predicted": ml_predicted_strategy(cpu_predicted),
        "RL-Optimized": np.array(rl_instances),
    }
    rows = []
    for name, instances in strategies.items():
        metrics = compute_cost_metrics(cpu_actual, instances)
        metrics["strategy"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("strategy")
    return df


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    cpu = 30 + 25 * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, 3, n)
    cpu = np.clip(cpu, 1, 100)
    predicted = cpu + np.random.normal(0, 2, n)
    rl_inst = np.clip(np.ceil(cpu / 28), 1, 20).astype(int)

    comparison = compare_strategies(cpu, predicted, rl_inst)
    print(comparison)
