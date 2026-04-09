"""
Cloud Resource Cost Optimization using Machine Learning
=======================================================
Main entry point: runs the full pipeline from data generation through
model training, RL optimization, cost comparison, and figure generation.

Usage:
    cd cloud-cost-optimization
    pip install -r requirements.txt
    python main.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_generator import save_dataset
from src.preprocessing import load_data, add_temporal_features, add_lag_features, generate_eda_plots
from src.supervised_models import train_and_evaluate as train_supervised
from src.timeseries_models import run_timeseries_models
from src.clustering import run_clustering
from src.rl_autoscaler import train_rl_autoscaler, evaluate_rl_agent
from src.cost_optimizer import compare_strategies, threshold_strategy, ml_predicted_strategy
from src.visualizations import (
    plot_model_comparison, plot_forecast_vs_actual, plot_cost_comparison,
    plot_instances_timeline, plot_architecture_diagram,
)

RESULTS_DIR = "results"
FIGURES_DIR = f"{RESULTS_DIR}/figures"


def main():
    print("=" * 70)
    print("  Cloud Resource Cost Optimization using Machine Learning")
    print("=" * 70)

    # --- Step 1: Generate data ---
    print("\n[1/7] Generating synthetic cloud workload data...")
    save_dataset()

    # --- Step 2: EDA ---
    print("\n[2/7] Running Exploratory Data Analysis...")
    df = load_data()
    generate_eda_plots(df)

    # --- Step 3: Supervised models ---
    print("\n[3/7] Training supervised forecasting models...")
    sup_results, sup_predictions, y_test_sup, df_test = train_supervised()

    # --- Step 4: Time-series models ---
    print("\n[4/7] Training time-series models (ARIMA + LSTM)...")
    ts_results, ts_data = run_timeseries_models()

    # --- Step 5: Clustering ---
    print("\n[5/7] Running workload clustering...")
    df_clustered, cluster_profiles = run_clustering()

    # --- Step 6: RL Autoscaler ---
    print("\n[6/7] Training RL autoscaler agent...")
    cpu_all = df["cpu_utilization"].values
    rl_agent, _ = train_rl_autoscaler(cpu_all, n_episodes=200)
    rl_metrics = evaluate_rl_agent(rl_agent, cpu_all)

    # --- Step 7: Compare and visualize ---
    print("\n[7/7] Comparing strategies and generating visualizations...")

    # Best supervised model predictions for cost comparison
    best_model = "Gradient Boosting"
    cpu_predicted = sup_predictions[best_model]
    cpu_actual_test = y_test_sup

    all_model_results = {**sup_results, **ts_results}
    plot_model_comparison(all_model_results, FIGURES_DIR)

    for name, preds in sup_predictions.items():
        plot_forecast_vs_actual(y_test_sup, preds, name, FIGURES_DIR)
    for name, (true, pred) in ts_data.items():
        plot_forecast_vs_actual(true, pred, name, FIGURES_DIR)

    # Cost comparison on full dataset
    threshold_inst = threshold_strategy(cpu_all)
    ml_inst = ml_predicted_strategy(cpu_all)
    rl_inst = np.array(rl_metrics["instances_history"])

    comparison = compare_strategies(cpu_all, cpu_all, rl_inst)
    plot_cost_comparison(comparison, FIGURES_DIR)
    plot_instances_timeline(cpu_all, threshold_inst, ml_inst, rl_inst, FIGURES_DIR)
    plot_architecture_diagram(FIGURES_DIR)

    # --- Save summary ---
    summary = {
        "supervised_models": sup_results,
        "timeseries_models": ts_results,
        "cost_comparison": comparison.to_dict(orient="index"),
        "rl_evaluation": {k: v for k, v in rl_metrics.items() if k != "instances_history"},
    }
    summary_path = Path(RESULTS_DIR) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print final results ---
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Forecasting Model Performance ---")
    print(pd.DataFrame(all_model_results).T.to_string())

    print("\n--- Cost Optimization Comparison ---")
    print(comparison.to_string())

    print(f"\n--- RL Autoscaler ---")
    print(f"  Total Cost: ${rl_metrics['total_cost']:,.2f}")
    print(f"  Avg Instances: {rl_metrics['avg_instances']}")
    print(f"  SLA Violation Rate: {rl_metrics['sla_violation_rate_pct']}%")

    # Cost savings
    thresh_cost = comparison.loc["Threshold-Based", "total_cost"]
    rl_cost = comparison.loc["RL-Optimized", "total_cost"]
    ml_cost = comparison.loc["ML-Predicted", "total_cost"]
    print(f"\n--- Cost Savings ---")
    print(f"  ML-Predicted vs Threshold: {(1 - ml_cost/thresh_cost)*100:.1f}% reduction")
    print(f"  RL-Optimized vs Threshold: {(1 - rl_cost/thresh_cost)*100:.1f}% reduction")

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print(f"Summary saved to {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
