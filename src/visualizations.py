"""
Result visualization module.

Generates all publication-quality plots for the report: model comparisons,
cost breakdowns, strategy comparisons, and architecture-style diagrams.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


def plot_model_comparison(results: dict, output_dir: str = "results/figures"):
    """Bar chart comparing MAE/RMSE/R2 across all models."""
    out = Path(output_dir)
    df = pd.DataFrame(results).T
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["MAE", "RMSE", "R2"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#EF5350", "#9C27B0"]

    for i, metric in enumerate(metrics):
        bars = axes[i].bar(df.index, df[metric], color=colors[:len(df)])
        axes[i].set_title(metric, fontsize=14, fontweight="bold")
        axes[i].set_ylabel(metric)
        for bar in bars:
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        axes[i].tick_params(axis="x", rotation=30)

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_vs_actual(y_true, y_pred, model_name, output_dir="results/figures", n_points=500):
    """Overlay of actual vs predicted values."""
    out = Path(output_dir)
    n_points = min(n_points, len(y_true), len(y_pred))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true[:n_points], label="Actual", color="#1565C0", linewidth=0.8)
    ax.plot(y_pred[:n_points], label=f"{model_name} Predicted", color="#EF5350",
            linewidth=0.8, alpha=0.8)
    ax.fill_between(range(n_points), y_true[:n_points], y_pred[:n_points],
                     alpha=0.15, color="#EF5350")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("CPU Utilization (%)")
    ax.set_title(f"{model_name}: Actual vs Predicted CPU Utilization")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / f"forecast_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def plot_cost_comparison(comparison_df: pd.DataFrame, output_dir: str = "results/figures"):
    """Side-by-side cost and waste comparison across strategies."""
    out = Path(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    strategies = comparison_df.index.tolist()
    colors = ["#EF5350", "#2196F3", "#4CAF50"]

    # Total cost
    axes[0].bar(strategies, comparison_df["total_cost"], color=colors)
    axes[0].set_title("Total Cost ($)", fontweight="bold")
    axes[0].set_ylabel("Cost ($)")
    for j, v in enumerate(comparison_df["total_cost"]):
        axes[0].text(j, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=10)

    # Waste %
    axes[1].bar(strategies, comparison_df["waste_pct"], color=colors)
    axes[1].set_title("Resource Waste (%)", fontweight="bold")
    axes[1].set_ylabel("Waste (%)")
    for j, v in enumerate(comparison_df["waste_pct"]):
        axes[1].text(j, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    # SLA violations
    axes[2].bar(strategies, comparison_df["sla_violation_rate_pct"], color=colors)
    axes[2].set_title("SLA Violation Rate (%)", fontweight="bold")
    axes[2].set_ylabel("Violation Rate (%)")
    for j, v in enumerate(comparison_df["sla_violation_rate_pct"]):
        axes[2].text(j, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Cost Optimization: Strategy Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out / "cost_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_instances_timeline(
    cpu_actual: np.ndarray,
    threshold_inst: np.ndarray,
    ml_inst: np.ndarray,
    rl_inst: np.ndarray,
    output_dir: str = "results/figures",
    n_points: int = 1440,  # 5 days at 5-min intervals
):
    """Timeline of instance counts across strategies overlaid on CPU demand."""
    out = Path(output_dir)
    fig, ax1 = plt.subplots(figsize=(16, 5))

    x = range(n_points)
    ax1.fill_between(x, cpu_actual[:n_points], alpha=0.2, color="#9E9E9E", label="CPU Demand")
    ax1.set_ylabel("CPU Utilization (%)", color="#616161")
    ax1.set_xlabel("Time Step (5-min intervals)")

    ax2 = ax1.twinx()
    ax2.step(x, threshold_inst[:n_points], where="post", label="Threshold", color="#EF5350", linewidth=1.2, alpha=0.8)
    ax2.step(x, ml_inst[:n_points], where="post", label="ML-Predicted", color="#2196F3", linewidth=1.2, alpha=0.8)
    ax2.step(x, rl_inst[:n_points], where="post", label="RL-Optimized", color="#4CAF50", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Instance Count")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Resource Allocation Timeline: All Strategies")
    plt.tight_layout()
    fig.savefig(out / "instances_timeline.png", dpi=150)
    plt.close(fig)


def plot_architecture_diagram(output_dir: str = "results/figures"):
    """Programmatic architecture diagram of the ML-driven cloud optimization system."""
    out = Path(output_dir)
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2)
    box_green = dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2)
    box_orange = dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2)
    box_red = dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", edgecolor="#C62828", linewidth=2)
    box_purple = dict(boxstyle="round,pad=0.5", facecolor="#F3E5F5", edgecolor="#6A1B9A", linewidth=2)

    # Title
    ax.text(9, 11.5, "ML-Driven Cloud Resource Cost Optimization Architecture",
            fontsize=16, fontweight="bold", ha="center", va="center")

    # Data Sources
    ax.text(2, 10, "Cloud Metrics\nCollector\n(CPU, Mem, Disk,\nNetwork, Requests)",
            fontsize=9, ha="center", va="center", bbox=box_style)

    ax.text(6, 10, "Historical\nWorkload DB\n(Time-Series Store)",
            fontsize=9, ha="center", va="center", bbox=box_style)

    # ML Pipeline
    ax.text(2, 7.5, "Data Preprocessing\n& Feature Engineering\n(Lag, Rolling, Temporal)",
            fontsize=9, ha="center", va="center", bbox=box_green)

    ax.text(6, 7.5, "Supervised Models\n(Linear Reg, RF,\nGradient Boosting)",
            fontsize=9, ha="center", va="center", bbox=box_green)

    ax.text(10, 7.5, "Time-Series Models\n(ARIMA, LSTM)",
            fontsize=9, ha="center", va="center", bbox=box_green)

    ax.text(14, 7.5, "Unsupervised\n(K-Means Clustering\nWorkload Profiles)",
            fontsize=9, ha="center", va="center", bbox=box_green)

    # Decision Engine
    ax.text(6, 5, "Demand\nForecasting\nEngine",
            fontsize=9, ha="center", va="center", bbox=box_orange)

    ax.text(12, 5, "RL Autoscaler\n(Q-Learning Agent)\nAction: Scale Up/Down/Keep",
            fontsize=9, ha="center", va="center", bbox=box_purple)

    # Cloud Platform
    ax.text(9, 2.5, "Cloud Platform (AWS/Azure/GCP)\nAuto-Scaling Group | VM Fleet | Spot Instances",
            fontsize=10, ha="center", va="center", bbox=box_red)

    # Outputs
    ax.text(3, 0.8, "Cost Dashboard\n& Alerts",
            fontsize=9, ha="center", va="center", bbox=box_style)

    ax.text(9, 0.8, "Optimized\nResource Allocation",
            fontsize=9, ha="center", va="center", bbox=box_style)

    ax.text(15, 0.8, "Performance\nMonitoring & SLA",
            fontsize=9, ha="center", va="center", bbox=box_style)

    # Arrows
    arrow_kw = dict(arrowstyle="->", color="#424242", lw=1.5)
    ax.annotate("", xy=(2, 8.8), xytext=(2, 9.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(6, 8.8), xytext=(6, 9.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(4, 7.5), xytext=(3.2, 7.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(8, 7.5), xytext=(7.2, 7.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(12, 7.5), xytext=(11.2, 7.5), arrowprops=arrow_kw)

    ax.annotate("", xy=(6, 5.8), xytext=(6, 6.7), arrowprops=arrow_kw)
    ax.annotate("", xy=(10, 5.8), xytext=(10, 6.7), arrowprops=arrow_kw)
    ax.annotate("", xy=(14, 5.8), xytext=(14, 6.7), arrowprops=arrow_kw)

    ax.annotate("", xy=(9, 5), xytext=(7.2, 5), arrowprops=arrow_kw)
    ax.annotate("", xy=(9, 3.3), xytext=(9, 4.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(12, 3.3), xytext=(12, 4.2), arrowprops=arrow_kw)

    ax.annotate("", xy=(3, 1.5), xytext=(6, 2.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(9, 1.5), xytext=(9, 1.9), arrowprops=arrow_kw)
    ax.annotate("", xy=(15, 1.5), xytext=(12, 2.0), arrowprops=arrow_kw)

    # Feedback loop
    ax.annotate("", xy=(16, 10), xytext=(16, 2.5),
                arrowprops=dict(arrowstyle="->", color="#6A1B9A", lw=1.5, ls="--"))
    ax.text(16.8, 6, "Feedback\nLoop", fontsize=8, ha="center", color="#6A1B9A", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(out / "architecture_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Architecture diagram saved to {out / 'architecture_diagram.png'}")


if __name__ == "__main__":
    plot_architecture_diagram()
