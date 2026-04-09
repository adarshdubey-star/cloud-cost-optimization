# Cloud Resource Cost Optimization using Machine Learning

An end-to-end ML pipeline that forecasts cloud resource demands, classifies workload profiles, and uses Reinforcement Learning to autonomously optimize autoscaling decisions — reducing cloud costs by up to 31% while maintaining SLA compliance.

## Project Structure

```
cloud-cost-optimization/
├── main.py                      # Full pipeline runner
├── requirements.txt             # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_generator.py        # Synthetic cloud workload data generator
│   ├── preprocessing.py         # Data loading, feature engineering, EDA
│   ├── supervised_models.py     # Linear Regression, Random Forest, Gradient Boosting
│   ├── timeseries_models.py     # ARIMA and LSTM forecasting
│   ├── clustering.py            # K-Means workload profiling
│   ├── rl_autoscaler.py         # Q-Learning autoscaler agent + environment
│   ├── cost_optimizer.py        # Strategy comparison engine
│   └── visualizations.py        # All plotting and architecture diagram
├── data/                        # Generated dataset (created on first run)
├── results/
│   ├── figures/                 # All generated plots and diagrams
│   └── summary.json             # Machine-readable results summary
└── report/
    └── REPORT.md                # Full project report (all 9 sections)
```

## Quick Start

```bash
# 1. Clone or navigate to the project directory
cd cloud-cost-optimization

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

The pipeline will:
1. Generate 90 days of synthetic cloud workload data (25,920 records at 5-min intervals)
2. Run Exploratory Data Analysis and save plots
3. Train 5 forecasting models (Linear Regression, Random Forest, Gradient Boosting, ARIMA, LSTM)
4. Perform K-Means workload clustering
5. Train a Q-Learning RL autoscaler agent
6. Compare three cost strategies (Threshold vs ML-Predicted vs RL-Optimized)
7. Generate all figures, comparison tables, and an architecture diagram

## Models Implemented

| Category | Model | Purpose |
|----------|-------|---------|
| Supervised | Linear Regression | Baseline demand forecasting |
| Supervised | Random Forest | Non-linear ensemble forecasting |
| Supervised | Gradient Boosting | State-of-the-art tabular forecasting |
| Time-Series | ARIMA(5,1,2) | Classical statistical forecasting |
| Time-Series | LSTM | Deep learning sequence forecasting |
| Unsupervised | K-Means (k=4) | Workload profile classification |
| Reinforcement Learning | Q-Learning | Dynamic autoscaling optimization |

## Key Results

- **Gradient Boosting** achieves best forecasting accuracy (R² ≈ 0.97)
- **RL-Optimized autoscaling** reduces costs by ~31% vs threshold-based scaling
- **ML-Predicted scaling** reduces costs by ~20% with minimal SLA impact
- **Resource waste** drops from ~45% (threshold) to ~22% (RL-optimized)

## Generated Outputs

After running `main.py`, check `results/figures/` for:
- `timeseries_overview.png` — Workload time-series visualization
- `correlation_heatmap.png` — Feature correlation matrix
- `cpu_hourly_boxplot.png` — CPU utilization by hour of day
- `daily_cost_vs_cpu.png` — Cost vs utilization trends
- `clustering_elbow.png` — Elbow and silhouette analysis
- `workload_profiles.png` — Cluster distribution
- `cluster_scatter.png` — 2D cluster visualization
- `rl_training.png` — RL agent training convergence
- `model_comparison.png` — All models performance comparison
- `forecast_*.png` — Actual vs predicted for each model
- `cost_comparison.png` — Strategy cost/waste/SLA comparison
- `instances_timeline.png` — Instance allocation over time
- `architecture_diagram.png` — System architecture

## Report

The full project report covering all 9 required sections is at [`report/REPORT.md`](report/REPORT.md).

## Technologies

Python 3.10+ | scikit-learn | TensorFlow/Keras | statsmodels | pandas | matplotlib | seaborn
