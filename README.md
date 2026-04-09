# Cloud Resource Cost Optimization using Machine Learning

An end-to-end ML pipeline deployed on **Google Cloud Platform (GCP)** that forecasts cloud resource demands, classifies workload profiles, and uses Reinforcement Learning to autonomously optimize autoscaling decisions — reducing cloud costs by up to 22% while maintaining SLA compliance.

## Project Structure

```
cloud-cost-optimization/
├── main.py                      # Full pipeline runner (local + GCP modes)
├── requirements.txt             # Python dependencies
├── README.md
├── terraform/                   # GCP Infrastructure as Code
│   ├── main.tf                  # MIG, autoscaler, GCS, monitoring, VPC
│   ├── variables.tf             # Configurable parameters
│   └── terraform.tfvars.example # Template for your GCP project settings
├── scripts/
│   ├── setup_gcp.sh             # Automated GCP setup (APIs, service account, Terraform)
│   └── generate_load.sh         # Stress-test load generator for MIG VMs
├── src/
│   ├── __init__.py
│   ├── data_generator.py        # Synthetic cloud workload data generator
│   ├── preprocessing.py         # Data loading, feature engineering, EDA
│   ├── supervised_models.py     # Linear Regression, Random Forest, Gradient Boosting
│   ├── timeseries_models.py     # ARIMA and LSTM forecasting
│   ├── clustering.py            # K-Means workload profiling
│   ├── rl_autoscaler.py         # Q-Learning autoscaler agent + environment
│   ├── cost_optimizer.py        # Strategy comparison engine
│   ├── visualizations.py        # All plotting and architecture diagram
│   ├── gcp_metrics.py           # GCP Cloud Monitoring API metrics collector
│   └── gcp_autoscaler.py        # GCP Compute Engine MIG autoscaler
├── data/                        # Generated dataset (created on first run)
├── results/
│   ├── figures/                 # All generated plots and diagrams
│   └── summary.json             # Machine-readable results summary
└── report/
    └── REPORT.md                # Full project report (all 9 sections)
```

## Quick Start — Local Simulation

```bash
cd cloud-cost-optimization
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## GCP Deployment

### Prerequisites
- GCP account with billing enabled
- `gcloud` CLI installed and authenticated
- Terraform installed

### Step-by-Step

```bash
# 1. Setup GCP (enable APIs, create service account, init Terraform)
./scripts/setup_gcp.sh YOUR_PROJECT_ID

# 2. Deploy infrastructure (MIG, autoscaler, GCS bucket, monitoring)
cd terraform && terraform apply

# 3. Generate workload on VMs (simulates diurnal traffic pattern)
./scripts/generate_load.sh YOUR_PROJECT_ID us-central1-a workload-mig 30

# 4. Collect real metrics from Cloud Monitoring
export GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json
export GCP_PROJECT_ID=YOUR_PROJECT_ID
python main.py --mode gcp-collect --project YOUR_PROJECT_ID --hours 1

# 5. Run full pipeline (train models + live autoscaling)
python main.py --mode gcp --project YOUR_PROJECT_ID

# 6. Run specific strategy live
python main.py --mode gcp-optimize --project YOUR_PROJECT_ID --strategy rl --duration 15

# 7. Tear down when done
cd terraform && terraform destroy
```

## GCP Services Used

| GCP Service | Purpose |
|-------------|---------|
| **Compute Engine** — Managed Instance Groups | Autoscaled VM fleet (e2-medium instances) |
| **Compute Engine** — Autoscaler | Baseline threshold-based CPU scaling (70% target) |
| **Cloud Monitoring** (Ops Agent) | Real-time CPU, memory, disk, network metrics collection |
| **Cloud Monitoring** — Alert Policies | SLA violation detection (CPU > 85%) |
| **Google Cloud Storage** | Model artifacts, workload data, and metrics archive |
| **Terraform** | Infrastructure as Code for all GCP resources |

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

- **LSTM** achieves best forecasting accuracy (R² = 0.977)
- **ML-Predicted autoscaling** reduces costs by **22.2%** vs threshold-based, with zero SLA violations
- **RL-Optimized autoscaling** reduces costs by **17.5%** with only 0.14% SLA violations
- **Resource waste** drops from 45.7% (threshold) to 30.2% (ML-Predicted)

## Execution Modes

| Mode | Command | Description |
|------|---------|-------------|
| Local simulation | `python main.py` | Train on synthetic data, compare strategies |
| GCP full pipeline | `python main.py --mode gcp --project ID` | Collect GCP metrics, train, live optimize |
| GCP collect only | `python main.py --mode gcp-collect --project ID` | Pull metrics from Cloud Monitoring |
| GCP live optimize | `python main.py --mode gcp-optimize --project ID --strategy rl` | Apply live scaling decisions |

## Report

The full project report covering all 9 required sections is at [`report/REPORT.md`](report/REPORT.md).

## Technologies

Python 3.10+ | scikit-learn | TensorFlow/Keras | statsmodels | Google Cloud Platform | Terraform | pandas | matplotlib | seaborn
