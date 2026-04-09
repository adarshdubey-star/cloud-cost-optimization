"""
Unsupervised learning for workload classification.

Uses K-Means clustering to group time intervals into workload profiles
(e.g., low, medium, high, spike) which inform resource allocation policies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path

from src.preprocessing import load_data, add_temporal_features

CLUSTER_FEATURES = [
    "cpu_utilization", "memory_utilization", "disk_io_mbps",
    "network_mbps", "request_count",
]


def run_clustering(n_clusters: int = 4, output_dir: str = "results/figures") -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = add_temporal_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

    # Elbow analysis
    inertias = []
    sil_scores = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels, sample_size=5000))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(K_range), inertias, "o-", color="#1565C0")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax2.plot(list(K_range), sil_scores, "o-", color="#4CAF50")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    plt.tight_layout()
    fig.savefig(out / "clustering_elbow.png", dpi=150)
    plt.close(fig)

    # Final clustering
    km_final = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df["cluster"] = km_final.fit_predict(X_scaled)

    cluster_profiles = df.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
    label_map = cluster_profiles["cpu_utilization"].sort_values().reset_index()
    label_map["profile"] = ["Low", "Medium", "High", "Spike"][:n_clusters]
    profile_lookup = dict(zip(label_map["cluster"], label_map["profile"]))
    df["workload_profile"] = df["cluster"].map(profile_lookup)

    print("[Clustering] Workload Profiles:")
    print(cluster_profiles)
    print(f"  Silhouette score (k={n_clusters}): "
          f"{silhouette_score(X_scaled, df['cluster'], sample_size=5000):.4f}")

    # Profile distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df["workload_profile"].value_counts().plot.bar(ax=ax, color=["#4CAF50", "#2196F3", "#FF9800", "#EF5350"])
    ax.set_title("Workload Profile Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out / "workload_profiles.png", dpi=150)
    plt.close(fig)

    # Cluster scatter (CPU vs Network)
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile, color in zip(["Low", "Medium", "High", "Spike"],
                               ["#4CAF50", "#2196F3", "#FF9800", "#EF5350"]):
        mask = df["workload_profile"] == profile
        ax.scatter(df.loc[mask, "cpu_utilization"], df.loc[mask, "network_mbps"],
                   alpha=0.15, s=5, label=profile, color=color)
    ax.set_xlabel("CPU Utilization (%)")
    ax.set_ylabel("Network Traffic (Mbps)")
    ax.set_title("Workload Clusters: CPU vs Network")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "cluster_scatter.png", dpi=150)
    plt.close(fig)

    return df, cluster_profiles


if __name__ == "__main__":
    df, profiles = run_clustering()
