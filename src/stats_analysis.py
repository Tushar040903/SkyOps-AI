"""
Statistical Validation and Unsupervised Route Clustering Module.
Implements:
1. Hypothesis Testing: One-Way ANOVA across airlines & hubs, Kruskal-Wallis, Chi-Square Test of Independence.
2. Correlation Analysis: Pearson and Spearman matrices with exact p-values.
3. Unsupervised K-Means Clustering on flight routes to uncover operational risk archetypes.
"""

from typing import Dict, Any, Tuple
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

from src.config import (
    ENRICHED_COMPLEXITY_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)
from src.utils import get_logger, set_plot_style

logger = get_logger(__name__)

def run_statistical_hypothesis_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform rigorous statistical tests:
    - One-way ANOVA & Kruskal-Wallis: Flight complexity variance across airline carriers.
    - Chi-Square test of independence: Airline company vs. Complexity Class distribution.
    - Pearson & Spearman correlation testing.
    """
    logger.info("Executing statistical hypothesis tests...")
    results = {}

    # 1. One-Way ANOVA: Does complexity significantly differ across airline carriers?
    companies = df["company_id"].dropna().unique()
    company_groups = [df[df["company_id"] == c]["overall_flight_complexity"].values for c in companies]

    f_stat, p_val_anova = stats.f_oneway(*company_groups)
    kw_stat, p_val_kw = stats.kruskal(*company_groups)

    # Eta squared effect size for ANOVA: SS_between / SS_total
    overall_mean = df["overall_flight_complexity"].mean()
    ss_total = np.sum((df["overall_flight_complexity"] - overall_mean) ** 2)
    ss_between = np.sum([len(g) * (np.mean(g) - overall_mean) ** 2 for g in company_groups])
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    results["anova_airline_complexity"] = {
        "test": "One-Way ANOVA",
        "groups": list(companies),
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_val_anova),
        "statistically_significant": bool(p_val_anova < 0.05),
        "eta_squared_effect_size": round(float(eta_squared), 4),
        "kruskal_wallis_h_statistic": round(float(kw_stat), 4),
        "kruskal_wallis_p_value": float(p_val_kw),
        "interpretation": "Rejects null hypothesis: Flight complexity distributions vary significantly across airline carriers." if p_val_anova < 0.05 else "Fail to reject null hypothesis.",
    }

    # 2. Chi-Square Test of Independence: Airline vs Complexity Class (Easy/Medium/Hard)
    contingency_table = pd.crosstab(df["company_id"], df["complexity_class"])
    chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

    # Cramér's V effect size
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0.0

    results["chi_square_airline_vs_class"] = {
        "test": "Chi-Square Test of Independence",
        "chi2_statistic": round(float(chi2), 4),
        "p_value": float(p_val_chi2),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "statistically_significant": bool(p_val_chi2 < 0.05),
        "contingency_table": contingency_table.to_dict(),
        "interpretation": "Rejects null hypothesis: Complexity tier assignment is strongly dependent on airline operator." if p_val_chi2 < 0.05 else "Fail to reject null hypothesis.",
    }

    # 3. Correlation Testing
    corr_features = [
        "delay_complexity_score",
        "passenger_complexity_score",
        "bag_complexity_score",
        "buffer_minutes",
        "load_factor",
        "hot_transfer_ratio",
        "ssr_total",
        "overall_flight_complexity",
    ]
    corr_sub = df[corr_features].dropna()
    pearson_corr = corr_sub.corr(method="pearson")
    spearman_corr = corr_sub.corr(method="spearman")

    # Save correlation heatmap
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    sns.heatmap(
        pearson_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    ax.set_title("Statistical Correlation Heatmap of Operational Complexity Drivers", pad=12)
    plt.tight_layout()
    corr_fig_path = FIGURES_DIR / "correlation_matrix_statistical.png"
    fig.savefig(corr_fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved correlation matrix plot to {corr_fig_path}")

    # Export statistical test findings
    stats_json_path = REPORTS_DIR / "statistical_testing_summary.json"
    with open(stats_json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Exported statistical testing report to: {stats_json_path}")

    return results

def run_route_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Perform Unsupervised K-Means Clustering on airline routes to discover operational risk archetypes.
    """
    logger.info("Executing unsupervised route-level operational clustering...")
    set_plot_style()

    # Aggregate operational metrics at the route level
    route_agg = df.groupby("route").agg(
        flight_volume=("flight_number", "count"),
        mean_complexity=("overall_flight_complexity", "mean"),
        mean_dep_delay=("departure_delay", "mean"),
        mean_arr_delay=("arrival_delay", "mean"),
        mean_buffer=("buffer_minutes", "mean"),
        mean_load_factor=("load_factor", "mean"),
        mean_transfer_ratio=("transfer_ratio", "mean"),
        mean_hot_transfer_ratio=("hot_transfer_ratio", "mean"),
        mean_ssr_total=("ssr_total", "mean"),
        hard_flight_ratio=("high_complexity_flag", "mean"),
    ).reset_index()

    # Filter for routes with meaningful frequency (>= 3 flights)
    active_routes = route_agg[route_agg["flight_volume"] >= 3].copy()
    logger.info(f"Clustering {len(active_routes)} active routes (volume >= 3 flights)...")

    cluster_features = [
        "mean_complexity",
        "mean_dep_delay",
        "mean_buffer",
        "mean_load_factor",
        "mean_transfer_ratio",
        "mean_hot_transfer_ratio",
        "mean_ssr_total",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(active_routes[cluster_features])

    # K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    active_routes["cluster"] = cluster_labels
    sil_score = silhouette_score(X_scaled, cluster_labels)
    logger.info(f"K-Means (k={n_clusters}) Silhouette Score: {sil_score:.4f}")

    # Assign meaningful operational archetype names based on cluster centroids
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_features)
    cluster_names = {}
    for c_idx in range(n_clusters):
        row = centroids.iloc[c_idx]
        if row["mean_complexity"] > centroids["mean_complexity"].median() and row["mean_hot_transfer_ratio"] > centroids["mean_hot_transfer_ratio"].median():
            cluster_names[c_idx] = "High-Risk Transfer Bottleneck"
        elif row["mean_buffer"] < centroids["mean_buffer"].median():
            cluster_names[c_idx] = "Turnaround-Squeezed Hub Feeder"
        else:
            cluster_names[c_idx] = "Standard Low-Stress Route"

    active_routes["cluster_name"] = active_routes["cluster"].map(cluster_names)

    # 2D PCA Visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    active_routes["pca1"] = coords[:, 0]
    active_routes["pca2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    palette = {"High-Risk Transfer Bottleneck": "#C0392B", "Turnaround-Squeezed Hub Feeder": "#E67E22", "Standard Low-Stress Route": "#27AE60"}
    sns.scatterplot(
        data=active_routes,
        x="pca1",
        y="pca2",
        hue="cluster_name",
        palette=palette,
        style="cluster_name",
        s=70,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title(f"Unsupervised Route Clustering (K-Means k={n_clusters}, Silhouette={sil_score:.3f})\nOperational Risk Archetypes", pad=12)
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", labelpad=8)
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", labelpad=8)
    ax.legend(title="Operational Risk Archetype", frameon=True)
    plt.tight_layout()
    cluster_fig_path = FIGURES_DIR / "route_clusters_kmeans.png"
    fig.savefig(cluster_fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved route cluster visualization to: {cluster_fig_path}")

    # Top 15 Hardest Routes
    top_hard_routes = active_routes.sort_values("mean_complexity", ascending=False).head(15)
    top_routes_path = REPORTS_DIR / "route_complexity_rankings.csv"
    active_routes.sort_values("mean_complexity", ascending=False).to_csv(top_routes_path, index=False)
    logger.info(f"Saved route rankings to: {top_routes_path}")

    # Save model
    joblib.dump({
        "kmeans": kmeans,
        "scaler": scaler,
        "features": cluster_features,
        "cluster_names": cluster_names,
        "silhouette_score": sil_score,
    }, MODELS_DIR / "kmeans_routes.joblib")

    return active_routes

if __name__ == "__main__":
    df_enriched = pd.read_csv(ENRICHED_COMPLEXITY_PATH)
    test_results = run_statistical_hypothesis_tests(df_enriched)
    routes_clustered = run_route_clustering(df_enriched)
    print("\nSTATISTICAL ANOVA RESULT:")
    print("F-stat:", test_results["anova_airline_complexity"]["f_statistic"], "p-val:", test_results["anova_airline_complexity"]["p_value"])
    print("\nCHI-SQUARE RESULT:")
    print("Chi2:", test_results["chi_square_airline_vs_class"]["chi2_statistic"], "p-val:", test_results["chi_square_airline_vs_class"]["p_value"])
    print("\nTOP 5 HARDEST ROUTES:")
    sample_routes = routes_clustered[["route", "flight_volume", "mean_complexity", "cluster_name"]].head(5).copy()
    sample_routes["route"] = sample_routes["route"].astype(str).str.replace("\u2192", "->")
    print(sample_routes.to_string(index=False))
