"""
Publication-Quality Figure & Table Generation Suite for AEN-PPO Research.
Generates all 13 mandatory scientific plots, evaluation tables, and statistical summaries:
1. Learning curves comparing the three algorithms with 95% CI bands.
2. Mean reward comparison bar chart with error bars.
3. Completed deliveries / throughput comparison.
4. Invalid action comparison across algorithms.
5. Raw entropy vs. normalized entropy dynamics.
6. Mean valid action count |A_valid| across configurations.
7. Performance comparison across different truck counts (2, 4, 6, 8).
8. Performance comparison across different pit counts (1, 2, 3, 4).
9. Performance comparison across different job counts (4, 6, 8, 12).
10. Multi-seed learning curves with confidence bands across training seeds.
11. Comprehensive evaluation table with all mandatory PPT metrics.
12. Scalability comparison table.
13. Statistical significance comparison table (p-values & Cohen's d).
14. Generalization gap plot (seen vs. unseen map).
15. Computational efficiency & inference latency trade-off.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_MAP = {
    "ppo": "#e74c3c",          # Standard PPO (Red)
    "masked_ppo": "#2980b9",   # Masked PPO (Blue)
    "aen_ppo": "#27ae60",      # Proposed AEN-PPO (Emerald Green)
}

LABEL_MAP = {
    "ppo": "Standard PPO",
    "masked_ppo": "Masked PPO",
    "aen_ppo": "Proposed AEN-PPO",
}


def load_all_training_logs(raw_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(raw_dir, "train_*.csv"))
    records = []
    for f in files:
        basename = os.path.basename(f).replace("train_", "").replace(".csv", "")
        parts = basename.split("_seed_")
        if len(parts) != 2:
            continue
        cfg_algo, seed = parts[0], int(parts[1])
        # Find which algo it is
        algo = None
        for a in ["aen_ppo", "masked_ppo", "ppo"]:
            if cfg_algo.endswith(f"_{a}"):
                algo = a
                cfg = cfg_algo[:-len(f"_{a}")]
                break
        if algo is None:
            continue

        df = pd.read_csv(f)
        df["config"] = cfg
        df["algorithm"] = algo
        df["seed"] = seed
        records.append(df)

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def generate_all_plots(results_root: str = "results"):
    raw_dir = os.path.join(results_root, "evaluation", "raw_results")
    agg_dir = os.path.join(results_root, "evaluation", "aggregated_results")
    stat_dir = os.path.join(results_root, "evaluation", "statistical_tests")
    plots_dir = os.path.join(results_root, "evaluation", "plots")
    tables_dir = os.path.join(results_root, "evaluation", "tables")

    for d in [plots_dir, tables_dir]:
        os.makedirs(d, exist_ok=True)

    summary_csv = os.path.join(agg_dir, "aggregated_metrics.csv")
    stat_csv = os.path.join(stat_dir, "statistical_significance.csv")

    if not os.path.exists(summary_csv):
        print(f"Error: {summary_csv} not found. Run experiments first!")
        return

    df_summary = pd.read_csv(summary_csv)
    df_stat = pd.read_csv(stat_csv) if os.path.exists(stat_csv) else pd.DataFrame()
    df_train = load_all_training_logs(raw_dir)

    print("Generating Publication-Quality Plots...")

    # =========================================================================
    # PLOT 1: Baseline Learning Curves with 95% Confidence Bands
    # =========================================================================
    if not df_train.empty:
        df_base = df_train[df_train["config"] == "baseline"]
        if not df_base.empty:
            plt.figure(figsize=(8, 5))
            for algo in ["ppo", "masked_ppo", "aen_ppo"]:
                sub = df_base[df_base["algorithm"] == algo]
                if sub.empty:
                    continue
                grouped = sub.groupby("episode")["reward"]
                mean_r = grouped.mean()
                sem_r = grouped.sem()
                episodes = mean_r.index

                plt.plot(episodes, mean_r, label=LABEL_MAP[algo], color=COLOR_MAP[algo], lw=2.2)
                plt.fill_between(episodes, mean_r - 1.96 * sem_r, mean_r + 1.96 * sem_r, color=COLOR_MAP[algo], alpha=0.18)

            plt.title("Learning Curves on Baseline Mine Dispatch Environment (95% CI)", fontweight="bold")
            plt.xlabel("Training Episode")
            plt.ylabel("Episodic Return")
            plt.legend(frameon=True, facecolor="white", edgecolor="#bdc3c7")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "01_learning_curves_baseline.png"))
            plt.close()

    # =========================================================================
    # PLOT 2: Mean Reward Comparison Across Algorithms
    # =========================================================================
    base_summary = df_summary[df_summary["config"] == "baseline"]
    if not base_summary.empty:
        plt.figure(figsize=(7, 4.8))
        algos = ["ppo", "masked_ppo", "aen_ppo"]
        sub = base_summary.set_index("algorithm").reindex(algos)
        bars = plt.bar(
            [LABEL_MAP[a] for a in algos],
            sub["mean_return"],
            yerr=sub["ci_95_return"],
            capsize=6,
            color=[COLOR_MAP[a] for a in algos],
            alpha=0.88,
            edgecolor="#2c3e50",
            width=0.55,
        )
        for bar in bars:
            yval = bar.get_height()
            offset = 4 if yval >= 0 else -18
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + offset, f"{yval:.1f}", ha="center", va="bottom" if yval < 0 else "top", fontweight="bold")

        plt.title("Mean Episodic Return Comparison (Baseline Environment)", fontweight="bold")
        plt.ylabel("Mean Return (95% CI)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "02_mean_reward_comparison.png"))
        plt.close()

    # =========================================================================
    # PLOT 3: Completed Deliveries / Throughput Comparison
    # =========================================================================
    if not base_summary.empty:
        plt.figure(figsize=(7, 4.8))
        algos = ["ppo", "masked_ppo", "aen_ppo"]
        sub = base_summary.set_index("algorithm").reindex(algos)
        bars = plt.bar(
            [LABEL_MAP[a] for a in algos],
            sub["mean_deliveries"],
            color=[COLOR_MAP[a] for a in algos],
            alpha=0.88,
            edgecolor="#2c3e50",
            width=0.55,
        )
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.15, f"{yval:.1f} loads", ha="center", va="bottom", fontweight="bold")

        plt.title("Fleet Delivery Throughput Comparison (Baseline Environment)", fontweight="bold")
        plt.ylabel("Completed Deliveries per Episode")
        plt.ylim(0, max(12, sub["mean_deliveries"].max() + 2))
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "03_completed_deliveries.png"))
        plt.close()

    # =========================================================================
    # PLOT 4: Invalid Action Count & Rate Comparison
    # =========================================================================
    if not base_summary.empty:
        fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
        algos = ["ppo", "masked_ppo", "aen_ppo"]
        sub = base_summary.set_index("algorithm").reindex(algos)
        x = np.arange(len(algos))

        bars = ax1.bar(x, sub["mean_invalid_actions"], color=[COLOR_MAP[a] for a in algos], alpha=0.85, width=0.5, edgecolor="#2c3e50")
        ax1.set_ylabel("Invalid Actions Chosen per Episode", color="#2c3e50")
        ax1.set_xticks(x)
        ax1.set_xticklabels([LABEL_MAP[a] for a in algos])

        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f} ({yval/100*100:.1f}%)", ha="center", va="bottom", fontweight="bold")

        plt.title("Invalid Action Rate & Feasibility Enforcement", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "04_invalid_actions_comparison.png"))
        plt.close()

    # =========================================================================
    # PLOT 5: Raw Entropy vs. Normalized Entropy (RQ2)
    # =========================================================================
    if not df_train.empty:
        df_base = df_train[df_train["config"] == "baseline"]
        if not df_base.empty:
            plt.figure(figsize=(8, 5))
            for algo in ["masked_ppo", "aen_ppo"]:
                sub = df_base[df_base["algorithm"] == algo]
                if sub.empty:
                    continue
                # Raw entropy
                grouped_raw = sub.groupby("episode")["mean_raw_entropy"].mean()
                grouped_norm = sub.groupby("episode")["mean_norm_entropy"].mean()
                episodes = grouped_raw.index

                if algo == "masked_ppo":
                    plt.plot(episodes, grouped_raw, label="Masked PPO: Raw Entropy $H(\\pi)$", color=COLOR_MAP["masked_ppo"], lw=2.0)
                else:
                    plt.plot(episodes, grouped_raw, label="AEN-PPO: Raw Entropy $H(\\pi)$", color="#16a085", lw=1.8, linestyle="--")
                    plt.plot(episodes, grouped_norm, label="AEN-PPO: Normalized Entropy $H_{norm}(\\pi)$", color=COLOR_MAP["aen_ppo"], lw=2.4)

            plt.title("Policy Entropy Trajectory: Raw vs. Normalized (Baseline)", fontweight="bold")
            plt.xlabel("Training Episode")
            plt.ylabel("Entropy Value (nats)")
            plt.legend(frameon=True, facecolor="white")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "05_raw_vs_normalized_entropy.png"))
            plt.close()

    # =========================================================================
    # PLOT 6: Valid Action Count Dynamics |A_valid|
    # =========================================================================
    if not df_train.empty:
        df_base = df_train[df_train["config"] == "baseline"]
        if not df_base.empty:
            plt.figure(figsize=(8, 4.8))
            for algo in ["masked_ppo", "aen_ppo"]:
                sub = df_base[df_base["algorithm"] == algo]
                if sub.empty:
                    continue
                grouped = sub.groupby("episode")["mean_valid_actions"].mean()
                plt.plot(grouped.index, grouped, label=LABEL_MAP[algo], color=COLOR_MAP[algo], lw=2.0)

            plt.axhline(y=25, color="#7f8c8d", linestyle=":", label="Total Discrete Action Space (25)")
            plt.title("Dynamic Action Space Feasibility $|A_{valid}(s)|$ Over Training", fontweight="bold")
            plt.xlabel("Episode")
            plt.ylabel("Mean Valid Actions $|A_{valid}|$")
            plt.legend(frameon=True, facecolor="white")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "06_mean_valid_actions_dynamics.png"))
            plt.close()

    # =========================================================================
    # PLOT 7: Scalability Across Truck Counts (2, 4, 6, 8)
    # =========================================================================
    truck_cfgs = ["trucks_2", "trucks_4", "trucks_6", "trucks_8"]
    df_trucks = df_summary[df_summary["config"].isin(truck_cfgs)]
    if not df_trucks.empty:
        plt.figure(figsize=(8.5, 5))
        for algo in ["ppo", "masked_ppo", "aen_ppo"]:
            sub = df_trucks[df_trucks["algorithm"] == algo].sort_values("num_trucks")
            if sub.empty:
                continue
            plt.errorbar(
                sub["num_trucks"],
                sub["mean_return"],
                yerr=sub["ci_95_return"],
                label=LABEL_MAP[algo],
                color=COLOR_MAP[algo],
                marker="o",
                lw=2.2,
                capsize=5,
            )
        plt.title("Environment Scalability: Performance Across Fleet Sizes ($K \\in \\{2, 4, 6, 8\\}$)", fontweight="bold")
        plt.xlabel("Number of Trucks ($K$)")
        plt.ylabel("Mean Return (95% CI)")
        plt.xticks([2, 4, 6, 8])
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "07_scalability_truck_count.png"))
        plt.close()

    # =========================================================================
    # PLOT 8: Scalability Across Pit / Shovel Counts (1, 2, 3, 4)
    # =========================================================================
    pit_cfgs = ["pits_1", "pits_2", "pits_3", "pits_4"]
    df_pits = df_summary[df_summary["config"].isin(pit_cfgs)].copy()
    if not df_pits.empty:
        df_pits["num_pits"] = df_pits["config"].apply(lambda c: int(c.replace("pits_", "")))
        plt.figure(figsize=(8.5, 5))
        for algo in ["ppo", "masked_ppo", "aen_ppo"]:
            sub = df_pits[df_pits["algorithm"] == algo].sort_values("num_pits")
            if sub.empty:
                continue
            plt.errorbar(
                sub["num_pits"],
                sub["mean_return"],
                yerr=sub["ci_95_return"],
                label=LABEL_MAP[algo],
                color=COLOR_MAP[algo],
                marker="s",
                lw=2.2,
                capsize=5,
            )
        plt.title("Environment Scalability: Performance Across Loading Pits ($P \\in \\{1, 2, 3, 4\\}$)", fontweight="bold")
        plt.xlabel("Number of Loading Pits / Shovels ($P$)")
        plt.ylabel("Mean Return (95% CI)")
        plt.xticks([1, 2, 3, 4])
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "08_scalability_pit_count.png"))
        plt.close()

    # =========================================================================
    # PLOT 9: Scalability Across Job Workloads (4, 6, 8, 12)
    # =========================================================================
    job_cfgs = ["jobs_4", "jobs_6", "jobs_8", "jobs_12"]
    df_jobs = df_summary[df_summary["config"].isin(job_cfgs)]
    if not df_jobs.empty:
        plt.figure(figsize=(8.5, 5))
        for algo in ["ppo", "masked_ppo", "aen_ppo"]:
            sub = df_jobs[df_jobs["algorithm"] == algo].sort_values("num_jobs")
            if sub.empty:
                continue
            plt.errorbar(
                sub["num_jobs"],
                sub["mean_return"],
                yerr=sub["ci_95_return"],
                label=LABEL_MAP[algo],
                color=COLOR_MAP[algo],
                marker="^",
                lw=2.2,
                capsize=5,
            )
        plt.title("Environment Scalability: Performance Across Job Workloads ($N_{jobs} \\in \\{4, 6, 8, 12\\}$)", fontweight="bold")
        plt.xlabel("Number of Job Slots ($N_{jobs}$)")
        plt.ylabel("Mean Return (95% CI)")
        plt.xticks([4, 6, 8, 12])
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "09_scalability_job_count.png"))
        plt.close()

    # =========================================================================
    # PLOT 10: Multi-Seed Stability & Convergence Curves
    # =========================================================================
    if not df_train.empty:
        df_base = df_train[df_train["config"] == "baseline"]
        if not df_base.empty:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
            for idx, algo in enumerate(["ppo", "masked_ppo", "aen_ppo"]):
                ax = axes[idx]
                sub = df_base[df_base["algorithm"] == algo]
                seeds = sub["seed"].unique()
                for s in seeds:
                    s_data = sub[sub["seed"] == s]
                    ax.plot(s_data["episode"], s_data["reward"], alpha=0.35, color=COLOR_MAP[algo], lw=1.2)

                grouped = sub.groupby("episode")["reward"].mean()
                ax.plot(grouped.index, grouped, color=COLOR_MAP[algo], lw=2.6, label=f"Mean (5 Seeds)")
                ax.set_title(LABEL_MAP[algo], fontweight="bold")
                ax.set_xlabel("Episode")
                if idx == 0:
                    ax.set_ylabel("Episodic Return")
                ax.legend(frameon=True, facecolor="white")

            fig.suptitle("Multi-Seed Training Stability Across 5 Independent Seeds", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "10_multi_seed_learning_curves.png"))
            plt.close()

    # =========================================================================
    # PLOT 11: Generalization Gap Comparison (Seen vs. Unseen Map)
    # =========================================================================
    if "generalization_gap" in base_summary.columns:
        plt.figure(figsize=(7, 4.8))
        algos = ["ppo", "masked_ppo", "aen_ppo"]
        sub = base_summary.set_index("algorithm").reindex(algos)
        bars = plt.bar(
            [LABEL_MAP[a] for a in algos],
            sub["generalization_gap"],
            color=[COLOR_MAP[a] for a in algos],
            alpha=0.88,
            width=0.5,
            edgecolor="#2c3e50",
        )
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"\\Delta: {yval:.1f}", ha="center", va="bottom", fontweight="bold")

        plt.title("Generalization Gap $\\Delta_{gen} = J_{seen} - J_{unseen}$ (Evaluation Map)", fontweight="bold")
        plt.ylabel("Performance Drop $\\Delta_{gen}$")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "11_generalization_gap.png"))
        plt.close()

    # =========================================================================
    # PLOT 12: Computational Efficiency & Latency Trade-off
    # =========================================================================
    if "inference_latency_ms" in base_summary.columns:
        plt.figure(figsize=(7.5, 4.8))
        algos = ["ppo", "masked_ppo", "aen_ppo"]
        sub = base_summary.set_index("algorithm").reindex(algos)
        for a in algos:
            plt.scatter(
                sub.loc[a, "inference_latency_ms"],
                sub.loc[a, "mean_return"],
                s=sub.loc[a, "parameter_count"] / 40.0,
                color=COLOR_MAP[a],
                label=f"{LABEL_MAP[a]} ({sub.loc[a, 'inference_latency_ms']:.2f} ms)",
                edgecolor="#2c3e50",
                alpha=0.85,
            )

        plt.title("Computational Efficiency vs. Decision Quality", fontweight="bold")
        plt.xlabel("Decision Inference Latency (ms / step)")
        plt.ylabel("Mean Episodic Return")
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "12_computational_efficiency.png"))
        plt.close()

    # =========================================================================
    # PLOT 13: Operational Radar / Summary Chart Across PPT Dimensions
    # =========================================================================
    if not base_summary.empty:
        categories = ["Throughput", "Safety (100-Invalid%)", "Fleet Util (%)", "Success Rate", "Sample Eff"]
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for algo in ["ppo", "masked_ppo", "aen_ppo"]:
            row = base_summary[base_summary["algorithm"] == algo].iloc[0]
            # Normalize to 0-100 scale
            vals = [
                min(100.0, row["mean_deliveries"] * 10.0),
                max(0.0, 100.0 - row["invalid_action_rate_pct"]),
                row["fleet_utilization_pct"],
                row["success_rate_pct"],
                min(100.0, max(0.0, row["return_growth_per_1k_steps"] * 20.0 + 50.0)),
            ]
            vals += vals[:1]
            ax.plot(angles, vals, color=COLOR_MAP[algo], linewidth=2.2, label=LABEL_MAP[algo])
            ax.fill(angles, vals, color=COLOR_MAP[algo], alpha=0.15)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 100)
        plt.title("Multi-Objective Mine Dispatching Performance Radar", y=1.08, fontweight="bold")
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, facecolor="white")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "13_multiobjective_radar.png"))
        plt.close()

    # =========================================================================
    # GENERATE COMPREHENSIVE RESEARCH TABLES
    # =========================================================================
    # Table 1: Comprehensive PPT Evaluation Metrics Table
    ppt_cols = [
        "config", "algorithm", "mean_return", "ci_95_return", "success_rate_pct",
        "cumulative_regret", "convergence_episodes", "return_growth_per_1k_steps",
        "seed_std", "generalization_gap", "mean_deliveries", "fleet_utilization_pct",
        "avg_waiting_time", "invalid_action_rate_pct", "inference_latency_ms"
    ]
    df_ppt_table = df_summary[[c for c in ppt_cols if c in df_summary.columns]].copy()
    df_ppt_table.to_csv(os.path.join(tables_dir, "comprehensive_ppt_metrics.csv"), index=False)

    # Markdown format
    with open(os.path.join(tables_dir, "comprehensive_ppt_metrics.md"), "w") as f:
        f.write("# Comprehensive Evaluation Metrics Table (Mandatory PPT Metrics)\n\n")
        f.write(df_ppt_table.to_markdown(index=False))

    # Table 2: Scalability Table
    scalability_cols = [
        "config", "category", "algorithm", "num_trucks", "num_jobs",
        "mean_return", "ci_95_return", "mean_deliveries", "fleet_utilization_pct",
        "avg_waiting_time", "invalid_action_rate_pct"
    ]
    df_scalability_table = df_summary[[c for c in scalability_cols if c in df_summary.columns]].copy()
    df_scalability_table.to_csv(os.path.join(tables_dir, "scalability_comparison.csv"), index=False)
    with open(os.path.join(tables_dir, "scalability_comparison.md"), "w") as f:
        f.write("# Environment Scalability Comparison Table\n\n")
        f.write(df_scalability_table.to_markdown(index=False))

    # Table 3: Statistical Significance Table
    if not df_stat.empty:
        df_stat.to_csv(os.path.join(tables_dir, "statistical_significance.csv"), index=False)
        with open(os.path.join(tables_dir, "statistical_significance.md"), "w") as f:
            f.write("# Statistical Significance Analysis Table (Paired Tests & Effect Sizes)\n\n")
            f.write(df_stat.to_markdown(index=False))

    print("All 13+ Publication Plots and Research Tables generated successfully!")
    print(f"Plots directory:  {plots_dir}")
    print(f"Tables directory: {tables_dir}")


def main():
    generate_all_plots(results_root="results")


if __name__ == "__main__":
    main()
