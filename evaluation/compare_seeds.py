"""
Multi-Seed Aggregation and Publication Figure Generator.
Computes mean learning curves, variance/confidence bands, and summary tables across random seeds.
Usage:
    python evaluation/compare_seeds.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def smooth_curve(scalars: np.ndarray, weight: float = 0.85) -> np.ndarray:
    """Exponential moving average for clean research curves."""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def load_runs_for_method(method_dir_pattern: str):
    reward_runs = []
    invalid_runs = []
    paths = glob.glob(os.path.join(PROJECT_ROOT, "results", method_dir_pattern))

    for p in paths:
        r_file = os.path.join(p, "episode_rewards.npy")
        inv_file = os.path.join(p, "invalid_actions.npy")
        if os.path.exists(r_file):
            reward_runs.append(np.load(r_file))
        if os.path.exists(inv_file):
            invalid_runs.append(np.load(inv_file))

    return reward_runs, invalid_runs


def generate_comparison_plots():
    out_dir = os.path.join(PROJECT_ROOT, "results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_plot = os.path.join(out_dir, "multiseed_learning_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    methods = {
        "PPO (Unmasked)": ("ppo*", "#e74c3c"),
        "Masked PPO": ("masked_ppo*", "#3498db"),
        "AEN-PPO (Proposed)": ("aen_ppo*", "#00b894"),
    }

    print("\n" + "=" * 65)
    print("AGGREGATING MULTI-SEED EXPERIMENT RUNS")
    print("=" * 65)

    for name, (pat, col) in methods.items():
        r_runs, inv_runs = load_runs_for_method(pat)
        if not r_runs:
            print(f"Warning: No runs found for {name} with pattern {pat}")
            continue

        min_len = min(len(r) for r in r_runs)
        r_matrix = np.array([r[:min_len] for r in r_runs])
        mean_r = np.mean(r_matrix, axis=0)
        std_r = np.std(r_matrix, axis=0)

        episodes = np.arange(1, min_len + 1)
        smoothed_mean = smooth_curve(mean_r, weight=0.8)

        # Plot Reward Curve with shaded std dev band
        axes[0].plot(episodes, smoothed_mean, label=name, color=col, linewidth=2.2)
        axes[0].fill_between(
            episodes,
            smooth_curve(mean_r - std_r, 0.8),
            smooth_curve(mean_r + std_r, 0.8),
            color=col,
            alpha=0.18,
        )

        # Plot Invalid Actions
        if inv_runs:
            min_inv_len = min(len(inv) for inv in inv_runs)
            inv_matrix = np.array([inv[:min_inv_len] for inv in inv_runs])
            mean_inv = np.mean(inv_matrix, axis=0)
            inv_episodes = np.arange(1, min_inv_len + 1)
            axes[1].plot(inv_episodes, smooth_curve(mean_inv, 0.8), label=name, color=col, linewidth=2.2)

        print(f"{name:20s} | Seeds: {len(r_runs)} | Final Mean Reward: {np.mean(mean_r[-10:]):.2f}")

    # Configure axes
    axes[0].set_title("Training Reward Curve (Mean +/- 1 Std)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Episode", fontsize=11)
    axes[0].set_ylabel("Episode Reward", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Invalid Action Rate / Infeasibility", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].set_ylabel("Invalid Actions Selected", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Reinforcement Learning in Dynamic Action Spaces: Benchmark", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_plot, bbox_inches="tight")
    plt.close()

    print(f"\nMulti-seed learning curves saved to: {out_plot}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    generate_comparison_plots()
