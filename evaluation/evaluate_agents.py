"""
Comparative Evaluation and Benchmarking Suite.
Runs Standard PPO, Masked PPO, and AEN-PPO under identical evaluation seeds.
Generates research metrics tables and publication-grade comparison plots.
Usage:
    python evaluation/evaluate_agents.py --episodes 20 --seed 999
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environments.grid_map import GridMap
from environments.visual_truck_dispatch_env import VisualTruckDispatchEnv
from run_visual_simulation import load_agent, select_policy_action


def evaluate_agent_policy(agent_name: str, env: VisualTruckDispatchEnv, num_episodes: int = 20, seed: int = 999):
    agent = load_agent(agent_name)
    rewards = []
    completed = []
    invalids = []
    valid_counts = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_invalid = 0
        ep_valid_action_lens = []

        done = False
        while not done:
            mask = env.action_masks()
            ep_valid_action_lens.append(int(np.sum(mask)))

            action = select_policy_action(agent, agent_name, obs, mask, env)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            ep_reward += float(reward)
            if not step_info.get("valid_action", True):
                ep_invalid += 1

            obs = next_obs

        rewards.append(ep_reward)
        completed.append(env.completed_jobs)
        invalids.append(ep_invalid)
        valid_counts.append(float(np.mean(ep_valid_action_lens)))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_completed": float(np.mean(completed)),
        "std_completed": float(np.std(completed)),
        "mean_invalids": float(np.mean(invalids)),
        "std_invalids": float(np.std(invalids)),
        "mean_valid_actions": float(np.mean(valid_counts)),
        "rewards_raw": rewards,
        "completed_raw": completed,
        "invalids_raw": invalids,
    }


def plot_comparative_results(results: dict, output_path: str):
    """
    Renders high-resolution scientific comparative charts.
    """
    agents = list(results.keys())
    agent_labels = [a.replace("_", "-").upper() for a in agents]

    mean_rewards = [results[a]["mean_reward"] for a in agents]
    std_rewards = [results[a]["std_reward"] for a in agents]

    mean_completed = [results[a]["mean_completed"] for a in agents]
    std_completed = [results[a]["std_completed"] for a in agents]

    mean_invalids = [results[a]["mean_invalids"] for a in agents]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    colors = ["#e74c3c", "#3498db", "#00b894"]

    # 1. Episode Reward
    axes[0].bar(agent_labels, mean_rewards, yerr=std_rewards, capsize=5, color=colors, alpha=0.85, edgecolor="#2d3436")
    axes[0].set_title("Mean Episode Reward", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Reward")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Completed Jobs / Deliveries
    axes[1].bar(agent_labels, mean_completed, yerr=std_completed, capsize=5, color=colors, alpha=0.85, edgecolor="#2d3436")
    axes[1].set_title("Completed Job Deliveries", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Jobs Delivered")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Invalid Actions Count
    axes[2].bar(agent_labels, mean_invalids, color=colors, alpha=0.85, edgecolor="#2d3436")
    axes[2].set_title("Invalid Action Count (Feasibility)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Violations per Episode")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Research Benchmark: Standard PPO vs Masked PPO vs AEN-PPO", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved comparative figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Dispatch Agents")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes per agent")
    parser.add_argument("--seed", type=int, default=999, help="Held-out evaluation seed")
    parser.add_argument("--map", type=str, default="maps/starter_map.json", help="Map path")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    grid_map = GridMap.load_from_json(args.map) if os.path.exists(args.map) else GridMap.create_default_map()
    env = VisualTruckDispatchEnv(grid_map=grid_map, max_steps=100)

    print("\n" + "=" * 70)
    print("STARTING HELD-OUT COMPARATIVE BENCHMARK EVALUATION")
    print(f"Episodes: {args.episodes} | Seed: {args.seed} | Map: {args.map}")
    print("=" * 70)

    methods = ["ppo", "masked_ppo", "aen_ppo"]
    benchmark_results = {}

    for method in methods:
        print(f"\n--> Evaluating {method.upper()}...")
        res = evaluate_agent_policy(method, env, num_episodes=args.episodes, seed=args.seed)
        benchmark_results[method] = res
        print(f"    Mean Reward:          {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}")
        print(f"    Completed Deliveries: {res['mean_completed']:.2f} +/- {res['std_completed']:.2f}")
        print(f"    Invalid Actions:      {res['mean_invalids']:.2f}")
        print(f"    Mean |A_valid|:       {res['mean_valid_actions']:.2f}")

    # Save JSON summary
    summary_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump({m: {k: v for k, v in r.items() if not k.endswith("_raw")} for m, r in benchmark_results.items()}, f, indent=2)

    # Plot Figure
    plot_path = os.path.join(out_dir, "benchmark_comparison.png")
    plot_comparative_results(benchmark_results, plot_path)

    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION COMPLETE")
    print(f"Summary JSON:   {summary_path}")
    print(f"Comparison Plot: {plot_path}")
    print("=" * 70 + "\n")

    env.close()


if __name__ == "__main__":
    main()
