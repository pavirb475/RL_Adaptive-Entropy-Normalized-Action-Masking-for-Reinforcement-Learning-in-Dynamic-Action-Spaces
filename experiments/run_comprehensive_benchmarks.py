"""
Comprehensive Multi-Seed & Scalability Experiment Runner.
Executes Phase 2, 3, 4, 5, and 6:
- Compares Standard PPO, Masked PPO, and Proposed AEN-PPO.
- Evaluates 5 independent training seeds: [11, 22, 33, 44, 55].
- Evaluates Scalability across Trucks (2, 4, 6, 8), Pits (1, 2, 3, 4), Jobs (4, 6, 8, 12).
- Profiles Computational Efficiency (inference latency, training time, parameters).
- Evaluates Generalization on unseen held-out mine maps.
- Computes every mandatory metric from PPTperformance.pdf.
- Stores raw CSVs, aggregated results, and paired statistical significance tests.
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.distributions.categorical import Categorical

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environments.grid_map import GridMap
from environments.visual_truck_dispatch_env import VisualTruckDispatchEnv
from agents.ppo import PPOAgent
from agents.masked_ppo import MaskedPPOAgent
from agents.aen_ppo import AENPPOAgent
from evaluation.metrics import (
    compute_return_statistics,
    compute_success_rate,
    compute_cumulative_regret,
    compute_convergence_speed,
    compute_sample_efficiency,
    compute_generalization_gap,
    compute_stability_across_seeds,
    compute_paired_significance,
    compute_mine_dispatch_metrics,
    compute_action_exploration_metrics,
)


CONFIG_DEFINITIONS = {
    # 1. Baseline
    "baseline": {
        "num_trucks": 4,
        "num_jobs": 6,
        "map_path": "maps/pit_2_map.json",
        "category": "baseline",
    },
    # 2. Truck Fleet Variations (RQ1)
    "trucks_2": {
        "num_trucks": 2,
        "num_jobs": 6,
        "map_path": "maps/pit_2_map.json",
        "category": "truck_variations",
    },
    "trucks_4": {
        "num_trucks": 4,
        "num_jobs": 6,
        "map_path": "maps/pit_2_map.json",
        "category": "truck_variations",
    },
    "trucks_6": {
        "num_trucks": 6,
        "num_jobs": 6,
        "map_path": "maps/pit_2_map.json",
        "category": "truck_variations",
    },
    "trucks_8": {
        "num_trucks": 8,
        "num_jobs": 6,
        "map_path": "maps/pit_2_map.json",
        "category": "truck_variations",
    },
    # 3. Pit / Shovel Variations (RQ1)
    "pits_1": {
        "num_trucks": 4,
        "num_jobs": 6,
        "map_path": "maps/pit_1_map.json",
        "category": "pit_variations",
    },
    "pits_2": {
        "num_trucks": 4,
        "num_jobs": 6,
        "map_path": "maps/pit_2_map.json",
        "category": "pit_variations",
    },
    "pits_3": {
        "num_trucks": 4,
        "num_jobs": 6,
        "map_path": "maps/pit_3_map.json",
        "category": "pit_variations",
    },
    "pits_4": {
        "num_trucks": 4,
        "num_jobs": 6,
        "map_path": "maps/pit_4_map.json",
        "category": "pit_variations",
    },
    # 4. Job Workload Variations (RQ1)
    "jobs_4": {
        "num_trucks": 4,
        "num_jobs": 4,
        "map_path": "maps/pit_2_map.json",
        "category": "job_variations",
    },
    "jobs_6": {
        "num_trucks": 4,
        "num_jobs": 6,
        "map_path": "maps/pit_2_map.json",
        "category": "job_variations",
    },
    "jobs_8": {
        "num_trucks": 4,
        "num_jobs": 8,
        "map_path": "maps/pit_2_map.json",
        "category": "job_variations",
    },
    "jobs_12": {
        "num_trucks": 4,
        "num_jobs": 12,
        "map_path": "maps/pit_2_map.json",
        "category": "job_variations",
    },
    # 5. Combined Configurations
    "combined_low": {
        "num_trucks": 2,
        "num_jobs": 4,
        "map_path": "maps/pit_1_map.json",
        "category": "combined_configurations",
    },
    "combined_high": {
        "num_trucks": 8,
        "num_jobs": 12,
        "map_path": "maps/pit_4_map.json",
        "category": "combined_configurations",
    },
}


def create_agent(agent_name: str, obs_dim: int, action_dim: int, device: str = "cpu"):
    if agent_name == "aen_ppo":
        return AENPPOAgent(obs_dim, action_dim, device=device)
    elif agent_name == "masked_ppo":
        return MaskedPPOAgent(obs_dim, action_dim, device=device)
    elif agent_name == "ppo":
        return PPOAgent(obs_dim, action_dim, device=device)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def count_parameters(agent) -> int:
    return sum(p.numel() for p in agent.network.parameters() if p.requires_grad)


def train_single_run(
    agent_name: str,
    config_name: str,
    config: Dict[str, Any],
    seed: int,
    num_episodes: int,
    device: str = "cpu",
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, float]]:
    """
    Trains one agent instance on a specific configuration and seed.
    Returns (trained_agent, training_trajectory_records, compute_telemetry).
    """
    # Deterministic seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    map_path = os.path.join(PROJECT_ROOT, config["map_path"])
    grid_map = GridMap.load_from_json(map_path) if os.path.exists(map_path) else GridMap.create_default_map()

    env = VisualTruckDispatchEnv(
        grid_map=grid_map,
        num_trucks=config["num_trucks"],
        num_jobs=config["num_jobs"],
        max_steps=100,
        seed=seed,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = create_agent(agent_name, obs_dim, action_dim, device=device)

    rollout_steps = 512
    buffer = {
        "obs": [], "actions": [], "log_probs": [], "rewards": [],
        "values": [], "dones": [], "terminated": [], "next_values": [], "masks": []
    }

    trajectory_records = []
    inference_latencies = []

    t_train_start = time.perf_counter()
    total_env_steps = 0
    episodes_completed = 0

    while episodes_completed < num_episodes:
        obs, info = env.reset(seed=seed + episodes_completed * 1000)
        ep_reward = 0.0
        ep_len = 0
        ep_invalids = 0
        ep_valid_lens = []
        ep_raw_entropies = []
        ep_norm_entropies = []

        done = False
        while not done:
            mask = env.action_masks()
            n_valid = int(np.sum(mask))
            ep_valid_lens.append(n_valid)

            # Profile inference latency
            t_inf_0 = time.perf_counter()
            with torch.no_grad():
                ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                feat = agent.network.shared(ot)
                logits = agent.network.actor(feat)
                if agent_name in ["aen_ppo", "masked_ppo"]:
                    mt = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    masked_logits = logits.masked_fill(~mt, -1e9)
                    dist = Categorical(logits=masked_logits)
                else:
                    dist = Categorical(logits=logits)

                raw_h = float(dist.entropy().item())
                norm_h = (raw_h / np.log(max(2, n_valid))) if n_valid > 1 else 0.0

            if agent_name in ["aen_ppo", "masked_ppo"]:
                action, log_prob, value = agent.select_action(obs, mask)
            else:
                action, log_prob, value = agent.select_action(obs)

            t_inf_1 = time.perf_counter()
            inference_latencies.append((t_inf_1 - t_inf_0) * 1000.0)  # ms

            ep_raw_entropies.append(raw_h)
            ep_norm_entropies.append(norm_h)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            next_val = 0.0 if terminated else float(agent.network.critic(agent.network.shared(
                torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            )).squeeze(-1).item())

            buffer["obs"].append(np.asarray(obs, dtype=np.float32).copy())
            buffer["actions"].append(int(action))
            buffer["log_probs"].append(float(log_prob))
            buffer["rewards"].append(float(reward))
            buffer["values"].append(float(value))
            buffer["dones"].append(float(done))
            buffer["terminated"].append(float(terminated))
            buffer["next_values"].append(float(next_val))
            buffer["masks"].append(mask.copy())

            ep_reward += float(reward)
            ep_len += 1
            total_env_steps += 1
            if not step_info.get("valid_action", True):
                ep_invalids += 1

            obs = next_obs

            # PPO Batch Update
            if len(buffer["obs"]) >= rollout_steps or (done and episodes_completed >= num_episodes - 1 and len(buffer["obs"]) > 0):
                adv, ret = agent.compute_gae(
                    rewards=np.asarray(buffer["rewards"], dtype=np.float32),
                    values=np.asarray(buffer["values"], dtype=np.float32),
                    dones=np.asarray(buffer["dones"], dtype=np.float32),
                    next_value=0.0,
                    terminated=np.asarray(buffer["terminated"], dtype=np.float32),
                    next_values=np.asarray(buffer["next_values"], dtype=np.float32),
                )
                if agent_name in ["aen_ppo", "masked_ppo"]:
                    agent.update(
                        observations=np.asarray(buffer["obs"], dtype=np.float32),
                        actions=np.asarray(buffer["actions"], dtype=np.int64),
                        old_log_probabilities=np.asarray(buffer["log_probs"], dtype=np.float32),
                        advantages=adv,
                        returns=ret,
                        action_masks=np.asarray(buffer["masks"], dtype=bool),
                    )
                else:
                    agent.update(
                        observations=np.asarray(buffer["obs"], dtype=np.float32),
                        actions=np.asarray(buffer["actions"], dtype=np.int64),
                        old_log_probabilities=np.asarray(buffer["log_probs"], dtype=np.float32),
                        advantages=adv,
                        returns=ret,
                    )
                buffer = {k: [] for k in buffer}

        episodes_completed += 1
        trajectory_records.append({
            "episode": episodes_completed,
            "reward": float(ep_reward),
            "length": ep_len,
            "invalid_actions": ep_invalids,
            "completed_jobs": env.completed_jobs,
            "expired_jobs": env.total_expired_jobs,
            "fleet_utilization": float(env.total_truck_busy_steps / max(1, env.total_truck_possible_steps)),
            "avg_waiting_time": float(np.mean(env.job_waiting_times)) if env.job_waiting_times else 0.0,
            "avg_completion_time": float(np.mean(env.job_completion_times)) if env.job_completion_times else 0.0,
            "total_travel_distance": int(sum(t.total_distance_traveled for t in env.trucks)),
            "mean_valid_actions": float(np.mean(ep_valid_lens)),
            "mean_raw_entropy": float(np.mean(ep_raw_entropies)),
            "mean_norm_entropy": float(np.mean(ep_norm_entropies)),
        })

    t_train_end = time.perf_counter()
    train_duration = t_train_end - t_train_start
    env.close()

    compute_telemetry = {
        "training_time_sec": float(train_duration),
        "steps_per_sec": float(total_env_steps / max(0.001, train_duration)),
        "time_per_10k_steps_sec": float((train_duration / max(1, total_env_steps)) * 10000.0),
        "mean_inference_latency_ms": float(np.mean(inference_latencies)) if inference_latencies else 0.0,
        "parameter_count": count_parameters(agent),
    }

    return agent, trajectory_records, compute_telemetry


def evaluate_agent_on_scenarios(
    agent,
    agent_name: str,
    config: Dict[str, Any],
    eval_seeds: List[int],
    map_override: Optional[str] = None,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Evaluates a trained model on held-out evaluation scenarios under deterministic seeds.
    """
    map_file = map_override if map_override is not None else config["map_path"]
    full_map_path = os.path.join(PROJECT_ROOT, map_file)
    grid_map = GridMap.load_from_json(full_map_path) if os.path.exists(full_map_path) else GridMap.create_default_map()

    env = VisualTruckDispatchEnv(
        grid_map=grid_map,
        num_trucks=config["num_trucks"],
        num_jobs=config["num_jobs"],
        max_steps=100,
    )

    eval_results = []

    for ep_idx, eseed in enumerate(eval_seeds):
        obs, info = env.reset(seed=eseed)
        ep_reward = 0.0
        ep_len = 0
        ep_invalids = 0
        ep_valid_lens = []
        ep_raw_entropies = []
        ep_norm_entropies = []

        done = False
        while not done:
            mask = env.action_masks()
            n_valid = int(np.sum(mask))
            ep_valid_lens.append(n_valid)

            with torch.no_grad():
                ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                feat = agent.network.shared(ot)
                logits = agent.network.actor(feat)
                if agent_name in ["aen_ppo", "masked_ppo"]:
                    mt = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    dist = Categorical(logits=logits.masked_fill(~mt, -1e9))
                else:
                    dist = Categorical(logits=logits)
                raw_h = float(dist.entropy().item())
                norm_h = (raw_h / np.log(max(2, n_valid))) if n_valid > 1 else 0.0

            ep_raw_entropies.append(raw_h)
            ep_norm_entropies.append(norm_h)

            if agent_name in ["aen_ppo", "masked_ppo"]:
                action, _, _ = agent.select_action(obs, mask)
            else:
                action, _, _ = agent.select_action(obs)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            ep_reward += float(reward)
            ep_len += 1
            if not step_info.get("valid_action", True):
                ep_invalids += 1

            obs = next_obs

        eval_results.append({
            "eval_episode": ep_idx + 1,
            "eval_seed": eseed,
            "reward": float(ep_reward),
            "length": ep_len,
            "invalid_actions": ep_invalids,
            "completed_jobs": env.completed_jobs,
            "expired_jobs": env.total_expired_jobs,
            "fleet_utilization": float(env.total_truck_busy_steps / max(1, env.total_truck_possible_steps)),
            "avg_waiting_time": float(np.mean(env.job_waiting_times)) if env.job_waiting_times else 0.0,
            "avg_completion_time": float(np.mean(env.job_completion_times)) if env.job_completion_times else 0.0,
            "total_travel_distance": int(sum(t.total_distance_traveled for t in env.trucks)),
            "mean_valid_actions": float(np.mean(ep_valid_lens)),
            "mean_raw_entropy": float(np.mean(ep_raw_entropies)),
            "mean_norm_entropy": float(np.mean(ep_norm_entropies)),
        })

    env.close()
    return eval_results


def run_benchmarks(
    selected_configs: Optional[List[str]] = None,
    training_seeds: List[int] = [11, 22, 33, 44, 55],
    eval_seeds: List[int] = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010],
    num_episodes: int = 60,
    device: str = "cpu",
):
    """
    Main benchmark execution loop.
    """
    configs_to_run = selected_configs if selected_configs else list(CONFIG_DEFINITIONS.keys())
    algorithms = ["ppo", "masked_ppo", "aen_ppo"]

    # Prepare directories
    raw_dir = os.path.join(PROJECT_ROOT, "results", "evaluation", "raw_results")
    agg_dir = os.path.join(PROJECT_ROOT, "results", "evaluation", "aggregated_results")
    stat_dir = os.path.join(PROJECT_ROOT, "results", "evaluation", "statistical_tests")
    multi_seed_dir = os.path.join(PROJECT_ROOT, "results", "multi_seed")
    scalability_dir = os.path.join(PROJECT_ROOT, "results", "scalability")

    for d in [raw_dir, agg_dir, stat_dir, multi_seed_dir, scalability_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 80)
    print("AEN-PPO COMPREHENSIVE EXPERIMENTAL SUITE")
    print(f"Configurations ({len(configs_to_run)}): {configs_to_run}")
    print(f"Algorithms (3):              {algorithms}")
    print(f"Training Seeds ({len(training_seeds)}):       {training_seeds}")
    print(f"Evaluation Seeds ({len(eval_seeds)}):     {eval_seeds}")
    print(f"Episodes per run:            {num_episodes}")
    print(f"Device:                      {device}")
    print("=" * 80)

    # Master tracking dicts
    all_eval_data: Dict[str, Dict[str, Dict[int, List[Dict[str, Any]]]]] = {}
    all_train_data: Dict[str, Dict[str, Dict[int, List[Dict[str, Any]]]]] = {}
    compute_profiles: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    total_runs = len(configs_to_run) * len(algorithms) * len(training_seeds)
    current_run = 0

    for cfg_name in configs_to_run:
        cfg = CONFIG_DEFINITIONS[cfg_name]
        all_eval_data[cfg_name] = {algo: {} for algo in algorithms}
        all_train_data[cfg_name] = {algo: {} for algo in algorithms}
        compute_profiles[cfg_name] = {algo: [] for algo in algorithms}

        # Destination subdirectory for checkpoints
        cat_dir = os.path.join(scalability_dir, cfg["category"])
        os.makedirs(cat_dir, exist_ok=True)

        for algo in algorithms:
            algo_multi_dir = os.path.join(multi_seed_dir, algo)
            os.makedirs(algo_multi_dir, exist_ok=True)

            for seed in training_seeds:
                current_run += 1
                progress_pct = (current_run / total_runs) * 100.0
                print(f"[{progress_pct:5.1f}% | Run {current_run:3d}/{total_runs:3d}] "
                      f"Config: {cfg_name:14s} | Algo: {algo:10s} | Seed: {seed:3d} ... ", end="", flush=True)

                t0 = time.perf_counter()
                trained_agent, train_logs, comp_telemetry = train_single_run(
                    agent_name=algo,
                    config_name=cfg_name,
                    config=cfg,
                    seed=seed,
                    num_episodes=num_episodes,
                    device=device,
                )
                t1 = time.perf_counter()

                # Evaluate on held-out seeds
                eval_logs = evaluate_agent_on_scenarios(
                    agent=trained_agent,
                    agent_name=algo,
                    config=cfg,
                    eval_seeds=eval_seeds,
                    device=device,
                )

                all_train_data[cfg_name][algo][seed] = train_logs
                all_eval_data[cfg_name][algo][seed] = eval_logs
                compute_profiles[cfg_name][algo].append(comp_telemetry)

                # Save model checkpoints
                seed_ckpt_dir = os.path.join(algo_multi_dir, f"seed_{seed}")
                os.makedirs(seed_ckpt_dir, exist_ok=True)
                torch.save(trained_agent.network.state_dict(), os.path.join(seed_ckpt_dir, f"{cfg_name}.pt"))
                torch.save(trained_agent.network.state_dict(), os.path.join(cat_dir, f"{cfg_name}_{algo}_seed_{seed}.pt"))

                # Save raw CSVs
                df_train = pd.DataFrame(train_logs)
                df_train.to_csv(os.path.join(raw_dir, f"train_{cfg_name}_{algo}_seed_{seed}.csv"), index=False)

                df_eval = pd.DataFrame(eval_logs)
                df_eval.to_csv(os.path.join(raw_dir, f"eval_{cfg_name}_{algo}_seed_{seed}.csv"), index=False)

                mean_eval_rew = float(np.mean([x["reward"] for x in eval_logs]))
                mean_eval_jobs = float(np.mean([x["completed_jobs"] for x in eval_logs]))
                print(f"Done in {t1-t0:4.1f}s | Eval Reward: {mean_eval_rew:6.2f} | Jobs: {mean_eval_jobs:4.1f}")

    # -------------------------------------------------------------------------
    # GENERALIZATION TEST: Evaluate baseline models on unseen evaluation map
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING GENERALIZATION BENCHMARK ON UNSEEN HELD-OUT MINE MAP")
    print("=" * 80)
    generalization_results = {algo: [] for algo in algorithms}
    base_cfg = CONFIG_DEFINITIONS["baseline"]

    for algo in algorithms:
        for seed in training_seeds:
            # Recreate agent & load weights
            obs_dim = base_cfg["num_trucks"] * 3 + base_cfg["num_jobs"] * 4
            act_dim = base_cfg["num_trucks"] * base_cfg["num_jobs"] + 1
            agent = create_agent(algo, obs_dim, act_dim, device=device)
            ckpt_path = os.path.join(multi_seed_dir, algo, f"seed_{seed}", "baseline.pt")
            if os.path.exists(ckpt_path):
                agent.network.load_state_dict(torch.load(ckpt_path, map_location=device))

            unseen_eval = evaluate_agent_on_scenarios(
                agent=agent,
                agent_name=algo,
                config=base_cfg,
                eval_seeds=eval_seeds,
                map_override="maps/evaluation_map.json",
                device=device,
            )
            generalization_results[algo].extend([x["reward"] for x in unseen_eval])

    # -------------------------------------------------------------------------
    # AGGREGATE EVALUATION METRICS ACROSS SEEDS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTING PPT EVALUATION METRICS & COMPILING AGGREGATED SUMMARY")
    print("=" * 80)

    summary_records = []
    oracle_benchmarks: Dict[str, float] = {}

    # First pass: find oracle benchmark per config (max mean return among algos)
    for cfg_name in configs_to_run:
        max_r = -1e9
        for algo in algorithms:
            all_r = []
            for seed in training_seeds:
                all_r.extend([x["reward"] for x in all_eval_data[cfg_name][algo][seed]])
            if all_r and np.mean(all_r) > max_r:
                max_r = float(np.mean(all_r))
        oracle_benchmarks[cfg_name] = max_r

    for cfg_name in configs_to_run:
        for algo in algorithms:
            # Flatten evaluation episodes across all 5 seeds
            all_rewards = []
            all_completed = []
            all_expired = []
            all_invalids = []
            all_steps = []
            all_waiting_times = []
            all_completion_times = []
            all_utilizations = []
            all_travel_distances = []
            all_valid_lens = []
            all_raw_h = []
            all_norm_h = []
            seed_mean_rewards = []

            for seed in training_seeds:
                logs = all_eval_data[cfg_name][algo][seed]
                seed_rewards = [x["reward"] for x in logs]
                seed_mean_rewards.append(float(np.mean(seed_rewards)))
                all_rewards.extend(seed_rewards)
                all_completed.extend([x["completed_jobs"] for x in logs])
                all_expired.extend([x["expired_jobs"] for x in logs])
                all_invalids.extend([x["invalid_actions"] for x in logs])
                all_steps.extend([x["length"] for x in logs])
                all_waiting_times.extend([x["avg_waiting_time"] for x in logs])
                all_completion_times.extend([x["avg_completion_time"] for x in logs])
                all_utilizations.extend([x["fleet_utilization"] for x in logs])
                all_travel_distances.extend([x["total_travel_distance"] for x in logs])
                all_valid_lens.extend([x["mean_valid_actions"] for x in logs])
                all_raw_h.extend([x["mean_raw_entropy"] for x in logs])
                all_norm_h.extend([x["mean_norm_entropy"] for x in logs])

            # 1. RL Performance Metrics
            ret_stats = compute_return_statistics(all_rewards)
            succ_rate = compute_success_rate(all_rewards, all_completed, all_expired, return_threshold=0.0)
            cum_regret = compute_cumulative_regret(all_rewards, oracle_benchmarks[cfg_name])
            mean_cum_regret = float(cum_regret[-1]) if cum_regret else 0.0

            # Learning curve metrics (average training curve across seeds)
            mean_train_curve = []
            for ep_i in range(num_episodes):
                ep_r = [all_train_data[cfg_name][algo][s][ep_i]["reward"] for s in training_seeds]
                mean_train_curve.append(float(np.mean(ep_r)))

            conv_speed = compute_convergence_speed(mean_train_curve, threshold=ret_stats["mean"] * 0.8, consecutive_episodes=5)
            sample_eff = compute_sample_efficiency(mean_train_curve, steps_per_episode=100, milestone=15.0)
            stability = compute_stability_across_seeds(seed_mean_rewards)

            # 2. Mine Operational Metrics
            mine_metrics = compute_mine_dispatch_metrics(
                completed_jobs=all_completed,
                expired_jobs=all_expired,
                waiting_times=all_waiting_times,
                completion_times=all_completion_times,
                fleet_utilizations=all_utilizations,
                travel_distances=all_travel_distances,
                total_steps=sum(all_steps),
            )

            # 3. Action Masking & Exploration Metrics
            exp_metrics = compute_action_exploration_metrics(
                invalid_counts=all_invalids,
                total_action_counts=all_steps,
                mean_valid_counts=all_valid_lens,
                raw_entropies=all_raw_h,
                normalized_entropies=all_norm_h,
            )

            # 4. Computational Telemetry
            c_profs = compute_profiles[cfg_name][algo]
            avg_inf_lat = float(np.mean([c["mean_inference_latency_ms"] for c in c_profs]))
            avg_time_10k = float(np.mean([c["time_per_10k_steps_sec"] for c in c_profs]))
            param_cnt = c_profs[0]["parameter_count"] if c_profs else 0

            # Generalization gap (for baseline)
            gen_gap = 0.0
            if cfg_name == "baseline":
                unseen_r = generalization_results[algo]
                gen_gap = compute_generalization_gap(all_rewards, unseen_r)

            record = {
                "config": cfg_name,
                "algorithm": algo,
                "category": CONFIG_DEFINITIONS[cfg_name]["category"],
                "num_trucks": CONFIG_DEFINITIONS[cfg_name]["num_trucks"],
                "num_jobs": CONFIG_DEFINITIONS[cfg_name]["num_jobs"],
                # Return
                "mean_return": ret_stats["mean"],
                "median_return": ret_stats["median"],
                "std_return": ret_stats["std"],
                "ci_95_return": ret_stats["ci_95"],
                "min_return": ret_stats["min"],
                "max_return": ret_stats["max"],
                # RL Metrics
                "success_rate_pct": succ_rate * 100.0,
                "cumulative_regret": mean_cum_regret,
                "convergence_episodes": conv_speed if conv_speed is not None else num_episodes,
                "return_growth_per_1k_steps": sample_eff["return_growth_per_1k_steps"],
                "steps_to_milestone": sample_eff["steps_to_milestone"],
                "seed_std": stability["seed_std"],
                "ci_95_seed_width": stability["ci_95_width"],
                "generalization_gap": gen_gap,
                # Mine Operational
                "mean_deliveries": mine_metrics["mean_deliveries"],
                "total_deliveries": mine_metrics["total_deliveries"],
                "job_completion_rate_pct": mine_metrics["job_completion_rate_pct"],
                "missed_deadline_rate_pct": mine_metrics["missed_deadline_rate_pct"],
                "avg_waiting_time": mine_metrics["avg_waiting_time"],
                "avg_completion_time": mine_metrics["avg_completion_time"],
                "fleet_utilization_pct": mine_metrics["fleet_utilization_pct"],
                "mean_travel_distance": mine_metrics["mean_travel_distance"],
                # Exploration & Masking
                "mean_invalid_actions": exp_metrics["mean_invalid_actions"],
                "invalid_action_rate_pct": exp_metrics["invalid_action_rate_pct"],
                "mean_valid_action_count": exp_metrics["mean_valid_action_count"],
                "mean_raw_entropy": exp_metrics["mean_raw_entropy"],
                "mean_norm_entropy": exp_metrics["mean_normalized_entropy"],
                "entropy_std": exp_metrics["entropy_std"],
                # Computational
                "inference_latency_ms": avg_inf_lat,
                "training_time_10k_steps_sec": avg_time_10k,
                "parameter_count": param_cnt,
            }
            summary_records.append(record)

    df_summary = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(agg_dir, "aggregated_metrics.csv")
    summary_json_path = os.path.join(agg_dir, "aggregated_metrics.json")
    df_summary.to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w") as f:
        json.dump(summary_records, f, indent=2)

    # -------------------------------------------------------------------------
    # PAIRED STATISTICAL SIGNIFICANCE TESTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CALCULATING PAIRED STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    stat_records = []

    pairs = [
        ("ppo", "masked_ppo"),
        ("masked_ppo", "aen_ppo"),
        ("ppo", "aen_ppo"),
    ]

    for cfg_name in configs_to_run:
        for algo_a, algo_b in pairs:
            # Collect matched evaluation seed returns
            returns_a = []
            returns_b = []
            for seed in training_seeds:
                logs_a = all_eval_data[cfg_name][algo_a][seed]
                logs_b = all_eval_data[cfg_name][algo_b][seed]
                for ea, eb in zip(logs_a, logs_b):
                    returns_a.append(ea["reward"])
                    returns_b.append(eb["reward"])

            diff = [b - a for a, b in zip(returns_a, returns_b)]
            test_res = compute_paired_significance(returns_b, returns_a)

            stat_records.append({
                "config": cfg_name,
                "comparison": f"{algo_a.upper()} vs {algo_b.upper()}",
                "baseline_algo": algo_a,
                "target_algo": algo_b,
                "mean_diff": float(np.mean(diff)),
                "std_diff": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
                "t_statistic": test_res["t_stat"],
                "p_value_paired_t": test_res["p_value_t"],
                "wilcoxon_statistic": test_res["wilcoxon_stat"],
                "p_value_wilcoxon": test_res["p_value_wilcoxon"],
                "cohens_d": test_res["cohens_d"],
                "is_significant_p05": bool(test_res["p_value_t"] < 0.05),
                "is_significant_p01": bool(test_res["p_value_t"] < 0.01),
            })

    df_stat = pd.DataFrame(stat_records)
    stat_csv_path = os.path.join(stat_dir, "statistical_significance.csv")
    stat_json_path = os.path.join(stat_dir, "statistical_significance.json")
    df_stat.to_csv(stat_csv_path, index=False)
    with open(stat_json_path, "w") as f:
        json.dump(stat_records, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Summary Table saved to:       {summary_csv_path}")
    print(f"Statistical Tests saved to:   {stat_csv_path}")
    print(f"Raw Logs saved to:            {raw_dir}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run AEN-PPO Multi-Seed Scalability Benchmarks")
    parser.add_argument("--configs", nargs="+", default=None, help="List of configuration keys to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55], help="Random seeds")
    parser.add_argument("--episodes", type=int, default=60, help="Training episodes per seed")
    parser.add_argument("--quick", action="store_true", help="Quick test run with fewer episodes")

    args = parser.parse_args()
    eps = 10 if args.quick else args.episodes
    run_benchmarks(
        selected_configs=args.configs,
        training_seeds=args.seeds,
        num_episodes=eps,
    )


if __name__ == "__main__":
    main()
