"""
Unified Training Pipeline on the Visual Grid Environment.
Trains Standard PPO, Masked PPO, or AEN-PPO with reproducible seeds and research metrics.
Usage:
    python training/train_visual.py --agent aen_ppo --episodes 100 --seed 42
    python training/train_visual.py --agent masked_ppo --episodes 100 --seed 42
    python training/train_visual.py --agent ppo --episodes 100 --seed 42
"""

import os
import sys
import random
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environments.grid_map import GridMap
from environments.visual_truck_dispatch_env import VisualTruckDispatchEnv
from agents.ppo import PPOAgent
from agents.masked_ppo import MaskedPPOAgent
from agents.aen_ppo import AENPPOAgent


def train(agent_name: str, num_episodes: int = 100, seed: int = 42, map_path: str = "maps/starter_map.json"):
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory
    save_dir = os.path.join(PROJECT_ROOT, "results", "visual_environment", f"{agent_name}_seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{agent_name}.pt")

    # Map & Env
    if os.path.exists(map_path):
        grid_map = GridMap.load_from_json(map_path)
    else:
        grid_map = GridMap.create_default_map()

    env = VisualTruckDispatchEnv(grid_map=grid_map, max_steps=100, seed=seed)
    obs, info = env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent setup
    if agent_name == "aen_ppo":
        agent = AENPPOAgent(obs_dim, action_dim, device=device)
    elif agent_name == "masked_ppo":
        agent = MaskedPPOAgent(obs_dim, action_dim, device=device)
    elif agent_name == "ppo":
        agent = PPOAgent(obs_dim, action_dim, device=device)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    print("=" * 65)
    print(f"TRAINING {agent_name.upper()} ON VISUAL GRID WORLD")
    print("=" * 65)
    print(f"Seed:               {seed}")
    print(f"Episodes:           {num_episodes}")
    print(f"Device:             {device}")
    print(f"Output directory:   {save_dir}")
    print("=" * 65)

    rollout_steps = 512
    buffer = {
        "obs": [], "actions": [], "log_probs": [], "rewards": [],
        "values": [], "dones": [], "terminated": [], "next_values": [], "masks": []
    }

    episode_rewards = []
    episode_lengths = []
    episode_invalids = []
    episode_completed = []

    curr_ep_reward = 0.0
    curr_ep_len = 0
    curr_ep_invalid = 0
    episodes_done = 0
    total_steps = 0
    updates_done = 0

    while episodes_done < num_episodes:
        mask = env.action_masks()

        # Action selection
        if agent_name in ["aen_ppo", "masked_ppo"]:
            action, log_prob, value = agent.select_action(obs, mask)
        else:
            action, log_prob, value = agent.select_action(obs)

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        # Value bootstrap estimation
        if terminated:
            next_val = 0.0
        else:
            with torch.no_grad():
                ot = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                feat = agent.network.shared(ot)
                next_val = float(agent.network.critic(feat).squeeze(-1).item())

        # Store transition
        buffer["obs"].append(np.asarray(obs, dtype=np.float32).copy())
        buffer["actions"].append(int(action))
        buffer["log_probs"].append(float(log_prob))
        buffer["rewards"].append(float(reward))
        buffer["values"].append(float(value))
        buffer["dones"].append(float(done))
        buffer["terminated"].append(float(terminated))
        buffer["next_values"].append(float(next_val))
        buffer["masks"].append(mask.copy())

        curr_ep_reward += float(reward)
        curr_ep_len += 1
        total_steps += 1

        if not step_info.get("valid_action", True):
            curr_ep_invalid += 1

        obs = next_obs

        # Episode completion
        if done:
            episodes_done += 1
            episode_rewards.append(curr_ep_reward)
            episode_lengths.append(curr_ep_len)
            episode_invalids.append(curr_ep_invalid)
            episode_completed.append(env.completed_jobs)

            if episodes_done % 10 == 0 or episodes_done == num_episodes:
                print(
                    f"Episode {episodes_done:3d}/{num_episodes} | "
                    f"Reward: {curr_ep_reward:7.2f} | "
                    f"Jobs Done: {env.completed_jobs:2d} | "
                    f"Invalid Actions: {curr_ep_invalid:2d}"
                )

            curr_ep_reward = 0.0
            curr_ep_len = 0
            curr_ep_invalid = 0

            if episodes_done < num_episodes:
                obs, info = env.reset()

        # PPO Update
        if len(buffer["obs"]) >= rollout_steps or (done and episodes_done >= num_episodes and len(buffer["obs"]) > 0):
            adv, ret = agent.compute_gae(
                rewards=np.asarray(buffer["rewards"], dtype=np.float32),
                values=np.asarray(buffer["values"], dtype=np.float32),
                dones=np.asarray(buffer["dones"], dtype=np.float32),
                next_value=0.0,
                terminated=np.asarray(buffer["terminated"], dtype=np.float32),
                next_values=np.asarray(buffer["next_values"], dtype=np.float32),
            )

            if agent_name in ["aen_ppo", "masked_ppo"]:
                stats = agent.update(
                    observations=np.asarray(buffer["obs"], dtype=np.float32),
                    actions=np.asarray(buffer["actions"], dtype=np.int64),
                    old_log_probabilities=np.asarray(buffer["log_probs"], dtype=np.float32),
                    advantages=adv,
                    returns=ret,
                    action_masks=np.asarray(buffer["masks"], dtype=bool),
                )
            else:
                stats = agent.update(
                    observations=np.asarray(buffer["obs"], dtype=np.float32),
                    actions=np.asarray(buffer["actions"], dtype=np.int64),
                    old_log_probabilities=np.asarray(buffer["log_probs"], dtype=np.float32),
                    advantages=adv,
                    returns=ret,
                )

            updates_done += 1
            buffer = {k: [] for k in buffer}

    # Save model checkpoint and metrics
    torch.save(agent.network.state_dict(), model_path)
    np.save(os.path.join(save_dir, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(os.path.join(save_dir, "episode_lengths.npy"), np.asarray(episode_lengths, dtype=np.int32))
    np.save(os.path.join(save_dir, "invalid_actions.npy"), np.asarray(episode_invalids, dtype=np.int32))
    np.save(os.path.join(save_dir, "completed_jobs.npy"), np.asarray(episode_completed, dtype=np.int32))

    print("\n" + "=" * 65)
    print(f"TRAINING COMPLETE: {agent_name.upper()}")
    print("=" * 65)
    print(f"Mean Reward (last 10):         {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Mean Invalid Actions (last 10): {np.mean(episode_invalids[-10:]):.2f}")
    print(f"Mean Completed Jobs (last 10):  {np.mean(episode_completed[-10:]):.2f}")
    print(f"Saved Checkpoint:              {model_path}")
    print(f"Saved Metrics:                 {save_dir}")
    print("=" * 65 + "\n")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL Agents on Visual Truck Dispatch Environment")
    parser.add_argument("--agent", type=str, choices=["aen_ppo", "masked_ppo", "ppo"], default="aen_ppo")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map", type=str, default="maps/starter_map.json")

    args = parser.parse_args()
    train(agent_name=args.agent, num_episodes=args.episodes, seed=args.seed, map_path=args.map)


if __name__ == "__main__":
    main()
