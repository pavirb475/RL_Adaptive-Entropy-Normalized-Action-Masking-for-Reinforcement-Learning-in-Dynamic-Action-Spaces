"""
Master Visual Simulation and Map Editor Launcher for AEN-PPO.
Usage:
    python run_visual_simulation.py               # Launch visual simulator with AEN-PPO
    python run_visual_simulation.py --mode editor  # Launch interactive Map Editor
    python run_visual_simulation.py --agent masked_ppo
    python run_visual_simulation.py --agent ppo
    python run_visual_simulation.py --map maps/evaluation_map.json
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environments.grid_map import GridMap
from environments.visual_truck_dispatch_env import VisualTruckDispatchEnv
from visualization.simulator_window import SimulatorWindow
from visualization.map_editor import MapEditor
from agents.ppo import PPOAgent
from agents.masked_ppo import MaskedPPOAgent
from agents.aen_ppo import AENPPOAgent


def load_agent(agent_name: str, obs_dim: int = 36, action_dim: int = 25, device: str = "cpu"):
    """
    Instantiates the requested agent and attempts to load its trained weights.
    """
    ckpt_map = {
        "aen_ppo": [
            os.path.join(PROJECT_ROOT, "results", "aen_ppo_controlled", "aen_ppo.pt"),
            os.path.join(PROJECT_ROOT, "results", "aen_ppo_multiseed", "seed_123", "aen_ppo.pt"),
            os.path.join(PROJECT_ROOT, "results", "aen_ppo_multiseed", "seed_42", "aen_ppo.pt"),
        ],
        "masked_ppo": [
            os.path.join(PROJECT_ROOT, "results", "masked_ppo_controlled", "masked_ppo.pt"),
            os.path.join(PROJECT_ROOT, "results", "masked_ppo", "masked_ppo.pt"),
        ],
        "ppo": [
            os.path.join(PROJECT_ROOT, "results", "ppo_controlled", "ppo_baseline.pt"),
            os.path.join(PROJECT_ROOT, "results", "ppo", "ppo_baseline.pt"),
        ],
    }

    if agent_name == "aen_ppo":
        agent = AENPPOAgent(obs_dim, action_dim, device=device)
    elif agent_name == "masked_ppo":
        agent = MaskedPPOAgent(obs_dim, action_dim, device=device)
    elif agent_name == "ppo":
        agent = PPOAgent(obs_dim, action_dim, device=device)
    else:
        return None

    # Try loading weights
    candidates = ckpt_map.get(agent_name, [])
    loaded = False
    for path in candidates:
        if os.path.exists(path):
            try:
                st = torch.load(path, map_location=device)
                if isinstance(st, dict) and "model_state_dict" in st:
                    st = st["model_state_dict"]
                agent.network.load_state_dict(st)
                print(f"[AEN-PPO Launcher] Loaded {agent_name.upper()} weights from: {path}")
                loaded = True
                break
            except Exception as e:
                print(f"[AEN-PPO Launcher] Warning loading {path}: {e}")

    if not loaded and agent is not None:
        print(f"[AEN-PPO Launcher] Note: No pretrained checkpoint found for {agent_name}; using initialized policy.")

    return agent


def select_policy_action(agent, agent_name: str, obs: np.ndarray, mask: np.ndarray, env: VisualTruckDispatchEnv) -> int:
    """
    Selects action according to policy mode.
    """
    if agent_name in ["aen_ppo", "masked_ppo"] and agent is not None:
        action, _, _ = agent.select_action(obs, mask)
        return int(action)
    elif agent_name == "ppo" and agent is not None:
        action, _, _ = agent.select_action(obs)
        return int(action)
    elif agent_name == "random":
        valid = np.flatnonzero(mask)
        if len(valid) > 0:
            return int(np.random.choice(valid))
        return int(env.wait_action)
    elif agent_name == "manual":
        # First valid dispatch, or wait
        valid = np.flatnonzero(mask)
        return int(valid[0]) if len(valid) > 0 else int(env.wait_action)

    # Fallback random
    return int(np.random.randint(0, env.num_actions))


def run_simulation(
    map_path: str = "maps/starter_map.json",
    initial_agent: str = "aen_ppo",
    seed: int = 42,
    num_trucks: int = 4,
    num_jobs: int = 6,
):
    """
    Runs the main interactive graphical simulation loop with Pygame.
    Supports variable fleet sizes, pit counts, and job queues.
    """
    if os.path.exists(map_path):
        grid_map = GridMap.load_from_json(map_path)
    else:
        grid_map = GridMap.create_default_map()

    env = VisualTruckDispatchEnv(
        grid_map=grid_map,
        num_trucks=num_trucks,
        num_jobs=num_jobs,
        max_steps=100,
        seed=seed,
    )
    obs, info = env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    window = SimulatorWindow(env=env)
    window.controls.set_agent(initial_agent)

    # Cache agent models with matching environment dimensions
    cached_agents = {
        "aen_ppo": load_agent("aen_ppo", obs_dim=obs_dim, action_dim=action_dim),
        "masked_ppo": load_agent("masked_ppo", obs_dim=obs_dim, action_dim=action_dim),
        "ppo": load_agent("ppo", obs_dim=obs_dim, action_dim=action_dim),
    }


    last_step_time = time.time()
    is_running = True

    print("\n" + "=" * 65)
    print("OPEN-PIT MINE TRUCK DISPATCH SIMULATOR LAUNCHED")
    print("=" * 65)
    print("Controls:")
    print("  [SPACE]       : Play / Pause toggle")
    print("  [RIGHT ARROW] : Step 1 simulation step")
    print("  [UP / DOWN]   : Increase / Decrease speed")
    print("  [R]           : Reset simulation")
    print("  [A]           : Cycle agent policy (AEN-PPO / Masked PPO / PPO / Random)")
    print("  [P]           : Toggle route trajectory lines")
    print("  [ESC]         : Exit")
    print("=" * 65 + "\n")

    while is_running:
        if not window.handle_events():
            is_running = False
            break

        current_time = time.time()
        speed = window.controls.current_speed
        step_interval = 1.0 / max(1, speed)

        # Handle simulation advance
        should_step = False
        if window.controls.step_requested:
            should_step = True
            window.controls.step_requested = False
        elif not window.controls.is_paused and (current_time - last_step_time >= step_interval):
            should_step = True
            last_step_time = current_time

        if should_step:
            active_name = window.controls.selected_agent
            agent_obj = cached_agents.get(active_name, None)
            mask = env.action_masks()

            action = select_policy_action(agent_obj, active_name, obs, mask, env)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            window.record_step_result(action, reward, step_info)

            obs = next_obs

            if terminated or truncated:
                print(f"[Sim] Episode finished! Completed: {env.completed_jobs} jobs | Cumulative Reward: {window.cumulative_reward:.2f}")
                obs, _ = env.reset()
                window.reset_sim()

        # Update smooth vehicle movement animation
        window.update_animation()

        # Render display
        window.render_frame()
        window.clock.tick(60)

    window.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="AEN-PPO Visual Simulation & Map Editor")
    parser.add_argument("--mode", type=str, choices=["sim", "editor"], default="sim", help="Launch mode: 'sim' or 'editor'")
    parser.add_argument("--agent", type=str, choices=["aen_ppo", "masked_ppo", "ppo", "random", "manual"], default="aen_ppo", help="RL Agent policy")
    parser.add_argument("--map", type=str, default="maps/starter_map.json", help="Path to map JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Simulation random seed")

    parser.add_argument("--trucks", type=int, default=4, help="Number of trucks (e.g. 2, 4, 6, 8)")
    parser.add_argument("--jobs", type=int, default=6, help="Number of jobs (e.g. 4, 6, 8, 12)")

    args = parser.parse_args()

    if args.mode == "editor":
        editor = MapEditor(map_filepath=args.map)
        launch_sim = editor.run()
        if launch_sim:
            run_simulation(
                map_path=args.map,
                initial_agent=args.agent,
                seed=args.seed,
                num_trucks=args.trucks,
                num_jobs=args.jobs,
            )
    else:
        run_simulation(
            map_path=args.map,
            initial_agent=args.agent,
            seed=args.seed,
            num_trucks=args.trucks,
            num_jobs=args.jobs,
        )


if __name__ == "__main__":
    main()

