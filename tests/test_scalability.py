"""
Unit tests for Environment Scalability and Multi-Config Support.
Validates varying trucks (2, 4, 6, 8), pits (1, 2, 3, 4), and jobs (4, 6, 8, 12).
"""

import pytest
import numpy as np
import torch

from environments.grid_map import GridMap
from environments.visual_truck_dispatch_env import VisualTruckDispatchEnv
from environments.pathfinding import PathCache
from agents.ppo import ActorCritic, PPOAgent
from agents.masked_ppo import MaskedPPOAgent
from agents.aen_ppo import AENPPOAgent


@pytest.mark.parametrize("num_trucks", [2, 4, 6, 8])
def test_truck_scalability(num_trucks):
    num_jobs = 6
    expected_obs_dim = num_trucks * 3 + num_jobs * 4
    expected_act_dim = num_trucks * num_jobs + 1

    env = VisualTruckDispatchEnv(num_trucks=num_trucks, num_jobs=num_jobs, seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (expected_obs_dim,), f"Expected obs shape {(expected_obs_dim,)}, got {obs.shape}"
    assert env.action_space.n == expected_act_dim
    assert len(info["action_mask"]) == expected_act_dim
    assert len(env.trucks) == num_trucks

    # Take a step with valid action
    mask = env.action_masks()
    valid_actions = np.flatnonzero(mask)
    assert len(valid_actions) > 0
    act = int(valid_actions[0])
    next_obs, reward, term, trunc, step_info = env.step(act)
    assert next_obs.shape == (expected_obs_dim,)
    assert "fleet_utilization" in step_info
    assert "avg_waiting_time" in step_info
    assert "avg_completion_time" in step_info
    env.close()


@pytest.mark.parametrize("num_pits", [1, 2, 3, 4])
def test_pit_scalability(num_pits):
    m = GridMap.create_scalable_map(num_pits=num_pits, num_dumping=2, num_truck_spawns=8)
    assert len(m.loading_stations) == num_pits
    assert len(m.dumping_stations) == 2
    assert len(m.truck_spawns) == 8

    cache = PathCache(m)
    for spawn in m.truck_spawns:
        for pit in m.loading_stations:
            p = cache.get_path(tuple(spawn), tuple(pit["pos"]))
            assert p is not None, f"No path from spawn {spawn} to pit {pit}"

    env = VisualTruckDispatchEnv(grid_map=m, num_trucks=4, num_jobs=6, seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (4 * 3 + 6 * 4,)
    env.close()


@pytest.mark.parametrize("num_jobs", [4, 6, 8, 12])
def test_job_scalability(num_jobs):
    num_trucks = 4
    expected_obs_dim = num_trucks * 3 + num_jobs * 4
    expected_act_dim = num_trucks * num_jobs + 1

    env = VisualTruckDispatchEnv(num_trucks=num_trucks, num_jobs=num_jobs, seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (expected_obs_dim,)
    assert env.action_space.n == expected_act_dim
    assert len(info["action_mask"]) == expected_act_dim
    assert env.job_manager.num_jobs == num_jobs

    mask = env.action_masks()
    valid_actions = np.flatnonzero(mask)
    assert len(valid_actions) > 0
    act = int(valid_actions[0])
    next_obs, reward, term, trunc, step_info = env.step(act)
    assert next_obs.shape == (expected_obs_dim,)
    env.close()


def test_combined_large_scale():
    num_trucks = 8
    num_jobs = 12
    m = GridMap.create_scalable_map(num_pits=4, num_dumping=2, num_truck_spawns=8)
    env = VisualTruckDispatchEnv(grid_map=m, num_trucks=num_trucks, num_jobs=num_jobs, seed=42)
    obs, info = env.reset(seed=42)

    expected_obs_dim = 8 * 3 + 12 * 4  # 72
    expected_act_dim = 8 * 12 + 1       # 97
    assert obs.shape == (expected_obs_dim,)
    assert env.action_space.n == expected_act_dim

    # Test ActorCritic and all 3 agents
    ppo_agent = PPOAgent(expected_obs_dim, expected_act_dim)
    masked_agent = MaskedPPOAgent(expected_obs_dim, expected_act_dim)
    aen_agent = AENPPOAgent(expected_obs_dim, expected_act_dim)

    mask = env.action_masks()
    a_ppo, _, _ = ppo_agent.select_action(obs)
    a_mask, _, _ = masked_agent.select_action(obs, mask)
    a_aen, _, _ = aen_agent.select_action(obs, mask)

    assert 0 <= a_ppo < expected_act_dim
    assert mask[a_mask] == 1
    assert mask[a_aen] == 1

    env.close()
