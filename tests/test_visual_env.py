"""
Unit tests for the VisualTruckDispatchEnv Gymnasium environment.
"""

import numpy as np
from environments.visual_truck_dispatch_env import VisualTruckDispatchEnv


def test_visual_env_lifecycle():
    env = VisualTruckDispatchEnv(seed=42)
    obs, info = env.reset(seed=42)

    # Check shapes
    assert obs.shape == (36,)
    assert env.action_space.n == 25
    assert "action_mask" in info
    assert len(info["action_mask"]) == 25
    assert info["num_valid_actions"] > 0

    # Execute step with a valid action
    mask = env.action_masks()
    valid_actions = np.flatnonzero(mask)
    action = int(valid_actions[0])

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (36,)
    assert isinstance(reward, float)
    assert not terminated
    assert "action_mask" in step_info

    # Execute step with invalid action (if one exists)
    invalid_actions = np.flatnonzero(mask == 0)
    if len(invalid_actions) > 0:
        inv_act = int(invalid_actions[0])
        _, inv_reward, _, _, inv_info = env.step(inv_act)
        assert inv_reward == -5.0
        assert not inv_info["valid_action"]

    env.close()


def test_reproducibility():
    env1 = VisualTruckDispatchEnv(seed=100)
    obs1, info1 = env1.reset(seed=100)

    env2 = VisualTruckDispatchEnv(seed=100)
    obs2, info2 = env2.reset(seed=100)

    np.testing.assert_allclose(obs1, obs2, rtol=1e-5)
    np.testing.assert_array_equal(info1["action_mask"], info2["action_mask"])

    env1.close()
    env2.close()


if __name__ == "__main__":
    test_visual_env_lifecycle()
    test_reproducibility()
    print("test_visual_env.py PASSED!")
