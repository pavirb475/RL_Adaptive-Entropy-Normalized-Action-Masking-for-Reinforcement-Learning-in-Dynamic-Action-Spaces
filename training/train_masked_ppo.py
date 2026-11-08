
import os
import sys
import random

import numpy as np
import torch

# Add project root to Python path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environments.truck_dispatch_env import TruckDispatchEnv
from agents.masked_ppo import MaskedPPOAgent


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
NUM_EPISODES = 100
ROLLOUT_STEPS = 512

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFFICIENT = 0.01
VALUE_COEFFICIENT = 0.5

UPDATE_EPOCHS = 4
BATCH_SIZE = 64
MAX_GRAD_NORM = 0.5

# Use a new directory to preserve previous results.
SAVE_DIR = os.path.join(
    PROJECT_ROOT, "results", "masked_ppo_controlled"
)
MODEL_PATH = os.path.join(SAVE_DIR, "masked_ppo.pt")


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MASKED PPO TRAINING")
print("=" * 60)
print(f"Seed: {SEED}")
print(f"Episodes: {NUM_EPISODES}")
print(f"Rollout size: {ROLLOUT_STEPS}")
print(f"Device: {device}")
print(f"Saving results to: {SAVE_DIR}")


# ============================================================
# 3. Environment and agent
# ============================================================

env = TruckDispatchEnv()
observation, info = env.reset(seed=SEED)

observation_dim = int(
    np.asarray(observation).shape[0]
)
action_dim = int(env.action_space.n)

agent = MaskedPPOAgent(
    observation_dim=observation_dim,
    action_dim=action_dim,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_epsilon=CLIP_EPSILON,
    entropy_coefficient=ENTROPY_COEFFICIENT,
    value_coefficient=VALUE_COEFFICIENT,
    max_grad_norm=MAX_GRAD_NORM,
    update_epochs=UPDATE_EPOCHS,
    batch_size=BATCH_SIZE,
    device=device,
)

print(f"Observation dimension: {observation_dim}")
print(f"Action dimension: {action_dim}")


# ============================================================
# 4. Rollout buffer
# ============================================================

def create_buffer():
    return {
        "observations": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "values": [],
        # Episode ended: terminated OR truncated.
        "dones": [],
        # True MDP termination only.
        "terminated": [],
        # Value estimates of actual next observations.
        "next_values": [],
        # Valid-action mask for every stored state.
        "masks": [],
    }


buffer = create_buffer()


# ============================================================
# 5. Estimate a next-state value
# ============================================================

@torch.no_grad()
def estimate_value(obs):
    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    features = agent.network.shared(obs_tensor)
    value = agent.network.critic(features).squeeze(-1)

    return float(value.item())


# ============================================================
# 6. Update the agent using one rollout
# ============================================================

def update_agent(data):
    if not data["observations"]:
        return None

    observations = np.asarray(
        data["observations"], dtype=np.float32
    )
    actions = np.asarray(
        data["actions"], dtype=np.int64
    )
    old_log_probs = np.asarray(
        data["log_probs"], dtype=np.float32
    )
    rewards = np.asarray(
        data["rewards"], dtype=np.float32
    )
    values = np.asarray(
        data["values"], dtype=np.float32
    )
    dones = np.asarray(
        data["dones"], dtype=np.float32
    )
    terminated = np.asarray(
        data["terminated"], dtype=np.float32
    )
    next_values = np.asarray(
        data["next_values"], dtype=np.float32
    )
    masks = np.asarray(
        data["masks"], dtype=bool
    )

    # GAE bootstraps through truncations but stops the
    # advantage recursion at every episode boundary.
    advantages, returns = agent.compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=0.0,
        terminated=terminated,
        next_values=next_values,
    )

    stats = agent.update(
        observations,
        actions,
        old_log_probs,
        advantages,
        returns,
        masks,
    )

    return stats


# ============================================================
# 7. Training loop
# ============================================================

episode_rewards = []
episode_lengths = []
episode_invalid_actions = []

episodes_completed = 0
total_steps = 0
update_count = 0

current_episode_reward = 0.0
current_episode_length = 0
current_episode_invalid_actions = 0

while episodes_completed < NUM_EPISODES:

    # Collect a rollout across episode boundaries until
    # it reaches 512 transitions or training is complete.
    while (
        len(buffer["observations"]) < ROLLOUT_STEPS
        and episodes_completed < NUM_EPISODES
    ):
        action_mask = np.asarray(
            env.action_masks(), dtype=bool
        )

        if action_mask.ndim != 1:
            raise ValueError(
                "Expected a one-dimensional action mask."
            )

        if len(action_mask) != action_dim:
            raise ValueError(
                "Action-mask size does not match action space."
            )

        if not action_mask.any():
            raise ValueError(
                "The environment returned no valid actions."
            )

        # Choose an action only from the valid actions.
        action, log_prob, value = agent.select_action(
            observation, action_mask
        )

        if not action_mask[action]:
            raise RuntimeError(
                "Masked PPO selected an invalid action."
            )

        next_observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        # Evaluate the real next observation BEFORE resetting.
        # True termination: no bootstrap.
        # Time-limit truncation: bootstrap from final observation.
        if terminated:
            transition_next_value = 0.0
        else:
            transition_next_value = estimate_value(
                next_observation
            )

        # Store this transition.
        buffer["observations"].append(
            np.asarray(observation, dtype=np.float32).copy()
        )
        buffer["actions"].append(int(action))
        buffer["log_probs"].append(float(log_prob))
        buffer["rewards"].append(float(reward))
        buffer["values"].append(float(value))
        buffer["dones"].append(float(done))
        buffer["terminated"].append(float(terminated))
        buffer["next_values"].append(
            transition_next_value
        )
        buffer["masks"].append(action_mask.copy())

        # Track episode metrics.
        current_episode_reward += float(reward)
        current_episode_length += 1
        total_steps += 1

        # Count invalid actions from the mask recorded
        # before the action was taken.
        if not action_mask[action]:
            current_episode_invalid_actions += 1

        observation = next_observation

        if done:
            episodes_completed += 1

            episode_rewards.append(
                current_episode_reward
            )
            episode_lengths.append(
                current_episode_length
            )
            episode_invalid_actions.append(
                current_episode_invalid_actions
            )

            print(
                f"Episode {episodes_completed:3d}/"
                f"{NUM_EPISODES} | "
                f"Reward: {current_episode_reward:9.2f} | "
                f"Length: {current_episode_length:3d} | "
                f"Invalid actions: "
                f"{current_episode_invalid_actions}"
            )

            current_episode_reward = 0.0
            current_episode_length = 0
            current_episode_invalid_actions = 0

            # Reset only after the transition and its
            # next-state value have been stored.
            if episodes_completed < NUM_EPISODES:
                observation, info = env.reset()

    # Update after a complete rollout, or after the final
    # partial rollout when the episode limit is reached.
    if buffer["observations"]:
        stats = update_agent(buffer)

        if stats is not None:
            update_count += 1

        buffer = create_buffer()


# ============================================================
# 8. Save model and metrics
# ============================================================

torch.save(agent.network.state_dict(), MODEL_PATH)

np.save(
    os.path.join(SAVE_DIR, "episode_rewards.npy"),
    np.asarray(episode_rewards, dtype=np.float32),
)
np.save(
    os.path.join(SAVE_DIR, "episode_lengths.npy"),
    np.asarray(episode_lengths, dtype=np.int32),
)
np.save(
    os.path.join(SAVE_DIR, "invalid_actions.npy"),
    np.asarray(episode_invalid_actions, dtype=np.int32),
)

print("\n" + "=" * 60)
print("MASKED PPO TRAINING COMPLETE")
print("=" * 60)
print(f"Episodes completed: {episodes_completed}")
print(f"Total environment steps: {total_steps}")
print(f"Policy updates: {update_count}")
print(f"Mean reward (last 10): "
      f"{np.mean(episode_rewards[-10:]):.2f}")
print(f"Mean invalid actions (last 10): "
      f"{np.mean(episode_invalid_actions[-10:]):.2f}")
print(f"Model saved to: {MODEL_PATH}")
print(f"Metrics saved to: {SAVE_DIR}")

env.close()
