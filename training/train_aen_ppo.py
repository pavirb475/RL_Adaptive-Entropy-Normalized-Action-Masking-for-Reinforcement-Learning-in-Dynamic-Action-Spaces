
import os
import sys
import random
import argparse

import numpy as np
import torch

# --------------------------------------------------
# 1. PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environments.truck_dispatch_env import TruckDispatchEnv
from agents.aen_ppo import AENPPOAgent


# --------------------------------------------------
# 2. COMMAND-LINE ARGUMENTS
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train AEN-PPO with a configurable random seed."
)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--episodes", type=int, default=100)

args = parser.parse_args()

SEED = args.seed
NUM_EPISODES = args.episodes
ROLLOUT_STEPS = 512


# --------------------------------------------------
# 3. HYPERPARAMETERS
# --------------------------------------------------

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2

ENTROPY_COEFFICIENT = 0.01
VALUE_COEFFICIENT = 0.5

UPDATE_EPOCHS = 4
BATCH_SIZE = 64
MAX_GRAD_NORM = 0.5


# --------------------------------------------------
# 4. OUTPUT DIRECTORY
# --------------------------------------------------

SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "aen_ppo_multiseed",
    f"seed_{SEED}",
)

MODEL_PATH = os.path.join(SAVE_DIR, "aen_ppo.pt")

os.makedirs(SAVE_DIR, exist_ok=True)


# --------------------------------------------------
# 5. REPRODUCIBILITY AND DEVICE
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("AEN-PPO TRAINING")
print("=" * 65)
print(f"Seed:              {SEED}")
print(f"Episodes:          {NUM_EPISODES}")
print(f"Rollout steps:     {ROLLOUT_STEPS}")
print(f"Device:            {device}")
print(f"Output directory:  {SAVE_DIR}")
print("=" * 65)


# --------------------------------------------------
# 6. ENVIRONMENT AND AGENT
# --------------------------------------------------

env = TruckDispatchEnv()

observation, info = env.reset(seed=SEED)

observation_dim = int(
    np.prod(env.observation_space.shape)
)

action_dim = env.action_space.n

agent = AENPPOAgent(
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


# --------------------------------------------------
# 7. TRAINING LOGS
# --------------------------------------------------

episode_rewards = []
episode_lengths = []
episode_invalid_actions = []
episode_valid_actions = []

update_policy_losses = []
update_value_losses = []
update_raw_entropies = []
update_normalized_entropies = []

rollout_observations = []
rollout_actions = []
rollout_log_probabilities = []
rollout_rewards = []
rollout_dones = []
rollout_terminated = []
rollout_next_values = []
rollout_action_masks = []
rollout_values = []

episode_reward = 0.0
episode_length = 0
episode_invalid_count = 0
episode_valid_count = 0

total_steps = 0
completed_episodes = 0
update_count = 0


# --------------------------------------------------
# 8. TRAINING LOOP
# --------------------------------------------------

while completed_episodes < NUM_EPISODES:

    action_mask = np.asarray(
        env.action_masks(),
        dtype=bool,
    )

    action, log_probability, value = agent.select_action(
        observation,
        action_mask,
    )

    (
        next_observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    done = terminated or truncated

    # Bootstrap from the next state unless the state is terminal.
    with torch.no_grad():
        if terminated:
            next_value = 0.0
        else:
            next_obs_tensor = torch.as_tensor(
                next_observation,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            next_features = agent.network.shared(
                next_obs_tensor
            )

            next_value = (
                agent.network.critic(next_features)
                .squeeze(-1)
                .item()
            )

    # Store transition.
    rollout_observations.append(
        np.asarray(observation, dtype=np.float32).copy()
    )
    rollout_actions.append(int(action))
    rollout_log_probabilities.append(float(log_probability))
    rollout_rewards.append(float(reward))
    rollout_dones.append(float(done))
    rollout_terminated.append(float(terminated))
    rollout_next_values.append(float(next_value))
    rollout_action_masks.append(action_mask.copy())
    rollout_values.append(float(value))

    # Update episode statistics.
    episode_reward += float(reward)
    episode_length += 1
    total_steps += 1

    if info.get("invalid_action", False):
        episode_invalid_count += 1
    else:
        episode_valid_count += 1

    observation = next_observation

    # --------------------------------------------------
    # 9. PPO UPDATE
    # --------------------------------------------------

    rollout_full = len(rollout_rewards) >= ROLLOUT_STEPS

    if rollout_full or done:

        rewards_array = np.asarray(
            rollout_rewards,
            dtype=np.float32,
        )

        values_array = np.asarray(
            rollout_values,
            dtype=np.float32,
        )

        dones_array = np.asarray(
            rollout_dones,
            dtype=np.float32,
        )

        terminated_array = np.asarray(
            rollout_terminated,
            dtype=np.float32,
        )

        next_values_array = np.asarray(
            rollout_next_values,
            dtype=np.float32,
        )

        advantages, returns = agent.compute_gae(
            rewards=rewards_array,
            values=values_array,
            dones=dones_array,
            next_value=next_values_array[-1],
            terminated=terminated_array,
            next_values=next_values_array,
        )

        update_metrics = agent.update(
            observations=np.asarray(
                rollout_observations,
                dtype=np.float32,
            ),
            actions=np.asarray(
                rollout_actions,
                dtype=np.int64,
            ),
            old_log_probabilities=np.asarray(
                rollout_log_probabilities,
                dtype=np.float32,
            ),
            advantages=advantages,
            returns=returns,
            action_masks=np.asarray(
                rollout_action_masks,
                dtype=bool,
            ),
        )

        update_count += 1

        update_policy_losses.append(
            float(update_metrics["policy_loss"])
        )

        update_value_losses.append(
            float(update_metrics["value_loss"])
        )

        # Correct key: agents/aen_ppo.py returns "entropy".
        update_raw_entropies.append(
            float(update_metrics["entropy"])
        )

        update_normalized_entropies.append(
            float(update_metrics["normalized_entropy"])
        )

        print(
            f"Update {update_count:03d} | "
            f"Steps {total_steps:05d} | "
            f"Policy Loss: "
            f"{update_metrics['policy_loss']:.4f} | "
            f"Value Loss: "
            f"{update_metrics['value_loss']:.4f} | "
            f"Raw Entropy: "
            f"{update_metrics['entropy']:.4f} | "
            f"Normalized Entropy: "
            f"{update_metrics['normalized_entropy']:.4f}"
        )

        # Clear rollout buffers after updating.
        rollout_observations.clear()
        rollout_actions.clear()
        rollout_log_probabilities.clear()
        rollout_rewards.clear()
        rollout_dones.clear()
        rollout_terminated.clear()
        rollout_next_values.clear()
        rollout_action_masks.clear()
        rollout_values.clear()

    # --------------------------------------------------
    # 10. EPISODE COMPLETION
    # --------------------------------------------------

    if done:

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_invalid_actions.append(
            episode_invalid_count
        )
        episode_valid_actions.append(
            episode_valid_count
        )

        completed_episodes += 1

        print(
            f"Episode {completed_episodes:03d}/"
            f"{NUM_EPISODES} | "
            f"Reward: {episode_reward:.2f} | "
            f"Length: {episode_length} | "
            f"Invalid Actions: {episode_invalid_count} | "
            f"Valid Actions: {episode_valid_count}"
        )

        observation, info = env.reset()

        episode_reward = 0.0
        episode_length = 0
        episode_invalid_count = 0
        episode_valid_count = 0


# --------------------------------------------------
# 11. SAVE MODEL AND METRICS
# --------------------------------------------------

torch.save(
    agent.network.state_dict(),
    MODEL_PATH,
)

metrics_to_save = {
    "episode_rewards.npy": episode_rewards,
    "episode_lengths.npy": episode_lengths,
    "episode_invalid_actions.npy": episode_invalid_actions,
    "episode_valid_actions.npy": episode_valid_actions,
    "update_policy_losses.npy": update_policy_losses,
    "update_value_losses.npy": update_value_losses,
    "update_raw_entropies.npy": update_raw_entropies,
    "update_normalized_entropies.npy": update_normalized_entropies,
}

for filename, values in metrics_to_save.items():
    np.save(
        os.path.join(SAVE_DIR, filename),
        np.asarray(values, dtype=np.float32),
    )


# --------------------------------------------------
# 12. FINAL SUMMARY
# --------------------------------------------------

def mean_or_zero(values):
    return float(np.mean(values)) if len(values) else 0.0


def std_or_zero(values):
    return float(np.std(values)) if len(values) else 0.0


last_n = min(10, len(episode_rewards))

recent_rewards = episode_rewards[-last_n:]
recent_invalid = episode_invalid_actions[-last_n:]
recent_valid = episode_valid_actions[-last_n:]

print("\n" + "=" * 65)
print("AEN-PPO TRAINING COMPLETE")
print("=" * 65)

print(f"Seed:                         {SEED}")
print(f"Episodes completed:           {completed_episodes}")
print(f"Total environment steps:      {total_steps}")
print(f"Policy updates:               {update_count}")
print(
    f"Overall mean reward:          "
    f"{mean_or_zero(episode_rewards):.2f}"
)
print(
    f"Final-{last_n} mean reward:          "
    f"{mean_or_zero(recent_rewards):.2f}"
)
print(
    f"Final-{last_n} reward std:           "
    f"{std_or_zero(recent_rewards):.2f}"
)
print(
    f"Final-{last_n} mean invalid actions: "
    f"{mean_or_zero(recent_invalid):.2f}"
)
print(
    f"Final-{last_n} mean valid actions:   "
    f"{mean_or_zero(recent_valid):.2f}"
)
print(f"Model saved to:               {MODEL_PATH}")
print(f"Metrics saved in:             {SAVE_DIR}")
print("=" * 65)

env.close()
