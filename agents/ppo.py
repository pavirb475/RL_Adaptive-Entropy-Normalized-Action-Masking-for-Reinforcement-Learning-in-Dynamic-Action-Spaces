import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================
# ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):
    """
    Shared MLP with separate actor and critic heads.

    Input:
        observation vector

    Output:
        action logits
        state value
    """

    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=128,
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.Tanh(),
        )

        # Actor
        self.actor = nn.Linear(
            hidden_dim,
            action_dim,
        )

        # Critic
        self.critic = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, observation):

        features = self.shared(
            observation
        )

        logits = self.actor(
            features
        )

        value = self.critic(
            features
        ).squeeze(-1)

        return logits, value

    def get_action_and_value(
        self,
        observation,
        action=None,
    ):

        logits, value = self.forward(
            observation
        )

        distribution = Categorical(
            logits=logits
        )

        if action is None:

            action = distribution.sample()

        log_probability = (
            distribution.log_prob(
                action
            )
        )

        entropy = (
            distribution.entropy()
        )

        return (
            action,
            log_probability,
            entropy,
            value,
        )


# ============================================================
# PPO AGENT
# ============================================================

class PPOAgent:

    def __init__(
        self,
        observation_dim,
        action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64,
        device=None,
    ):

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        # ----------------------------------------------------
        # HYPERPARAMETERS
        # ----------------------------------------------------

        self.gamma = gamma

        self.gae_lambda = (
            gae_lambda
        )

        self.clip_epsilon = (
            clip_epsilon
        )

        self.entropy_coefficient = (
            entropy_coefficient
        )

        self.value_coefficient = (
            value_coefficient
        )

        self.max_grad_norm = (
            max_grad_norm
        )

        self.update_epochs = (
            update_epochs
        )

        self.batch_size = (
            batch_size
        )

        # ----------------------------------------------------
        # NETWORK
        # ----------------------------------------------------

        self.network = ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
        ).to(self.device)

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

    # ========================================================
    # SELECT ACTION
    # ========================================================

    @torch.no_grad()
    def select_action(
        self,
        observation,
    ):

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        (
            action,
            log_probability,
            entropy,
            value,
        ) = self.network.get_action_and_value(
            observation_tensor
        )

        return (
            int(action.item()),
            float(log_probability.item()),
            float(value.item()),
        )

    # ========================================================
    # COMPUTE GAE
    # ========================================================

    
    def compute_gae(
        self,
        rewards,
        values,
        dones,
        next_value,
        terminated=None,
        next_values=None
    ):
        """
        Compute Generalized Advantage Estimation.

        dones: episode ended (terminated OR truncated).
        terminated: true terminal states only.
        next_values: critic values of each transition's actual next observation.
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        if not (len(rewards) == len(values) == len(dones)):
            raise ValueError(
                "rewards, values and dones must have equal lengths."
            )

        if terminated is None:
            terminated = dones.copy()
        else:
            terminated = np.asarray(terminated, dtype=np.float32)

        if len(terminated) != len(rewards):
            raise ValueError(
                "terminated must have the same length as rewards."
            )

        if next_values is None:
            next_values = np.concatenate(
                [values[1:], np.asarray([next_value], dtype=np.float32)]
            )
        else:
            next_values = np.asarray(next_values, dtype=np.float32)

        if len(next_values) != len(rewards):
            raise ValueError(
                "next_values must have the same length as rewards."
            )

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for step in reversed(range(len(rewards))):
            # Bootstrap unless this is a true terminal state.
            delta = (
                rewards[step]
                + self.gamma * next_values[step]
                * (1.0 - terminated[step])
                - values[step]
            )

            # Stop GAE recursion at either termination or truncation.
            gae = (
                delta
                + self.gamma * self.gae_lambda
                * (1.0 - dones[step]) * gae
            )

            advantages[step] = gae

        returns = advantages + values
        return advantages, returns

    # ========================================================
    # PPO UPDATE
    # ========================================================

    def update(
        self,
        observations,
        actions,
        old_log_probabilities,
        advantages,
        returns,
    ):

        observations = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        old_log_probabilities = torch.as_tensor(
            old_log_probabilities,
            dtype=torch.float32,
            device=self.device,
        )

        advantages = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # NORMALIZE ADVANTAGES
        # ----------------------------------------------------

        advantages = (
            advantages
            - advantages.mean()
        ) / (
            advantages.std()
            + 1e-8
        )

        dataset_size = (
            len(observations)
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        update_count = 0

        # ----------------------------------------------------
        # PPO EPOCHS
        # ----------------------------------------------------

        for _ in range(
            self.update_epochs
        ):

            indices = np.random.permutation(
                dataset_size
            )

            # ------------------------------------------------
            # MINI-BATCHES
            # ------------------------------------------------

            for start in range(
                0,
                dataset_size,
                self.batch_size,
            ):

                end = min(
                    start
                    + self.batch_size,
                    dataset_size,
                )

                batch_indices = indices[
                    start:end
                ]

                batch_observations = (
                    observations[
                        batch_indices
                    ]
                )

                batch_actions = (
                    actions[
                        batch_indices
                    ]
                )

                batch_old_log_probs = (
                    old_log_probabilities[
                        batch_indices
                    ]
                )

                batch_advantages = (
                    advantages[
                        batch_indices
                    ]
                )

                batch_returns = (
                    returns[
                        batch_indices
                    ]
                )

                # ------------------------------------------------
                # CURRENT POLICY
                # ------------------------------------------------

                (
                    _,
                    new_log_probs,
                    entropy,
                    values,
                ) = self.network.get_action_and_value(
                    batch_observations,
                    batch_actions,
                )

                # ------------------------------------------------
                # POLICY RATIO
                # ------------------------------------------------

                ratio = torch.exp(
                    new_log_probs
                    -
                    batch_old_log_probs
                )

                # ------------------------------------------------
                # CLIPPED PPO OBJECTIVE
                # ------------------------------------------------

                unclipped_objective = (
                    ratio
                    * batch_advantages
                )

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0
                    - self.clip_epsilon,
                    1.0
                    + self.clip_epsilon,
                )

                clipped_objective = (
                    clipped_ratio
                    * batch_advantages
                )

                policy_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                # ------------------------------------------------
                # VALUE LOSS
                # ------------------------------------------------

                value_loss = (
                    (values - batch_returns)
                    ** 2
                ).mean()

                # ------------------------------------------------
                # ENTROPY
                # ------------------------------------------------

                entropy_mean = (
                    entropy.mean()
                )

                # ------------------------------------------------
                # TOTAL LOSS
                # ------------------------------------------------

                loss = (
                    policy_loss
                    +
                    self.value_coefficient
                    * value_loss
                    -
                    self.entropy_coefficient
                    * entropy_mean
                )

                # ------------------------------------------------
                # OPTIMIZATION
                # ------------------------------------------------

                self.optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                # ------------------------------------------------
                # LOGGING
                # ------------------------------------------------

                total_policy_loss += (
                    policy_loss.item()
                )

                total_value_loss += (
                    value_loss.item()
                )

                total_entropy += (
                    entropy_mean.item()
                )

                update_count += 1

        # ----------------------------------------------------
        # RETURN TRAINING STATISTICS
        # ----------------------------------------------------

        if update_count == 0:

            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            }

        return {

            "policy_loss":
                total_policy_loss
                / update_count,

            "value_loss":
                total_value_loss
                / update_count,

            "entropy":
                total_entropy
                / update_count,
        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PPO NETWORK TEST")
    print("=" * 60)

    observation_dim = 36

    action_dim = 25

    agent = PPOAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
    )

    print(
        "Observation dimension:",
        observation_dim,
    )

    print(
        "Action dimension:",
        action_dim,
    )

    print(
        "Device:",
        agent.device,
    )

    # Random observation
    observation = np.random.rand(
        observation_dim
    ).astype(
        np.float32
    )

    action, log_prob, value = (
        agent.select_action(
            observation
        )
    )

    print()
    print(
        "Sampled action:",
        action,
    )

    print(
        "Log probability:",
        log_prob,
    )

    print(
        "State value:",
        value,
    )

    print()
    print(
        "PPO network test successful."
    )