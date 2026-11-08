"""
Modular Value (Critic) Network for PPO and Generalized Advantage Estimation.
"""

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    State-Value Estimator V(s).
    """

    def __init__(self, observation_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)
