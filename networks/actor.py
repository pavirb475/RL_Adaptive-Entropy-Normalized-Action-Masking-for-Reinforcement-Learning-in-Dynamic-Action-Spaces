"""
Modular Policy (Actor) Network for PPO and Masked PPO.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    """
    Categorical Actor Head for Discrete Action Spaces with optional Action Masking.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim

    def forward(self, observation: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Categorical:
        features = self.feature_extractor(observation)
        logits = self.action_head(features)

        if mask is not None:
            mask = mask.bool()
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        return Categorical(logits=logits)

    def get_action(
        self, observation: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(observation, mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
