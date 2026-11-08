"""
Unit tests for evaluation metrics module.
"""

import numpy as np
import pytest
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


def test_return_statistics():
    returns = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_return_statistics(returns)
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["ci_95"] > 0.0


def test_success_rate():
    returns = [15.0, -2.0, 25.0, 30.0]
    completed = [5, 2, 6, 7]
    expired = [0, 2, 0, 1]  # Only episodes 0 and 2 are successful
    sr = compute_success_rate(returns, completed, expired, return_threshold=0.0)
    assert sr == 0.5


def test_cumulative_regret():
    returns = [20.0, 25.0, 30.0]
    oracle = 30.0
    regret = compute_cumulative_regret(returns, oracle)
    assert regret == [10.0, 15.0, 15.0]


def test_convergence_speed():
    curve = [5.0, 10.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
    speed = compute_convergence_speed(curve, threshold=20.0, consecutive_episodes=5)
    assert speed == 3  # Converged starting at index 2 (1-based episode 3)


def test_paired_significance():
    a = [25.0, 28.0, 32.0, 30.0, 29.0]
    b = [15.0, 18.0, 20.0, 19.0, 17.0]
    res = compute_paired_significance(a, b)
    assert res["p_value_t"] < 0.01
    assert res["cohens_d"] > 2.0


def test_mine_dispatch_metrics():
    res = compute_mine_dispatch_metrics(
        completed_jobs=[5, 6, 7],
        expired_jobs=[1, 0, 1],
        waiting_times=[3.5, 4.0, 2.5],
        completion_times=[12.0, 14.0, 11.0],
        fleet_utilizations=[0.75, 0.80, 0.85],
        travel_distances=[120, 135, 140],
        total_steps=300,
    )
    assert res["total_deliveries"] == 18
    assert res["job_completion_rate_pct"] == 90.0
    assert res["fleet_utilization_pct"] == 80.0
