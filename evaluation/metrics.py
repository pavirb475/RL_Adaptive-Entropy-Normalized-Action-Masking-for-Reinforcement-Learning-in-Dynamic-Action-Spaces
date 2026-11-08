"""
Comprehensive Evaluation Metrics Library for AEN-PPO Research Project.
Implements every metric specified in PPTperformance.pdf:
- RL Performance: Return (mean, median, std, 95% CI), Success Rate, Cumulative Regret,
  Convergence Speed, Sample Efficiency, Generalization Gap, Stability Across Seeds,
  Computational Efficiency (latency, training throughput, parameters).
- Action Masking: Invalid Action Count & Rate, |A_valid|, Raw Entropy, Normalized Entropy.
- Mine Dispatching: Completed Deliveries / Throughput, Waiting Time, Completion Time,
  Fleet Utilization, Travel Distance, Deadline Violations, Job Completion Rate.
- Statistical Significance: Paired t-test, Wilcoxon signed-rank test, Cohen's d effect size.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import scipy.stats as stats


def compute_return_statistics(returns: List[float]) -> Dict[str, float]:
    """
    Computes Mean, Median, Standard Deviation, and 95% Confidence Interval.
    Formula for 95% CI: mean +/- 1.96 * (std / sqrt(N))
    """
    arr = np.asarray(returns, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "ci_95": 0.0, "min": 0.0, "max": 0.0}

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std_val / np.sqrt(n) if n > 1 else 0.0
    ci_95 = 1.96 * sem

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "ci_95": ci_95,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_success_rate(
    returns: List[float],
    completed_jobs: List[int],
    expired_jobs: List[int],
    return_threshold: float = 0.0,
    min_completed: int = 1,
) -> float:
    """
    Success Rate (SR) = N_success / N_total.
    An episode is successful if return >= threshold, at least min_completed jobs delivered,
    and 0 expired deadline failures.
    """
    n = len(returns)
    if n == 0:
        return 0.0

    success_count = 0
    for r, c, e in zip(returns, completed_jobs, expired_jobs):
        if r >= return_threshold and c >= min_completed and e == 0:
            success_count += 1

    return float(success_count / n)


def compute_cumulative_regret(returns: List[float], oracle_benchmark: float) -> List[float]:
    """
    Cumulative Regret: R_N = sum_{i=1}^N (J* - J^pi_i)
    where J* is the top benchmark or oracle return for this scenario.
    """
    regrets = []
    cumulative = 0.0
    for r in returns:
        step_regret = max(0.0, oracle_benchmark - r)
        cumulative += step_regret
        regrets.append(cumulative)
    return regrets


def compute_convergence_speed(
    learning_curve: List[float],
    threshold: float,
    consecutive_episodes: int = 5,
) -> Optional[int]:
    """
    Convergence Speed: N* = min { N : G_bar_N >= G_th for K consecutive evaluations }
    Returns episode number N*, or None if threshold was not sustained.
    """
    count = 0
    for idx, r in enumerate(learning_curve):
        if r >= threshold:
            count += 1
            if count >= consecutive_episodes:
                return idx - consecutive_episodes + 2
        else:
            count = 0
    return None


def compute_sample_efficiency(
    learning_curve: List[float],
    steps_per_episode: int = 100,
    milestone: float = 20.0,
) -> Dict[str, Any]:
    """
    Sample Efficiency:
    - Steps to reach target return milestone
    - Rate of return increase per 1,000 environment steps (slope over early phase)
    """
    total_steps = len(learning_curve) * steps_per_episode
    milestone_step = None

    for idx, val in enumerate(learning_curve):
        if val >= milestone:
            milestone_step = (idx + 1) * steps_per_episode
            break

    # Early phase slope (first 30% of training)
    early_len = max(5, int(len(learning_curve) * 0.3))
    x = np.arange(early_len) * steps_per_episode / 1000.0  # per 1k steps
    y = np.array(learning_curve[:early_len])
    if len(x) > 1 and np.std(x) > 0:
        slope, _, _, _, _ = stats.linregress(x, y)
    else:
        slope = 0.0

    return {
        "steps_to_milestone": milestone_step,
        "return_growth_per_1k_steps": float(slope),
    }


def compute_generalization_gap(
    seen_returns: List[float],
    unseen_returns: List[float],
) -> float:
    """
    Generalization Gap: Delta_gen = J_seen - J_unseen
    """
    return float(np.mean(seen_returns) - np.mean(unseen_returns))


def compute_stability_across_seeds(seed_means: List[float]) -> Dict[str, float]:
    """
    Stability Across Seeds:
    - Standard deviation across training seed means
    - 95% Confidence Interval width across seeds
    """
    arr = np.asarray(seed_means, dtype=np.float64)
    s = len(arr)
    if s < 2:
        return {"seed_std": 0.0, "ci_95_width": 0.0}

    seed_std = float(np.std(arr, ddof=1))
    ci_95_width = float(2 * 1.96 * (seed_std / np.sqrt(s)))
    return {
        "seed_std": seed_std,
        "ci_95_width": ci_95_width,
    }


def compute_paired_significance(
    sample_a: List[float],
    sample_b: List[float],
) -> Dict[str, Any]:
    """
    Performs Paired Student's t-test, Wilcoxon signed-rank test, and Cohen's d effect size.
    """
    arr_a = np.asarray(sample_a, dtype=np.float64)
    arr_b = np.asarray(sample_b, dtype=np.float64)

    if len(arr_a) != len(arr_b) or len(arr_a) < 3:
        return {
            "t_stat": 0.0, "p_value_t": 1.0,
            "wilcoxon_stat": 0.0, "p_value_wilcoxon": 1.0,
            "cohens_d": 0.0,
        }

    # Paired t-test
    t_stat, p_val_t = stats.ttest_rel(arr_a, arr_b)

    # Wilcoxon signed-rank test
    diffs = arr_a - arr_b
    if np.all(diffs == 0):
        w_stat, p_val_w = 0.0, 1.0
    else:
        try:
            w_stat, p_val_w = stats.wilcoxon(arr_a, arr_b)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

    # Cohen's d effect size for paired samples: mean(diff) / std(diff)
    diff_std = np.std(diffs, ddof=1)
    cohens_d = float(np.mean(diffs) / diff_std) if diff_std > 1e-8 else 0.0

    return {
        "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_value_t": float(p_val_t) if not np.isnan(p_val_t) else 1.0,
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else 0.0,
        "p_value_wilcoxon": float(p_val_w) if not np.isnan(p_val_w) else 1.0,
        "cohens_d": cohens_d,
    }


def compute_mine_dispatch_metrics(
    completed_jobs: List[int],
    expired_jobs: List[int],
    waiting_times: List[float],
    completion_times: List[float],
    fleet_utilizations: List[float],
    travel_distances: List[int],
    total_steps: int,
) -> Dict[str, float]:
    """
    Computes all operational open-pit mine truck dispatching metrics:
    - Mean completed deliveries / throughput
    - Job completion rate (%)
    - Average waiting time
    - Average completion/cycle time
    - Fleet utilization rate (%)
    - Average haulage distance
    - Missed deadline rate (%)
    """
    tot_completed = sum(completed_jobs)
    tot_expired = sum(expired_jobs)
    tot_orders = tot_completed + tot_expired

    completion_rate = (tot_completed / tot_orders * 100.0) if tot_orders > 0 else 100.0
    missed_rate = (tot_expired / tot_orders * 100.0) if tot_orders > 0 else 0.0

    return {
        "mean_deliveries": float(np.mean(completed_jobs)) if completed_jobs else 0.0,
        "total_deliveries": int(tot_completed),
        "mean_expired_jobs": float(np.mean(expired_jobs)) if expired_jobs else 0.0,
        "job_completion_rate_pct": float(completion_rate),
        "missed_deadline_rate_pct": float(missed_rate),
        "avg_waiting_time": float(np.mean(waiting_times)) if waiting_times else 0.0,
        "avg_completion_time": float(np.mean(completion_times)) if completion_times else 0.0,
        "fleet_utilization_pct": float(np.mean(fleet_utilizations) * 100.0) if fleet_utilizations else 0.0,
        "mean_travel_distance": float(np.mean(travel_distances)) if travel_distances else 0.0,
    }


def compute_action_exploration_metrics(
    invalid_counts: List[int],
    total_action_counts: List[int],
    mean_valid_counts: List[float],
    raw_entropies: List[float],
    normalized_entropies: List[float],
) -> Dict[str, float]:
    """
    Computes action masking and exploration telemetry:
    - Invalid action count
    - Invalid action rate (%)
    - Mean |A_valid|
    - Raw policy entropy H(pi)
    - Normalized policy entropy H_norm(pi)
    - Entropy variation (std dev)
    """
    tot_invalid = sum(invalid_counts)
    tot_actions = sum(total_action_counts)
    invalid_rate = (tot_invalid / tot_actions * 100.0) if tot_actions > 0 else 0.0

    return {
        "mean_invalid_actions": float(np.mean(invalid_counts)) if invalid_counts else 0.0,
        "invalid_action_rate_pct": float(invalid_rate),
        "mean_valid_action_count": float(np.mean(mean_valid_counts)) if mean_valid_counts else 0.0,
        "mean_raw_entropy": float(np.mean(raw_entropies)) if raw_entropies else 0.0,
        "mean_normalized_entropy": float(np.mean(normalized_entropies)) if normalized_entropies else 0.0,
        "entropy_std": float(np.std(raw_entropies)) if len(raw_entropies) > 1 else 0.0,
    }
