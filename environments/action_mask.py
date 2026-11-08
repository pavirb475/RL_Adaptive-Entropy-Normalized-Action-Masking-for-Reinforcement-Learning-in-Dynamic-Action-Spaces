"""
Action masking engine for Dynamic Truck Dispatching.
Authoritative implementation for checking feasibility of dispatch and WAIT actions.
"""

from typing import List, Tuple, Optional
import numpy as np


def is_dispatch_valid(
    truck_idx: int,
    job_idx: int,
    truck_available: np.ndarray,
    truck_capacities: np.ndarray,
    job_pending: np.ndarray,
    job_loads: np.ndarray,
    job_deadlines: np.ndarray,
    current_step: int,
) -> bool:
    """
    Check if a specific truck-job assignment is feasible.
    """
    # Boundary checks
    if truck_idx < 0 or truck_idx >= len(truck_available):
        return False
    if job_idx < 0 or job_idx >= len(job_pending):
        return False

    # 1. Truck availability check
    if truck_available[truck_idx] == 0:
        return False

    # 2. Job pending check
    if job_pending[job_idx] == 0:
        return False

    # 3. Truck capacity check
    if truck_capacities[truck_idx] < job_loads[job_idx]:
        return False

    # 4. Job deadline check
    if current_step > job_deadlines[job_idx]:
        return False

    return True


def compute_action_mask(
    num_trucks: int,
    num_jobs: int,
    truck_available: np.ndarray,
    truck_capacities: np.ndarray,
    job_pending: np.ndarray,
    job_loads: np.ndarray,
    job_deadlines: np.ndarray,
    current_step: int,
) -> np.ndarray:
    """
    Construct the binary action mask of size (num_trucks * num_jobs + 1).

    The first (num_trucks * num_jobs) actions represent dispatch assignments.
    The final action represents WAIT (no-op).

    Rule: WAIT is valid ONLY when NO dispatch actions are currently feasible.
    """
    num_dispatch_actions = num_trucks * num_jobs
    total_actions = num_dispatch_actions + 1
    mask = np.zeros(total_actions, dtype=np.int8)

    # Check each possible dispatch action
    for t in range(num_trucks):
        for j in range(num_jobs):
            action_idx = t * num_jobs + j
            if is_dispatch_valid(
                truck_idx=t,
                job_idx=j,
                truck_available=truck_available,
                truck_capacities=truck_capacities,
                job_pending=job_pending,
                job_loads=job_loads,
                job_deadlines=job_deadlines,
                current_step=current_step,
            ):
                mask[action_idx] = 1

    # Check WAIT action validity
    # WAIT is valid only if zero dispatch actions are feasible
    if not np.any(mask[:num_dispatch_actions]):
        mask[num_dispatch_actions] = 1

    return mask


def decode_action(action: int, num_jobs: int, num_trucks: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    """
    Decode discrete action into (truck_id, job_id).
    Returns (None, None) for the WAIT action.
    """
    action = int(action)
    if num_trucks is not None and action >= num_trucks * num_jobs:
        return None, None
    truck = action // num_jobs
    job = action % num_jobs
    return truck, job



def get_valid_actions(mask: np.ndarray) -> np.ndarray:
    """Return indices of currently valid actions."""
    return np.flatnonzero(mask)


def count_valid_actions(mask: np.ndarray) -> int:
    """Return total number of valid actions |A_valid|."""
    return int(np.sum(mask))
