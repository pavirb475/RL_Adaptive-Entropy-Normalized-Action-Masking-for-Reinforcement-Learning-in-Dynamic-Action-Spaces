"""
Unit tests for action masking and WAIT condition logic.
"""

import numpy as np
from environments.action_mask import (
    compute_action_mask,
    is_dispatch_valid,
    get_valid_actions,
    count_valid_actions,
)


def test_action_mask_feasibility():
    num_trucks = 4
    num_jobs = 6
    current_step = 10

    truck_available = np.array([1, 0, 1, 0], dtype=np.int8)
    truck_capacities = np.array([0.8, 1.0, 0.5, 0.9], dtype=np.float32)

    job_pending = np.array([1, 1, 0, 1, 0, 1], dtype=np.int8)
    job_loads = np.array([0.6, 0.9, 0.4, 0.7, 0.5, 0.5], dtype=np.float32)
    job_deadlines = np.array([20, 25, 30, 8, 30, 15], dtype=np.int32)  # Job 3 deadline (8) expired!

    # Truck 0 (avail=1, cap=0.8):
    # - Job 0: pending=1, load=0.6 <= 0.8, deadline=20 > 10 -> VALID!
    # - Job 1: pending=1, load=0.9 > 0.8 -> INVALID (capacity)
    # - Job 2: pending=0 -> INVALID
    # - Job 3: deadline=8 < 10 -> INVALID (expired)
    # - Job 4: pending=0 -> INVALID
    # - Job 5: pending=1, load=0.5 <= 0.8, deadline=15 > 10 -> VALID!
    assert is_dispatch_valid(0, 0, truck_available, truck_capacities, job_pending, job_loads, job_deadlines, current_step)
    assert not is_dispatch_valid(0, 1, truck_available, truck_capacities, job_pending, job_loads, job_deadlines, current_step)
    assert not is_dispatch_valid(0, 2, truck_available, truck_capacities, job_pending, job_loads, job_deadlines, current_step)
    assert not is_dispatch_valid(0, 3, truck_available, truck_capacities, job_pending, job_loads, job_deadlines, current_step)

    mask = compute_action_mask(
        num_trucks, num_jobs,
        truck_available, truck_capacities,
        job_pending, job_loads, job_deadlines,
        current_step
    )
    assert len(mask) == 25
    # Since there are valid dispatch actions, WAIT (action 24) must be 0
    assert mask[24] == 0
    assert mask[0 * 6 + 0] == 1  # Truck 0, Job 0
    assert mask[0 * 6 + 5] == 1  # Truck 0, Job 5


def test_wait_action_rule():
    # If no trucks available, WAIT must be the ONLY valid action
    num_trucks = 4
    num_jobs = 6
    truck_available = np.zeros(4, dtype=np.int8)
    truck_capacities = np.ones(4, dtype=np.float32)
    job_pending = np.ones(6, dtype=np.int8)
    job_loads = np.full(6, 0.5, dtype=np.float32)
    job_deadlines = np.full(6, 100, dtype=np.int32)

    mask = compute_action_mask(
        num_trucks, num_jobs,
        truck_available, truck_capacities,
        job_pending, job_loads, job_deadlines,
        current_step=0
    )

    # All dispatch actions are 0, WAIT action is 1
    assert np.all(mask[:24] == 0)
    assert mask[24] == 1
    assert count_valid_actions(mask) == 1
    assert list(get_valid_actions(mask)) == [24]


if __name__ == "__main__":
    test_action_mask_feasibility()
    test_wait_action_rule()
    print("test_action_masks.py PASSED!")
