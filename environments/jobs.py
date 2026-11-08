"""
Job lifecycle management for the Truck Dispatch Environment.
Handles dynamic job generation, shovel assignments, destination crushers, deadlines, and payloads.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from environments.grid_map import GridMap


class Job:
    """
    Represents an ore haulage job order.
    """

    def __init__(
        self,
        job_id: int,
        loading_station_id: int,
        dumping_station_id: int,
        loading_pos: Tuple[int, int],
        dumping_pos: Tuple[int, int],
        load_requirement: float,
        deadline: int,
        arrival_step: int,
    ):
        self.job_id = job_id
        self.loading_station_id = loading_station_id
        self.dumping_station_id = dumping_station_id
        self.loading_pos = (int(loading_pos[0]), int(loading_pos[1]))
        self.dumping_pos = (int(dumping_pos[0]), int(dumping_pos[1]))
        self.load_requirement = float(load_requirement)
        self.deadline = int(deadline)
        self.arrival_step = int(arrival_step)
        
        self.is_pending: bool = True
        self.assigned_truck_id: Optional[int] = None
        self.completed: bool = False

    def is_expired(self, current_step: int) -> bool:
        """Job is expired if current simulation time exceeds its deadline."""
        return self.is_pending and (current_step > self.deadline)

    def remaining_deadline(self, current_step: int) -> int:
        return max(0, self.deadline - current_step)

    def normalized_deadline(self, current_step: int, max_deadline_horizon: float = 40.0) -> float:
        rem = self.remaining_deadline(current_step)
        return min(1.0, float(rem) / max_deadline_horizon)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "loading_station_id": self.loading_station_id,
            "dumping_station_id": self.dumping_station_id,
            "loading_pos": self.loading_pos,
            "dumping_pos": self.dumping_pos,
            "load_requirement": self.load_requirement,
            "deadline": self.deadline,
            "arrival_step": self.arrival_step,
            "is_pending": self.is_pending,
            "assigned_truck_id": self.assigned_truck_id,
        }


class JobManager:
    """
    Manages fixed job slots and stochastic job arrivals.
    """

    def __init__(
        self,
        num_jobs: int = 6,
        arrival_probability: float = 0.30,
        min_deadline: int = 15,
        max_deadline: int = 45,
    ):
        self.num_jobs = num_jobs
        self.arrival_probability = arrival_probability
        self.min_deadline = min_deadline
        self.max_deadline = max_deadline
        self.jobs: List[Optional[Job]] = [None] * num_jobs

    def reset(self, rng: np.random.Generator, grid_map: GridMap):
        self.jobs = [None] * self.num_jobs
        # Generate initial jobs with probability 0.6
        for slot in range(self.num_jobs):
            if rng.random() < 0.6:
                self.jobs[slot] = self._create_random_job(slot, current_step=0, rng=rng, grid_map=grid_map)

    def _create_random_job(
        self, slot_idx: int, current_step: int, rng: np.random.Generator, grid_map: GridMap
    ) -> Job:
        # Pick random loading station
        num_loaders = len(grid_map.loading_stations)
        loader_idx = int(rng.integers(0, max(1, num_loaders)))
        loader = grid_map.loading_stations[loader_idx] if num_loaders > 0 else {"id": 0, "pos": [2, 3]}

        # Pick random dumping station
        num_dumpers = len(grid_map.dumping_stations)
        dumper_idx = int(rng.integers(0, max(1, num_dumpers)))
        dumper = grid_map.dumping_stations[dumper_idx] if num_dumpers > 0 else {"id": 0, "pos": [21, 3]}

        load_req = float(rng.uniform(0.2, 1.0))
        deadline_offset = int(rng.integers(self.min_deadline, self.max_deadline + 1))
        deadline = current_step + deadline_offset

        return Job(
            job_id=slot_idx,
            loading_station_id=loader["id"],
            dumping_station_id=dumper["id"],
            loading_pos=tuple(loader["pos"]),
            dumping_pos=tuple(dumper["pos"]),
            load_requirement=load_req,
            deadline=deadline,
            arrival_step=current_step,
        )

    def update_arrivals_and_expiries(
        self, current_step: int, rng: np.random.Generator, grid_map: GridMap
    ) -> int:
        expired_count = 0
        for slot in range(self.num_jobs):
            job = self.jobs[slot]
            if job is not None and job.is_expired(current_step):
                # Expired job slot is cleared
                self.jobs[slot] = None
                expired_count += 1

            # If slot is empty, new job may arrive
            if self.jobs[slot] is None:
                if rng.random() < self.arrival_probability:
                    self.jobs[slot] = self._create_random_job(
                        slot_idx=slot,
                        current_step=current_step,
                        rng=rng,
                        grid_map=grid_map,
                    )
        return expired_count

    def get_job(self, slot_idx: int) -> Optional[Job]:
        if 0 <= slot_idx < self.num_jobs:
            return self.jobs[slot_idx]
        return None

    def mark_completed(self, slot_idx: int):
        if 0 <= slot_idx < self.num_jobs:
            self.jobs[slot_idx] = None
