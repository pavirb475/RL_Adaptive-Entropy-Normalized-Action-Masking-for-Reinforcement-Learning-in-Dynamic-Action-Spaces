"""
Visual Truck Dispatching Environment (Gymnasium compatible).
Couples tile-based open-pit mine simulation, realistic routing, and action masking
with standard RL observation, reward, and step interfaces.
"""

from typing import Optional, Dict, Any, Tuple, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environments.grid_map import GridMap
from environments.pathfinding import PathCache
from environments.truck import Truck, TruckStatus
from environments.jobs import JobManager, Job
from environments.action_mask import compute_action_mask, is_dispatch_valid


class VisualTruckDispatchEnv(gym.Env):
    """
    Open-Pit Mine Visual Dispatch Environment.
    Supports 4 Trucks and 6 Job slots on a 2D tile grid.
    Features state-dependent action masking and an explicit WAIT action.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        grid_map: Optional[GridMap] = None,
        num_trucks: int = 4,
        num_jobs: int = 6,
        max_steps: int = 100,
        job_arrival_probability: float = 0.30,
        min_deadline: int = 15,
        max_deadline: int = 45,
        truck_capacity_min: float = 0.6,
        truck_capacity_max: float = 1.0,
        truck_load_duration: int = 2,
        truck_dump_duration: int = 2,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.num_trucks = num_trucks
        self.num_jobs = num_jobs
        self.max_steps = max_steps
        self.job_arrival_probability = job_arrival_probability
        self.min_deadline = min_deadline
        self.max_deadline = max_deadline
        self.truck_capacity_min = truck_capacity_min
        self.truck_capacity_max = truck_capacity_max
        self.truck_load_duration = truck_load_duration
        self.truck_dump_duration = truck_dump_duration
        self.render_mode = render_mode

        # Random generator
        self.rng = np.random.default_rng(seed)

        # World Grid & Path Engine
        self.grid_map = grid_map if grid_map is not None else GridMap.create_default_map()
        self.path_cache = PathCache(self.grid_map)

        # Action Space: (num_trucks * num_jobs) dispatch actions + 1 WAIT action
        self.num_dispatch_actions = self.num_trucks * self.num_jobs
        self.wait_action = self.num_dispatch_actions
        self.num_actions = self.num_dispatch_actions + 1
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation Space: (num_trucks * 3) + (num_jobs * 4) features
        observation_size = self.num_trucks * 3 + self.num_jobs * 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )

        # Initialize Truck fleet
        self.trucks: List[Truck] = []
        self._init_trucks()

        # Job manager
        self.job_manager = JobManager(
            num_jobs=self.num_jobs,
            arrival_probability=self.job_arrival_probability,
            min_deadline=self.min_deadline,
            max_deadline=self.max_deadline,
        )

        # Simulation metrics
        self.current_step = 0
        self.completed_jobs = 0
        self.total_waiting_time = 0.0
        self.total_travel_cost = 0.0
        self.total_expired_jobs = 0
        self.total_truck_busy_steps = 0
        self.total_truck_possible_steps = 0
        self.job_completion_times: List[float] = []
        self.job_waiting_times: List[float] = []

        # Pygame visualizer instance (lazy loaded)
        self.visualizer = None

    def _init_trucks(self):
        self.trucks = []
        spawns = self.grid_map.truck_spawns
        for t in range(self.num_trucks):
            spawn = spawns[t % len(spawns)] if spawns else (11, 7)
            cap = float(self.rng.uniform(self.truck_capacity_min, self.truck_capacity_max))
            truck = Truck(truck_id=t, spawn_pos=spawn, capacity=cap)
            truck.load_duration = self.truck_load_duration
            truck.dump_duration = self.truck_dump_duration
            self.trucks.append(truck)


    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.completed_jobs = 0
        self.total_waiting_time = 0.0
        self.total_travel_cost = 0.0
        self.total_expired_jobs = 0
        self.total_truck_busy_steps = 0
        self.total_truck_possible_steps = 0
        self.job_completion_times = []
        self.job_waiting_times = []

        # Reset trucks
        spawns = self.grid_map.truck_spawns
        for t, truck in enumerate(self.trucks):
            spawn = spawns[t % len(spawns)] if spawns else (11, 7)
            cap = float(self.rng.uniform(self.truck_capacity_min, self.truck_capacity_max))
            truck.load_duration = self.truck_load_duration
            truck.dump_duration = self.truck_dump_duration
            truck.reset(spawn_pos=spawn, capacity=cap)

        # Reset jobs
        self.job_manager.reset(rng=self.rng, grid_map=self.grid_map)

        observation = self._get_observation()
        mask = self.action_masks()

        info = {
            "action_mask": mask,
            "num_valid_actions": int(np.sum(mask)),
            "completed_jobs": self.completed_jobs,
            "expired_jobs": 0,
            "total_expired_jobs": 0,
            "fleet_utilization": 0.0,
            "avg_waiting_time": 0.0,
            "avg_completion_time": 0.0,
            "total_travel_distance": 0,
        }

        return observation, info


    def decode_action(self, action: int) -> Tuple[Optional[int], Optional[int]]:
        action = int(action)
        if action == self.wait_action:
            return None, None
        truck = action // self.num_jobs
        job = action % self.num_jobs
        return truck, job

    def is_action_valid(self, truck_idx: int, job_idx: int) -> bool:
        if not (0 <= truck_idx < self.num_trucks and 0 <= job_idx < self.num_jobs):
            return False

        truck = self.trucks[truck_idx]
        job = self.job_manager.get_job(job_idx)

        # 1. Truck availability
        if not truck.is_available:
            return False

        # 2. Job existence and pending
        if job is None or not job.is_pending:
            return False

        # 3. Capacity feasibility
        if truck.capacity < job.load_requirement:
            return False

        # 4. Expiry deadline
        if self.current_step > job.deadline:
            return False

        return True

    def action_masks(self) -> np.ndarray:
        """
        Calculates action mask of size 25.
        WAIT action is valid iff NO dispatch actions are feasible.
        """
        mask = np.zeros(self.num_actions, dtype=np.int8)

        # Check all dispatch pairs
        for t in range(self.num_trucks):
            for j in range(self.num_jobs):
                action = t * self.num_jobs + j
                if self.is_action_valid(t, j):
                    mask[action] = 1

        # WAIT is valid ONLY when no dispatch actions are feasible
        if not np.any(mask[:self.num_dispatch_actions]):
            mask[self.wait_action] = 1

        return mask

    def num_valid_actions(self) -> int:
        return int(np.sum(self.action_masks()))

    def _get_observation(self) -> np.ndarray:
        """
        Constructs normalized 36-dimensional observation vector.
        """
        obs: List[float] = []

        # 1. Truck features: (avail, normalized_x, capacity)
        for truck in self.trucks:
            obs.append(1.0 if truck.is_available else 0.0)
            obs.append(float(truck.x) / float(max(1, self.grid_map.width)))
            obs.append(float(truck.capacity))

        # 2. Job features: (pending, normalized_load_x, load_req, normalized_deadline)
        for slot in range(self.num_jobs):
            job = self.job_manager.get_job(slot)
            if job is not None and job.is_pending:
                obs.append(1.0)
                obs.append(float(job.loading_pos[0]) / float(max(1, self.grid_map.width)))
                obs.append(float(job.load_requirement))
                obs.append(job.normalized_deadline(self.current_step, max_deadline_horizon=40.0))
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.asarray(obs, dtype=np.float32)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} outside valid range [0, {self.num_actions - 1}]")

        current_mask = self.action_masks()

        # ------------------------------------------------------------------
        # 1. INVALID ACTION TAKEN
        # ------------------------------------------------------------------
        if current_mask[action] == 0:
            reward = -5.0
            self.current_step += 1

            expired_count = self._advance_world()
            expiry_penalty = 3.0 * expired_count
            reward -= expiry_penalty

            terminated = False
            truncated = self.current_step >= self.max_steps
            next_mask = self.action_masks()

            info = {
                "valid_action": False,
                "wait_action": False,
                "action_mask": next_mask,
                "num_valid_actions": int(np.sum(next_mask)),
                "completed_jobs": self.completed_jobs,
                "invalid_action_penalty": 5.0,
                "expired_jobs": expired_count,
                "total_expired_jobs": self.total_expired_jobs,
                "fleet_utilization": self.total_truck_busy_steps / max(1, self.total_truck_possible_steps),
                "avg_waiting_time": float(np.mean(self.job_waiting_times)) if self.job_waiting_times else 0.0,
                "avg_completion_time": float(np.mean(self.job_completion_times)) if self.job_completion_times else 0.0,
                "total_travel_distance": int(sum(t.total_distance_traveled for t in self.trucks)),
            }
            return self._get_observation(), reward, terminated, truncated, info

        # ------------------------------------------------------------------
        # 2. VALID WAIT ACTION
        # ------------------------------------------------------------------
        if action == self.wait_action:
            self.current_step += 1
            expired_count = self._advance_world()
            expiry_penalty = 3.0 * expired_count

            reward = -0.1 - expiry_penalty
            terminated = False
            truncated = self.current_step >= self.max_steps
            next_mask = self.action_masks()

            info = {
                "valid_action": True,
                "wait_action": True,
                "truck": None,
                "job": None,
                "travel_time": 0,
                "waiting_time": 0,
                "reward": reward,
                "completed_jobs": self.completed_jobs,
                "action_mask": next_mask,
                "num_valid_actions": int(np.sum(next_mask)),
                "expired_jobs": expired_count,
                "total_expired_jobs": self.total_expired_jobs,
                "fleet_utilization": self.total_truck_busy_steps / max(1, self.total_truck_possible_steps),
                "avg_waiting_time": float(np.mean(self.job_waiting_times)) if self.job_waiting_times else 0.0,
                "avg_completion_time": float(np.mean(self.job_completion_times)) if self.job_completion_times else 0.0,
                "total_travel_distance": int(sum(t.total_distance_traveled for t in self.trucks)),
            }
            return self._get_observation(), reward, terminated, truncated, info

        # ------------------------------------------------------------------
        # 3. VALID DISPATCH ACTION
        # ------------------------------------------------------------------
        truck_idx, job_idx = self.decode_action(action)
        truck = self.trucks[truck_idx]
        job = self.job_manager.get_job(job_idx)

        # Plan paths using GridMap Dijkstra/A*
        path_to_load = self.path_cache.get_path((truck.x, truck.y), job.loading_pos)
        if path_to_load is None:
            path_to_load = [(truck.x, truck.y), job.loading_pos]

        path_to_dump = self.path_cache.get_path(job.loading_pos, job.dumping_pos)
        if path_to_dump is None:
            path_to_dump = [job.loading_pos, job.dumping_pos]

        travel_time = (
            max(1, len(path_to_load) - 1)
            + truck.load_duration
            + max(1, len(path_to_dump) - 1)
            + truck.dump_duration
        )
        waiting_time = max(0, self.current_step - job.arrival_step)

        # Assign mission to truck
        truck.assign_dispatch(
            job_id=job.job_id,
            load_pos=job.loading_pos,
            dump_pos=job.dumping_pos,
            path_to_load=path_to_load,
            path_to_dump=path_to_dump,
            current_step=self.current_step,
            total_travel_time=travel_time,
        )

        # Base delivery reward scaled by ore tonnage requirement
        payload_req = float(job.load_requirement)
        base_reward = 10.0 * payload_req

        # Clear pending status from slot and mark completed counter
        self.job_manager.mark_completed(job_idx)
        self.completed_jobs += 1
        self.total_waiting_time += waiting_time
        self.total_travel_cost += travel_time
        self.job_waiting_times.append(float(waiting_time))
        self.job_completion_times.append(float(waiting_time + travel_time))

        # Reward: delivery value - waiting backlog cost - travel cost
        reward = base_reward - (0.5 * waiting_time) - (0.2 * travel_time)

        self.current_step += 1
        expired_count = self._advance_world()
        expiry_penalty = 3.0 * expired_count
        reward -= expiry_penalty

        terminated = False
        truncated = self.current_step >= self.max_steps
        next_mask = self.action_masks()

        info = {
            "valid_action": True,
            "wait_action": False,
            "truck": truck_idx,
            "job": job_idx,
            "payload": payload_req,
            "travel_time": travel_time,
            "waiting_time": waiting_time,
            "reward": reward,
            "completed_jobs": self.completed_jobs,
            "expired_jobs": expired_count,
            "total_expired_jobs": self.total_expired_jobs,
            "fleet_utilization": self.total_truck_busy_steps / max(1, self.total_truck_possible_steps),
            "avg_waiting_time": float(np.mean(self.job_waiting_times)) if self.job_waiting_times else 0.0,
            "avg_completion_time": float(np.mean(self.job_completion_times)) if self.job_completion_times else 0.0,
            "total_travel_distance": int(sum(t.total_distance_traveled for t in self.trucks)),
            "action_mask": next_mask,
            "num_valid_actions": int(np.sum(next_mask)),
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _advance_world(self) -> int:
        """Advances physical truck motion, loading/dumping ticks, and job arrivals."""
        for truck in self.trucks:
            if not truck.is_available:
                self.total_truck_busy_steps += 1
            self.total_truck_possible_steps += 1
            truck.fast_forward_step()

        expired_count = self.job_manager.update_arrivals_and_expiries(
            current_step=self.current_step,
            rng=self.rng,
            grid_map=self.grid_map,
        )
        self.total_expired_jobs += int(expired_count)
        return int(expired_count)

    def render(self):
        if self.render_mode == "human":
            if self.visualizer is None:
                from visualization.simulator_window import SimulatorWindow
                self.visualizer = SimulatorWindow(env=self)
            self.visualizer.render_frame()

    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None
