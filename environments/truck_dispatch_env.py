import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TruckDispatchEnv(gym.Env):
    """
    Dynamic Truck Dispatching Environment
    =====================================

    Global action space:

        0 ... (num_trucks * num_jobs - 1)
            -> Truck-job assignment actions

        last action
            -> WAIT / NO-OP

    Truck-job actions are dynamically masked according to:

        1. Truck availability
        2. Job availability
        3. Truck capacity
        4. Job deadline

    WAIT is valid ONLY when no feasible truck-job
    assignment exists.

    This environment is designed for comparison of:

        PPO
        Masked PPO
        AEN-PPO
    """

    metadata = {"render_modes": ["human"]}

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        num_trucks=4,
        num_jobs=6,
        max_steps=100,
        job_arrival_probability=0.30,
        seed=None,
    ):

        super().__init__()

        # ------------------------------------------------------
        # PARAMETERS
        # ------------------------------------------------------

        self.num_trucks = num_trucks
        self.num_jobs = num_jobs
        self.max_steps = max_steps

        self.job_arrival_probability = (
            job_arrival_probability
        )

        # ------------------------------------------------------
        # RANDOM GENERATOR
        # ------------------------------------------------------

        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------
        # ACTION SPACE
        # ------------------------------------------------------

        # Truck-job actions
        self.num_dispatch_actions = (
            self.num_trucks * self.num_jobs
        )

        # One additional WAIT action
        self.wait_action = self.num_dispatch_actions

        # Total actions
        self.num_actions = (
            self.num_dispatch_actions + 1
        )

        self.action_space = spaces.Discrete(
            self.num_actions
        )

        # ------------------------------------------------------
        # OBSERVATION SPACE
        # ------------------------------------------------------

        observation_size = (
            self.num_trucks * 3
            +
            self.num_jobs * 4
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )

        # ======================================================
        # TRUCK VARIABLES
        # ======================================================

        self.truck_available = np.zeros(
            self.num_trucks,
            dtype=np.int8,
        )

        self.truck_location = np.zeros(
            self.num_trucks,
            dtype=np.float32,
        )

        self.truck_capacity = np.zeros(
            self.num_trucks,
            dtype=np.float32,
        )

        self.truck_busy_until = np.zeros(
            self.num_trucks,
            dtype=np.int32,
        )

        # ======================================================
        # JOB VARIABLES
        # ======================================================

        self.job_pending = np.zeros(
            self.num_jobs,
            dtype=np.int8,
        )

        self.job_location = np.zeros(
            self.num_jobs,
            dtype=np.float32,
        )

        self.job_load = np.zeros(
            self.num_jobs,
            dtype=np.float32,
        )

        self.job_deadline = np.zeros(
            self.num_jobs,
            dtype=np.int32,
        )

        self.job_arrival_time = np.zeros(
            self.num_jobs,
            dtype=np.int32,
        )

        # ======================================================
        # SIMULATION VARIABLES
        # ======================================================

        self.current_step = 0

        self.completed_jobs = 0

        self.total_waiting_time = 0.0

        self.total_travel_cost = 0.0

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        # Reinitialize RNG when a seed is provided.
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------
        # RESET SIMULATION
        # ------------------------------------------------------

        self.current_step = 0

        self.completed_jobs = 0

        self.total_waiting_time = 0.0

        self.total_travel_cost = 0.0

        # ------------------------------------------------------
        # RESET TRUCKS
        # ------------------------------------------------------

        self.truck_available[:] = 1

        self.truck_location = (
            self.rng.uniform(
                0.0,
                1.0,
                self.num_trucks,
            ).astype(np.float32)
        )

        self.truck_capacity = (
            self.rng.uniform(
                0.6,
                1.0,
                self.num_trucks,
            ).astype(np.float32)
        )

        self.truck_busy_until[:] = 0

        # ------------------------------------------------------
        # RESET JOBS
        # ------------------------------------------------------

        self.job_pending[:] = 0

        self.job_location[:] = 0.0

        self.job_load[:] = 0.0

        self.job_deadline[:] = 0

        self.job_arrival_time[:] = 0

        # ------------------------------------------------------
        # INITIAL JOB GENERATION
        # ------------------------------------------------------

        for job in range(self.num_jobs):

            if self.rng.random() < 0.6:

                self._create_job(job)

        # ------------------------------------------------------
        # INITIAL STATE
        # ------------------------------------------------------

        observation = self._get_observation()

        mask = self.action_masks()

        info = {
            "action_mask": mask,
            "num_valid_actions": int(
                np.sum(mask)
            ),
            "completed_jobs":
                self.completed_jobs,
        }

        return observation, info

    # ==========================================================
    # CREATE JOB
    # ==========================================================

    def _create_job(self, job):

        self.job_pending[job] = 1

        self.job_location[job] = (
            self.rng.uniform(
                0.0,
                1.0,
            )
        )

        self.job_load[job] = (
            self.rng.uniform(
                0.2,
                1.0,
            )
        )

        deadline_offset = self.rng.integers(
            10,
            40,
        )

        self.job_deadline[job] = (
            self.current_step
            + int(deadline_offset)
        )

        self.job_arrival_time[job] = (
            self.current_step
        )

    # ==========================================================
    # UPDATE TRUCKS
    # ==========================================================

    def _update_trucks(self):

        for truck in range(self.num_trucks):

            if (
                self.truck_available[truck] == 0
                and
                self.current_step
                >= self.truck_busy_until[truck]
            ):

                self.truck_available[truck] = 1

    # ==========================================================
    # GENERATE NEW JOBS
    # ==========================================================

    def _generate_new_jobs(self):

        for job in range(self.num_jobs):

            if self.job_pending[job] == 0:

                if (
                    self.rng.random()
                    <
                    self.job_arrival_probability
                ):

                    self._create_job(job)

    # ==========================================================
    # OBSERVATION
    # ==========================================================

    def _get_observation(self):

        observation = []

        # ------------------------------------------------------
        # TRUCK FEATURES
        # ------------------------------------------------------

        for truck in range(self.num_trucks):

            observation.append(
                float(
                    self.truck_available[truck]
                )
            )

            observation.append(
                float(
                    self.truck_location[truck]
                )
            )

            observation.append(
                float(
                    self.truck_capacity[truck]
                )
            )

        # ------------------------------------------------------
        # JOB FEATURES
        # ------------------------------------------------------

        for job in range(self.num_jobs):

            observation.append(
                float(
                    self.job_pending[job]
                )
            )

            observation.append(
                float(
                    self.job_location[job]
                )
            )

            observation.append(
                float(
                    self.job_load[job]
                )
            )

            remaining_deadline = max(
                0,
                int(
                    self.job_deadline[job]
                )
                -
                self.current_step,
            )

            normalized_deadline = min(
                remaining_deadline / 40.0,
                1.0,
            )

            observation.append(
                float(
                    normalized_deadline
                )
            )

        return np.asarray(
            observation,
            dtype=np.float32,
        )

    # ==========================================================
    # DECODE ACTION
    # ==========================================================

    def decode_action(self, action):

        action = int(action)

        if action == self.wait_action:

            return None, None

        truck = (
            action
            // self.num_jobs
        )

        job = (
            action
            % self.num_jobs
        )

        return truck, job

    # ==========================================================
    # CHECK DISPATCH ACTION
    # ==========================================================

    def is_action_valid(
        self,
        truck,
        job,
    ):

        if not (
            0 <= truck < self.num_trucks
            and
            0 <= job < self.num_jobs
        ):

            return False

        # Truck availability
        if self.truck_available[truck] == 0:

            return False

        # Job availability
        if self.job_pending[job] == 0:

            return False

        # Capacity constraint
        if (
            self.truck_capacity[truck]
            <
            self.job_load[job]
        ):

            return False

        # Deadline constraint
        if (
            self.current_step
            >
            self.job_deadline[job]
        ):

            return False

        return True

    # ==========================================================
    # CHECK WHETHER ANY DISPATCH IS POSSIBLE
    # ==========================================================

    def has_valid_dispatch_action(self):

        for truck in range(
            self.num_trucks
        ):

            for job in range(
                self.num_jobs
            ):

                if self.is_action_valid(
                    truck,
                    job,
                ):

                    return True

        return False

    # ==========================================================
    # ACTION MASK
    # ==========================================================

    def action_masks(self):

        mask = np.zeros(
            self.num_actions,
            dtype=np.int8,
        )

        # ------------------------------------------------------
        # DISPATCH ACTIONS
        # ------------------------------------------------------

        for truck in range(
            self.num_trucks
        ):

            for job in range(
                self.num_jobs
            ):

                action = (
                    truck
                    *
                    self.num_jobs
                    +
                    job
                )

                if self.is_action_valid(
                    truck,
                    job,
                ):

                    mask[action] = 1

        # ------------------------------------------------------
        # WAIT ACTION
        # ------------------------------------------------------

        # WAIT is valid ONLY when no dispatch
        # action is currently feasible.

        if not np.any(
            mask[:self.num_dispatch_actions]
        ):

            mask[self.wait_action] = 1

        return mask

    # ==========================================================
    # VALID ACTIONS
    # ==========================================================

    def valid_actions(self):

        mask = self.action_masks()

        return np.flatnonzero(mask)

    # ==========================================================
    # NUMBER OF VALID ACTIONS
    # ==========================================================

    def num_valid_actions(self):

        return int(
            np.sum(
                self.action_masks()
            )
        )

    # ==========================================================
    # TRAVEL TIME
    # ==========================================================

    def _calculate_travel_time(
        self,
        truck,
        job,
    ):

        distance = abs(
            self.truck_location[truck]
            -
            self.job_location[job]
        )

        travel_time = (
            1
            +
            int(
                distance * 20
            )
        )

        return travel_time

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, action):

        action = int(action)

        # ------------------------------------------------------
        # ACTION RANGE CHECK
        # ------------------------------------------------------

        if not self.action_space.contains(
            action
        ):

            raise ValueError(
                f"Invalid action {action}. "
                f"Valid range: "
                f"0 to "
                f"{self.num_actions - 1}"
            )

        # ------------------------------------------------------
        # CURRENT ACTION MASK
        # ------------------------------------------------------

        current_mask = self.action_masks()

        # ------------------------------------------------------
        # INVALID ACTION
        # ------------------------------------------------------

        if current_mask[action] == 0:

            reward = -5.0

            self.current_step += 1

            self._update_trucks()

            self._generate_new_jobs()

            terminated = False

            truncated = (
                self.current_step
                >= self.max_steps
            )

            next_mask = self.action_masks()

            info = {
                "valid_action": False,
                "action_mask": next_mask,
                "num_valid_actions":
                    int(
                        np.sum(next_mask)
                    ),
                "completed_jobs":
                    self.completed_jobs,
                "invalid_action_penalty":
                    5.0,
            }

            return (
                self._get_observation(),
                reward,
                terminated,
                truncated,
                info,
            )

        # ======================================================
        # WAIT ACTION
        # ======================================================

        if action == self.wait_action:

            # No dispatch is possible.
            # We advance simulation time.

            self.current_step += 1

            self._update_trucks()

            self._generate_new_jobs()

            # Small cost for waiting.
            reward = -0.1

            terminated = False

            truncated = (
                self.current_step
                >= self.max_steps
            )

            next_mask = self.action_masks()

            info = {
                "valid_action": True,
                "wait_action": True,
                "truck": None,
                "job": None,
                "travel_time": 0,
                "waiting_time": 0,
                "reward": reward,
                "completed_jobs":
                    self.completed_jobs,
                "action_mask": next_mask,
                "num_valid_actions":
                    int(
                        np.sum(next_mask)
                    ),
            }

            return (
                self._get_observation(),
                reward,
                terminated,
                truncated,
                info,
            )

        # ======================================================
        # DISPATCH ACTION
        # ======================================================

        truck, job = self.decode_action(
            action
        )

        # ------------------------------------------------------
        # TRAVEL TIME
        # ------------------------------------------------------

        travel_time = (
            self._calculate_travel_time(
                truck,
                job,
            )
        )

        # ------------------------------------------------------
        # WAITING TIME
        # ------------------------------------------------------

        waiting_time = (
            self.current_step
            -
            self.job_arrival_time[job]
        )

        # ------------------------------------------------------
        # UPDATE TRUCK
        # ------------------------------------------------------

        self.truck_available[truck] = 0

        self.truck_busy_until[truck] = (
            self.current_step
            +
            travel_time
        )

        self.truck_location[truck] = (
            self.job_location[job]
        )

        # ------------------------------------------------------
        # COMPLETE JOB
        # ------------------------------------------------------

        self.job_pending[job] = 0

        self.completed_jobs += 1

        # ------------------------------------------------------
        # COSTS
        # ------------------------------------------------------

        self.total_waiting_time += (
            waiting_time
        )

        self.total_travel_cost += (
            travel_time
        )

        # ------------------------------------------------------
        # REWARD
        # ------------------------------------------------------

        reward = (
            10.0
            -
            0.5 * waiting_time
            -
            0.2 * travel_time
        )

        # ------------------------------------------------------
        # ADVANCE SIMULATION
        # ------------------------------------------------------

        self.current_step += 1

        self._update_trucks()

        self._generate_new_jobs()

        # ------------------------------------------------------
        # TERMINATION
        # ------------------------------------------------------

        terminated = False

        truncated = (
            self.current_step
            >= self.max_steps
        )

        # ------------------------------------------------------
        # NEXT MASK
        # ------------------------------------------------------

        next_mask = self.action_masks()

        # ------------------------------------------------------
        # INFO
        # ------------------------------------------------------

        info = {

            "valid_action": True,

            "wait_action": False,

            "truck": truck,

            "job": job,

            "travel_time": travel_time,

            "waiting_time": waiting_time,

            "reward": reward,

            "completed_jobs":
                self.completed_jobs,

            "action_mask": next_mask,

            "num_valid_actions":
                int(
                    np.sum(next_mask)
                ),
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    # ==========================================================
    # RENDER
    # ==========================================================

    def render(self):

        print()
        print("=" * 60)

        print(
            f"Simulation step: "
            f"{self.current_step}"
        )

        print()

        print(
            "Truck availability:",
            self.truck_available,
        )

        print(
            "Truck locations:",
            np.round(
                self.truck_location,
                2,
            ),
        )

        print(
            "Truck capacity:",
            np.round(
                self.truck_capacity,
                2,
            ),
        )

        print()

        print(
            "Pending jobs:",
            self.job_pending,
        )

        print(
            "Job locations:",
            np.round(
                self.job_location,
                2,
            ),
        )

        print(
            "Job loads:",
            np.round(
                self.job_load,
                2,
            ),
        )

        print(
            "Job deadlines:",
            self.job_deadline,
        )

        print()

        mask = self.action_masks()

        print(
            "Action mask:",
            mask,
        )

        print(
            "Valid actions:",
            np.flatnonzero(mask),
        )

        print(
            "Number of valid actions:",
            int(np.sum(mask)),
        )

        if mask[self.wait_action] == 1:

            print(
                "WAIT action is currently VALID."
            )

        print(
            "Completed jobs:",
            self.completed_jobs,
        )

        print("=" * 60)

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(self):
        pass


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    env = TruckDispatchEnv(
        num_trucks=4,
        num_jobs=6,
        max_steps=30,
        job_arrival_probability=0.30,
        seed=42,
    )

    observation, info = env.reset(
        seed=42
    )

    print()
    print("=" * 60)
    print(
        "DYNAMIC TRUCK DISPATCHING ENVIRONMENT"
    )
    print("=" * 60)

    print()

    print(
        "Observation shape:",
        observation.shape,
    )

    print(
        "Global action space:",
        env.action_space,
    )

    print(
        "Total possible actions:",
        env.num_actions,
    )

    print(
        "Truck-job actions:",
        env.num_dispatch_actions,
    )

    print(
        "WAIT action index:",
        env.wait_action,
    )

    print()

    print(
        "Initial action mask:",
        info["action_mask"],
    )

    print(
        "Initial valid actions:",
        info["num_valid_actions"],
    )

    env.render()

    # ==========================================================
    # TEST EPISODE
    # ==========================================================

    print()
    print("=" * 60)
    print("RUNNING TEST EPISODE")
    print("=" * 60)

    valid_action_counts = []

    wait_count = 0

    for step in range(30):

        mask = env.action_masks()

        valid_actions = np.flatnonzero(
            mask
        )

        num_valid = len(
            valid_actions
        )

        valid_action_counts.append(
            num_valid
        )

        print()
        print(
            f"Step {env.current_step}"
        )

        print(
            "Valid actions:",
            valid_actions,
        )

        print(
            "Number of valid actions:",
            num_valid,
        )

        # ------------------------------------------------------
        # SELECT VALID ACTION
        # ------------------------------------------------------

        action = int(
            env.rng.choice(
                valid_actions
            )
        )

        if action == env.wait_action:

            print(
                "Selected action: WAIT"
            )

            wait_count += 1

        else:

            truck, job = (
                env.decode_action(
                    action
                )
            )

            print(
                f"Selected action: "
                f"{action}"
            )

            print(
                f"Truck {truck} -> Job {job}"
            )

        # ------------------------------------------------------
        # STEP
        # ------------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"Reward: {reward:.2f}"
        )

        print(
            "Completed jobs:",
            env.completed_jobs,
        )

        print(
            "Next valid actions:",
            info[
                "num_valid_actions"
            ],
        )

        if terminated or truncated:

            print(
                "\nEpisode ended."
            )

            break

    # ==========================================================
    # VARIABILITY CHECK
    # ==========================================================

    print()
    print("=" * 60)
    print("ACTION-SPACE VARIABILITY CHECK")
    print("=" * 60)

    print(
        "Valid-action counts:",
        valid_action_counts,
    )

    print(
        "Unique valid-action counts:",
        sorted(
            set(
                valid_action_counts
            )
        ),
    )

    print(
        "WAIT selections:",
        wait_count,
    )

    if (
        len(
            set(
                valid_action_counts
            )
        )
        > 1
    ):

        print(
            "SUCCESS: Dynamic action space "
            "verified."
        )

    else:

        print(
            "WARNING: Action-space variation "
            "was not observed."
        )

    # ==========================================================
    # GYMNASIUM CHECK
    # ==========================================================

    print()
    print("=" * 60)
    print("GYMNASIUM CHECK")
    print("=" * 60)

    print(
        "Observation valid:",
        env.observation_space.contains(
            observation
        ),
    )

    print(
        "Action space:",
        env.action_space,
    )

    env.close()