"""
Truck entity model for the Open-Pit Mine Haul Simulator.
Tracks truck kinematics, capacity, loading cycle state, and route execution.
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np


class TruckStatus:
    IDLE = "IDLE"
    TO_LOAD = "TO_LOAD"
    LOADING = "LOADING"
    TO_DUMP = "TO_DUMP"
    UNLOADING = "UNLOADING"


class Truck:
    """
    Simulates a heavy haul truck operating in an open-pit mine.
    """

    def __init__(
        self,
        truck_id: int,
        spawn_pos: Tuple[int, int],
        capacity: float = 1.0,
    ):
        self.truck_id = truck_id
        self.spawn_pos = (int(spawn_pos[0]), int(spawn_pos[1]))
        
        # Grid coordinates
        self.x: int = self.spawn_pos[0]
        self.y: int = self.spawn_pos[1]
        
        # Continuous coordinates for smooth animation rendering
        self.render_x: float = float(self.x)
        self.render_y: float = float(self.y)
        
        # Heading angle in degrees (0 = East, 90 = South, 180 = West, 270 = North)
        self.heading: float = 0.0
        
        # Payload and specs
        self.capacity: float = float(capacity)
        self.current_load: float = 0.0
        
        # State machine
        self.status: str = TruckStatus.IDLE
        self.current_job_id: Optional[int] = None
        self.busy_until_step: int = 0
        
        # Movement and route planning
        self.path: List[Tuple[int, int]] = []
        self.target_station_name: str = ""
        self.path_to_dump: List[Tuple[int, int]] = []
        
        # Cycle timers (in simulation steps)
        self.operation_timer: int = 0
        self.load_duration: int = 2
        self.dump_duration: int = 2
        
        # Statistics
        self.total_jobs_completed: int = 0
        self.total_distance_traveled: int = 0

    @property
    def is_available(self) -> bool:
        """A truck is available for dispatch only when IDLE and empty."""
        return self.status == TruckStatus.IDLE

    def assign_dispatch(
        self,
        job_id: int,
        load_pos: Tuple[int, int],
        dump_pos: Tuple[int, int],
        path_to_load: List[Tuple[int, int]],
        path_to_dump: List[Tuple[int, int]],
        current_step: int,
        total_travel_time: int,
    ):
        """
        Assigns a new dispatch mission to this truck.
        """
        self.current_job_id = job_id
        self.status = TruckStatus.TO_LOAD
        self.path = list(path_to_load)
        self.path_to_dump = list(path_to_dump)
        self.busy_until_step = current_step + total_travel_time
        self.operation_timer = 0

    def fast_forward_step(self):
        """
        Discrete simulation step update (used for fast training and headless mode).
        Advances one tile or one loading/unloading tick.
        """
        if self.status == TruckStatus.IDLE:
            return

        if self.status == TruckStatus.TO_LOAD:
            if len(self.path) > 1:
                self.path.pop(0)
                next_pos = self.path[0]
                self._update_heading(next_pos[0], next_pos[1])
                self.x, self.y = next_pos
                self.render_x, self.render_y = float(self.x), float(self.y)
                self.total_distance_traveled += 1
            else:
                # Arrived at loading shovel
                self.status = TruckStatus.LOADING
                self.operation_timer = self.load_duration

        elif self.status == TruckStatus.LOADING:
            self.operation_timer -= 1
            if self.operation_timer <= 0:
                # Finished loading payload
                self.current_load = self.capacity
                self.status = TruckStatus.TO_DUMP
                self.path = list(self.path_to_dump)
                self.path_to_dump = []

        elif self.status == TruckStatus.TO_DUMP:
            if len(self.path) > 1:
                self.path.pop(0)
                next_pos = self.path[0]
                self._update_heading(next_pos[0], next_pos[1])
                self.x, self.y = next_pos
                self.render_x, self.render_y = float(self.x), float(self.y)
                self.total_distance_traveled += 1
            else:
                # Arrived at crusher / dump
                self.status = TruckStatus.UNLOADING
                self.operation_timer = self.dump_duration

        elif self.status == TruckStatus.UNLOADING:
            self.operation_timer -= 1
            if self.operation_timer <= 0:
                # Finished dumping payload
                self.current_load = 0.0
                self.status = TruckStatus.IDLE
                self.current_job_id = None
                self.total_jobs_completed += 1

    def update_animation(self, speed: float = 0.15):
        """
        Smoothly interpolates render_x and render_y towards (x, y) for 60fps rendering.
        """
        target_x = float(self.x)
        target_y = float(self.y)
        dx = target_x - self.render_x
        dy = target_y - self.render_y
        dist = math.hypot(dx, dy)
        if dist < speed:
            self.render_x = target_x
            self.render_y = target_y
        else:
            self.render_x += (dx / dist) * speed
            self.render_y += (dy / dist) * speed

    def _update_heading(self, next_x: int, next_y: int):
        dx = next_x - self.x
        dy = next_y - self.y
        if dx != 0 or dy != 0:
            angle_rad = math.atan2(dy, dx)
            self.heading = math.degrees(angle_rad)

    def reset(self, spawn_pos: Optional[Tuple[int, int]] = None, capacity: Optional[float] = None):
        if spawn_pos is not None:
            self.spawn_pos = (int(spawn_pos[0]), int(spawn_pos[1]))
        if capacity is not None:
            self.capacity = float(capacity)
        self.x, self.y = self.spawn_pos
        self.render_x, self.render_y = float(self.x), float(self.y)
        self.heading = 0.0
        self.current_load = 0.0
        self.status = TruckStatus.IDLE
        self.current_job_id = None
        self.busy_until_step = 0
        self.path = []
        self.path_to_dump = []
        self.operation_timer = 0
        self.total_jobs_completed = 0
        self.total_distance_traveled = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truck_id": self.truck_id,
            "x": self.x,
            "y": self.y,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "status": self.status,
            "current_job_id": self.current_job_id,
            "busy_until_step": self.busy_until_step,
            "jobs_completed": self.total_jobs_completed,
        }
