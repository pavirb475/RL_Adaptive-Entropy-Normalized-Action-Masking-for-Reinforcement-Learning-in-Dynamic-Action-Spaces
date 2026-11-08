"""
Pathfinding engine for the Truck Dispatch Simulator.
Implements Breadth-First Search (BFS) and A* routing on the GridMap with road preference and caching.
"""

from typing import List, Tuple, Optional, Dict
from collections import deque
import heapq
import numpy as np
from environments.grid_map import GridMap, TILE_OBSTACLE, TILE_ROAD


def get_tile_cost(grid_map: GridMap, x: int, y: int) -> float:
    """
    Returns traversal cost of a tile.
    Haul roads have lowest cost (1.0).
    Open ground is passable but slightly slower (1.8).
    Obstacles are impassable (infinity).
    """
    if not grid_map.in_bounds(x, y):
        return float("inf")
    tile = grid_map.get_tile(x, y)
    if tile == TILE_OBSTACLE:
        return float("inf")
    elif tile == TILE_ROAD:
        return 1.0
    else:
        return 1.8


def find_path(
    grid_map: GridMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Find shortest passable route from start to goal using Dijkstra/A*.
    Prefers haul roads while allowing traversal across open benches if needed.
    Returns list of coordinate tuples [(x_start, y_start), ..., (x_goal, y_goal)].
    """
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    if start == goal:
        return [start]

    if not grid_map.is_walkable(start[0], start[1]) or not grid_map.is_walkable(goal[0], goal[1]):
        return None

    # Priority queue: (f_score, cost, (x, y))
    open_set = []
    heapq.heappush(open_set, (0.0, 0.0, start))
    
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}

    # 4-directional movement (orthogonal)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            curr = goal
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            path.reverse()
            return path

        if current_cost > cost_so_far.get(current, float("inf")):
            continue

        cx, cy = current
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if not grid_map.in_bounds(nx, ny):
                continue
            step_cost = get_tile_cost(grid_map, nx, ny)
            if step_cost == float("inf"):
                continue

            new_cost = current_cost + step_cost
            neighbor = (nx, ny)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                # Manhattan heuristic
                h = abs(nx - goal[0]) + abs(ny - goal[1])
                f_score = new_cost + h
                heapq.heappush(open_set, (f_score, new_cost, neighbor))
                came_from[neighbor] = current

    return None


class PathCache:
    """
    Caches calculated paths between frequently accessed coordinates
    (e.g., loading stations, dumping stations, and depot bays).
    """

    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        self._cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Optional[List[Tuple[int, int]]]] = {}

    def get_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        s = (int(start[0]), int(start[1]))
        g = (int(goal[0]), int(goal[1]))
        key = (s, g)
        if key not in self._cache:
            path = find_path(self.grid_map, s, g)
            self._cache[key] = path
        return self._cache[key]

    def get_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        path = self.get_path(start, goal)
        if path is None:
            return 999
        return max(1, len(path) - 1)

    def clear(self):
        self._cache.clear()
