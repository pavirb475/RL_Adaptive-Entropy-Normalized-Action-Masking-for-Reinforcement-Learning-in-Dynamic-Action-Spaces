"""
Grid Map representation for the Open-Pit Mine Truck Dispatch Simulator.
Defines tile types, coordinate management, station locations, and JSON serialization.
"""

from typing import List, Tuple, Dict, Any, Optional
import json
import os
import numpy as np


# Tile Type Constants
TILE_EMPTY = 0             # Open ground / mine bench
TILE_ROAD = 1              # Paved / graded haul road (preferred traversal)
TILE_OBSTACLE = 2          # Boulders, highwalls, pit edges (impassable)
TILE_LOADING_STATION = 3   # Shovels / excavators loading ore
TILE_DUMPING_STATION = 4   # Primary crushers / waste dumps
TILE_PARKING = 5           # Truck depot / dispatch parking bay

TILE_NAMES = {
    TILE_EMPTY: "Empty Ground",
    TILE_ROAD: "Haul Road",
    TILE_OBSTACLE: "Pit Wall / Rock",
    TILE_LOADING_STATION: "Loading Station (Shovel)",
    TILE_DUMPING_STATION: "Dumping Station (Crusher)",
    TILE_PARKING: "Truck Depot / Parking",
}


class GridMap:
    """
    2D Grid representation of an open-pit mine site.
    """

    def __init__(self, width: int = 24, height: int = 16):
        self.width = width
        self.height = height
        self.tiles = np.zeros((height, width), dtype=np.int32)
        
        # Station locations
        self.loading_stations: List[Dict[str, Any]] = []
        self.dumping_stations: List[Dict[str, Any]] = []
        self.truck_spawns: List[Tuple[int, int]] = []

    def in_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates lie within the grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        """
        Check if a tile is passable by trucks.
        Obstacles cannot be traversed.
        """
        if not self.in_bounds(x, y):
            return False
        return self.tiles[y, x] != TILE_OBSTACLE

    def get_tile(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            return TILE_OBSTACLE
        return int(self.tiles[y, x])

    def set_tile(self, x: int, y: int, tile_type: int):
        if self.in_bounds(x, y):
            self.tiles[y, x] = int(tile_type)

    def add_loading_station(self, x: int, y: int, name: str = "", station_id: Optional[int] = None):
        if not self.in_bounds(x, y):
            return
        self.set_tile(x, y, TILE_LOADING_STATION)
        sid = station_id if station_id is not None else len(self.loading_stations)
        sname = name or f"Shovel {sid + 1}"
        # Check if already registered
        for s in self.loading_stations:
            if s["pos"] == [x, y]:
                s["name"] = sname
                return
        self.loading_stations.append({"id": sid, "name": sname, "pos": [x, y]})

    def add_dumping_station(self, x: int, y: int, name: str = "", station_id: Optional[int] = None):
        if not self.in_bounds(x, y):
            return
        self.set_tile(x, y, TILE_DUMPING_STATION)
        sid = station_id if station_id is not None else len(self.dumping_stations)
        sname = name or f"Crusher {sid + 1}"
        for s in self.dumping_stations:
            if s["pos"] == [x, y]:
                s["name"] = sname
                return
        self.dumping_stations.append({"id": sid, "name": sname, "pos": [x, y]})

    def add_truck_spawn(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return
        self.set_tile(x, y, TILE_PARKING)
        if [x, y] not in self.truck_spawns:
            self.truck_spawns.append([x, y])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "tiles": self.tiles.tolist(),
            "loading_stations": self.loading_stations,
            "dumping_stations": self.dumping_stations,
            "truck_spawns": self.truck_spawns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridMap":
        grid = cls(width=data["width"], height=data["height"])
        grid.tiles = np.array(data["tiles"], dtype=np.int32)
        grid.loading_stations = data.get("loading_stations", [])
        grid.dumping_stations = data.get("dumping_stations", [])
        grid.truck_spawns = data.get("truck_spawns", [])
        return grid

    def save_to_json(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "GridMap":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_default_map(cls, width: int = 24, height: int = 16) -> "GridMap":
        """
        Creates a realistic open-pit mine haul road network.
        Includes 2 Loading Stations (Shovels), 2 Dumping Stations (Crushers),
        and 4 truck depot bays, connected by haul roads and surrounded by pit walls.
        """
        grid = cls(width=width, height=height)

        # 1. Fill borders with rocks / pit walls
        for x in range(width):
            grid.set_tile(x, 0, TILE_OBSTACLE)
            grid.set_tile(x, height - 1, TILE_OBSTACLE)
        for y in range(height):
            grid.set_tile(0, y, TILE_OBSTACLE)
            grid.set_tile(width - 1, y, TILE_OBSTACLE)

        # 2. Add interior rock massifs / unmined benches
        for y in range(2, 6):
            for x in range(7, 10):
                grid.set_tile(x, y, TILE_OBSTACLE)
        for y in range(10, 14):
            for x in range(7, 10):
                grid.set_tile(x, y, TILE_OBSTACLE)
        for y in range(4, 12):
            for x in range(14, 17):
                grid.set_tile(x, y, TILE_OBSTACLE)

        # 3. Main arterial haul roads
        # East-West corridors
        for x in range(1, width - 1):
            grid.set_tile(x, 7, TILE_ROAD)
            grid.set_tile(x, 8, TILE_ROAD)
            grid.set_tile(x, 1, TILE_ROAD)
            grid.set_tile(x, height - 2, TILE_ROAD)

        # North-South connecting ramps
        for y in range(1, height - 1):
            grid.set_tile(3, y, TILE_ROAD)
            grid.set_tile(11, y, TILE_ROAD)
            grid.set_tile(12, y, TILE_ROAD)
            grid.set_tile(20, y, TILE_ROAD)

        # 4. Loading Stations (West benches)
        # Loading Station 1 (North Shovel) at (2, 3)
        grid.add_loading_station(2, 3, name="North Pit Shovel #1", station_id=0)
        grid.set_tile(1, 3, TILE_ROAD)

        # Loading Station 2 (South Shovel) at (2, 12)
        grid.add_loading_station(2, 12, name="South Pit Shovel #2", station_id=1)
        grid.set_tile(1, 12, TILE_ROAD)

        # 5. Dumping Stations (East dump & crusher)
        # Dumping Station 1 (Primary Crusher) at (21, 3)
        grid.add_dumping_station(21, 3, name="Primary Gyratory Crusher", station_id=0)
        grid.set_tile(22, 3, TILE_ROAD)

        # Dumping Station 2 (Waste Rock Dump) at (21, 12)
        grid.add_dumping_station(21, 12, name="East Waste Rock Dump", station_id=1)
        grid.set_tile(22, 12, TILE_ROAD)

        # 6. Truck Depot / Parking bays (Central dispatch area)
        depot_spots = [(11, 7), (12, 7), (11, 8), (12, 8)]
        for x, y in depot_spots:
            grid.add_truck_spawn(x, y)

        return grid

    @classmethod
    def create_scalable_map(
        cls,
        num_pits: int = 2,
        num_dumping: int = 2,
        num_truck_spawns: int = 8,
        width: int = 24,
        height: int = 16,
    ) -> "GridMap":
        """
        Creates a configurable, realistic open-pit mine haul road network.
        Supports 1 to 4 distinct loading pits/shovels, 1 to 3 crushers/dumps,
        and up to 8 dedicated truck depot bays, ensuring full road connectivity.
        """
        grid = cls(width=width, height=height)

        # 1. Fill borders with rocks / pit walls
        for x in range(width):
            grid.set_tile(x, 0, TILE_OBSTACLE)
            grid.set_tile(x, height - 1, TILE_OBSTACLE)
        for y in range(height):
            grid.set_tile(0, y, TILE_OBSTACLE)
            grid.set_tile(width - 1, y, TILE_OBSTACLE)

        # 2. Add interior rock massifs / unmined benches
        for y in range(2, 6):
            for x in range(7, 10):
                grid.set_tile(x, y, TILE_OBSTACLE)
        for y in range(10, 14):
            for x in range(7, 10):
                grid.set_tile(x, y, TILE_OBSTACLE)
        for y in range(4, 12):
            for x in range(14, 17):
                grid.set_tile(x, y, TILE_OBSTACLE)

        # 3. Main arterial haul roads
        # East-West corridors
        for x in range(1, width - 1):
            grid.set_tile(x, 7, TILE_ROAD)
            grid.set_tile(x, 8, TILE_ROAD)
            grid.set_tile(x, 1, TILE_ROAD)
            grid.set_tile(x, height - 2, TILE_ROAD)

        # North-South connecting ramps
        for y in range(1, height - 1):
            grid.set_tile(3, y, TILE_ROAD)
            grid.set_tile(11, y, TILE_ROAD)
            grid.set_tile(12, y, TILE_ROAD)
            grid.set_tile(20, y, TILE_ROAD)

        # 4. Loading Stations (Pits / Shovels: 1 to 4)
        pit_candidates = [
            (2, 3, "Northwest Pit Shovel #1", (1, 3)),
            (2, 12, "Southwest Pit Shovel #2", (1, 12)),
            (10, 2, "Mid-North Pit Shovel #3", (10, 1)),
            (10, 13, "Mid-South Pit Shovel #4", (10, 14)),
        ]

        active_pits = max(1, min(num_pits, len(pit_candidates)))
        for i in range(active_pits):
            px, py, pname, road_conn = pit_candidates[i]
            grid.add_loading_station(px, py, name=pname, station_id=i)
            grid.set_tile(road_conn[0], road_conn[1], TILE_ROAD)

        # 5. Dumping Stations (Crushers / Dumps: 1 to 2)
        dump_candidates = [
            (21, 3, "Primary Gyratory Crusher #1", (22, 3)),
            (21, 12, "East Waste Rock Dump #2", (22, 12)),
            (21, 7, "Secondary Cone Crusher #3", (22, 7)),
        ]

        active_dumps = max(1, min(num_dumping, len(dump_candidates)))
        for i in range(active_dumps):
            dx, dy, dname, road_conn = dump_candidates[i]
            grid.add_dumping_station(dx, dy, name=dname, station_id=i)
            grid.set_tile(road_conn[0], road_conn[1], TILE_ROAD)

        # 6. Truck Depot Bays (Central dispatch parking spots: 2 to 8)
        depot_spots = [
            (11, 7), (12, 7), (11, 8), (12, 8),
            (10, 7), (13, 7), (10, 8), (13, 8),
        ]

        active_spawns = max(2, min(num_truck_spawns, len(depot_spots)))
        for x, y in depot_spots[:active_spawns]:
            grid.add_truck_spawn(x, y)

        return grid

