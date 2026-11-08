"""
Unit tests for GridMap and serialization.
"""

import os
import tempfile
import numpy as np
from environments.grid_map import (
    GridMap,
    TILE_EMPTY,
    TILE_ROAD,
    TILE_OBSTACLE,
    TILE_LOADING_STATION,
    TILE_DUMPING_STATION,
)


def test_grid_map_basics():
    grid = GridMap(width=10, height=8)
    assert grid.width == 10
    assert grid.height == 8
    assert grid.in_bounds(0, 0)
    assert grid.in_bounds(9, 7)
    assert not grid.in_bounds(10, 8)
    assert not grid.in_bounds(-1, 0)

    # Walkable
    assert grid.is_walkable(2, 3)
    grid.set_tile(2, 3, TILE_OBSTACLE)
    assert not grid.is_walkable(2, 3)


def test_stations_and_serialization():
    grid = GridMap(width=12, height=10)
    grid.add_loading_station(2, 2, "Shovel A", station_id=0)
    grid.add_dumping_station(9, 8, "Crusher A", station_id=0)
    grid.add_truck_spawn(5, 5)

    assert len(grid.loading_stations) == 1
    assert len(grid.dumping_stations) == 1
    assert len(grid.truck_spawns) == 1
    assert grid.get_tile(2, 2) == TILE_LOADING_STATION
    assert grid.get_tile(9, 8) == TILE_DUMPING_STATION

    # Test JSON roundtrip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name

    try:
        grid.save_to_json(path)
        loaded = GridMap.load_from_json(path)
        assert loaded.width == 12
        assert loaded.height == 10
        assert len(loaded.loading_stations) == 1
        assert loaded.loading_stations[0]["name"] == "Shovel A"
        assert loaded.get_tile(2, 2) == TILE_LOADING_STATION
        assert loaded.get_tile(9, 8) == TILE_DUMPING_STATION
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    test_grid_map_basics()
    test_stations_and_serialization()
    print("test_grid_map.py PASSED!")
