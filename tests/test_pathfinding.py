"""
Unit tests for pathfinding engine and route caching.
"""

from environments.grid_map import GridMap, TILE_OBSTACLE, TILE_ROAD
from environments.pathfinding import find_path, PathCache


def test_shortest_path():
    grid = GridMap(width=10, height=10)
    # Straight line
    path = find_path(grid, (1, 1), (5, 1))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (1, 1)
    assert path[-1] == (5, 1)


def test_obstacle_avoidance():
    grid = GridMap(width=10, height=10)
    # Put a wall of obstacles across x=3 from y=0 to y=8
    for y in range(0, 9):
        grid.set_tile(3, y, TILE_OBSTACLE)

    # Route from (1, 5) to (5, 5) must detour around the obstacle at y=9
    path = find_path(grid, (1, 5), (5, 5))
    assert path is not None
    # Path should not touch any obstacle tile
    for (x, y) in path:
        assert grid.get_tile(x, y) != TILE_OBSTACLE


def test_path_cache():
    grid = GridMap.create_default_map()
    cache = PathCache(grid)

    p1 = cache.get_path((11, 7), (2, 3))
    p2 = cache.get_path((11, 7), (2, 3))
    assert p1 is not None
    assert p1 == p2
    assert cache.get_distance((11, 7), (2, 3)) == len(p1) - 1


if __name__ == "__main__":
    test_shortest_path()
    test_obstacle_avoidance()
    test_path_cache()
    print("test_pathfinding.py PASSED!")
