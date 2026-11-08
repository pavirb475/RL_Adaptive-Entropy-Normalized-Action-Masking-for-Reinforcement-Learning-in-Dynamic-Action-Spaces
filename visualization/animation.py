"""
Rendering and Animation helpers for Haul Trucks and Routes.
Provides high-fidelity vector rendering of mining trucks, payload bars, and route trajectories.
"""

from typing import List, Tuple, Optional
import math
import pygame
from environments.truck import Truck, TruckStatus


# Color Palette
COLOR_TRUCK_BODY = (245, 185, 20)       # Caterpillar mining yellow
COLOR_TRUCK_CABIN = (30, 36, 48)        # Dark slate cabin
COLOR_TRUCK_TIRES = (20, 22, 26)        # Heavy rubber black
COLOR_ORE_PAYLOAD = (180, 115, 60)      # Copper / iron ore gravel
COLOR_PAYLOAD_EMPTY = (70, 75, 85)
COLOR_TEXT_WHITE = (245, 248, 252)

STATUS_COLORS = {
    TruckStatus.IDLE: (46, 204, 113),       # Emerald Green
    TruckStatus.TO_LOAD: (52, 152, 219),    # Azure Blue / Cyan
    TruckStatus.LOADING: (241, 196, 15),    # Golden Amber
    TruckStatus.TO_DUMP: (230, 126, 34),    # Bright Orange
    TruckStatus.UNLOADING: (155, 89, 182),  # Purple
}


def draw_route_trajectory(
    surface: pygame.Surface,
    path: List[Tuple[int, int]],
    origin_x: int,
    origin_y: int,
    tile_size: int,
    color: Tuple[int, int, int] = (52, 152, 219),
):
    """
    Renders the truck's scheduled waypoints as a glowing path.
    """
    if not path or len(path) < 2:
        return

    points = []
    for x, y in path:
        px = origin_x + int(x * tile_size + tile_size // 2)
        py = origin_y + int(y * tile_size + tile_size // 2)
        points.append((px, py))

    # Draw semi-transparent glow trail
    if len(points) >= 2:
        glow_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(glow_surf, (*color, 80), False, points, width=5)
        pygame.draw.lines(glow_surf, (*color, 200), False, points, width=2)
        surface.blit(glow_surf, (0, 0))

        # Destination target circle
        dest_x, dest_y = points[-1]
        pygame.draw.circle(surface, color, (dest_x, dest_y), 5)
        pygame.draw.circle(surface, (255, 255, 255), (dest_x, dest_y), 2)


def render_truck_sprite(
    surface: pygame.Surface,
    truck: Truck,
    origin_x: int,
    origin_y: int,
    tile_size: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
):
    """
    Renders an open-pit mining haul truck with heading orientation,
    payload bar, and status indicator badge.
    """
    # Calculate exact screen pixel center
    center_x = origin_x + int(truck.render_x * tile_size + tile_size / 2)
    center_y = origin_y + int(truck.render_y * tile_size + tile_size / 2)

    truck_w = int(tile_size * 0.85)
    truck_h = int(tile_size * 0.65)

    # Base truck surface for rotation
    truck_surf = pygame.Surface((truck_w, truck_h), pygame.SRCALPHA)

    # 1. Tires (6-wheel heavy haul truck layout)
    tire_w = int(truck_w * 0.22)
    tire_h = int(truck_h * 0.24)
    # Front tires
    pygame.draw.rect(truck_surf, COLOR_TRUCK_TIRES, (truck_w - tire_w - 2, 0, tire_w, tire_h), border_radius=2)
    pygame.draw.rect(truck_surf, COLOR_TRUCK_TIRES, (truck_w - tire_w - 2, truck_h - tire_h, tire_w, tire_h), border_radius=2)
    # Rear tandem dual tires
    pygame.draw.rect(truck_surf, COLOR_TRUCK_TIRES, (2, 0, tire_w, tire_h), border_radius=2)
    pygame.draw.rect(truck_surf, COLOR_TRUCK_TIRES, (2, truck_h - tire_h, tire_w, tire_h), border_radius=2)
    pygame.draw.rect(truck_surf, COLOR_TRUCK_TIRES, (tire_w + 4, 0, tire_w, tire_h), border_radius=2)
    pygame.draw.rect(truck_surf, COLOR_TRUCK_TIRES, (tire_w + 4, truck_h - tire_h, tire_w, tire_h), border_radius=2)

    # 2. Main Dump Tray / Chassis
    tray_rect = pygame.Rect(tire_w, 3, truck_w - tire_w - 4, truck_h - 6)
    pygame.draw.rect(truck_surf, COLOR_TRUCK_BODY, tray_rect, border_radius=4)
    pygame.draw.rect(truck_surf, (180, 130, 10), tray_rect, width=1, border_radius=4)

    # 3. Ore Payload in the dump tray
    if truck.current_load > 0.05:
        fill_ratio = min(1.0, truck.current_load / max(0.1, truck.capacity))
        ore_w = int((tray_rect.width - 6) * fill_ratio)
        ore_rect = pygame.Rect(tray_rect.x + 3, tray_rect.y + 3, ore_w, tray_rect.height - 6)
        pygame.draw.rect(truck_surf, COLOR_ORE_PAYLOAD, ore_rect, border_radius=2)
        # Texture lines
        for offset in range(4, ore_w, 6):
            pygame.draw.line(
                truck_surf,
                (140, 85, 40),
                (ore_rect.x + offset, ore_rect.y + 2),
                (ore_rect.x + offset, ore_rect.y + ore_rect.height - 2),
            )

    # 4. Front Operator Cab (positioned toward front of vehicle)
    cab_w = int(truck_w * 0.26)
    cab_h = int(truck_h * 0.55)
    cab_rect = pygame.Rect(truck_w - cab_w - 4, (truck_h - cab_h) // 2, cab_w, cab_h)
    pygame.draw.rect(truck_surf, COLOR_TRUCK_CABIN, cab_rect, border_radius=3)
    # Windshield / glass
    windshield = pygame.Rect(cab_rect.right - 5, cab_rect.y + 2, 3, cab_h - 4)
    pygame.draw.rect(truck_surf, (100, 200, 255), windshield, border_radius=1)

    # Rotate truck surface according to heading
    rotated_surf = pygame.transform.rotate(truck_surf, -truck.heading)
    rot_rect = rotated_surf.get_rect(center=(center_x, center_y))
    surface.blit(rotated_surf, rot_rect)

    # 5. Overhead Truck Badge: "T{id+1}"
    badge_color = STATUS_COLORS.get(truck.status, (150, 150, 150))
    id_text = f"T{truck.truck_id + 1}"
    badge_surf = font.render(id_text, True, COLOR_TEXT_WHITE)
    badge_bg = pygame.Rect(center_x - 12, center_y - tile_size // 2 - 8, 24, 16)
    pygame.draw.rect(surface, (20, 25, 35), badge_bg, border_radius=4)
    pygame.draw.rect(surface, badge_color, badge_bg, width=2, border_radius=4)
    surface.blit(badge_surf, badge_surf.get_rect(center=badge_bg.center))

    # 6. Overhead Load Capacity Bar
    bar_w = tile_size - 4
    bar_h = 4
    bar_x = center_x - bar_w // 2
    bar_y = center_y + tile_size // 2 - 2
    load_ratio = min(1.0, truck.current_load / max(0.1, truck.capacity))

    pygame.draw.rect(surface, (40, 45, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=2)
    if load_ratio > 0:
        fill_col = (46, 204, 113) if load_ratio >= 0.9 else (241, 196, 15)
        pygame.draw.rect(surface, fill_col, (bar_x, bar_y, int(bar_w * load_ratio), bar_h), border_radius=2)
