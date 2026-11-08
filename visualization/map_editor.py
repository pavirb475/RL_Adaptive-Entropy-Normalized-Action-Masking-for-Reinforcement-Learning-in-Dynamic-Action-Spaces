"""
Interactive Tile-based Mine Map Editor using Pygame.
Allows clicking and dragging to place haul roads, pit walls, shovels, crushers, and truck depots.
Includes JSON saving, loading, and automatic route connectivity validation.
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import pygame
import numpy as np

from environments.grid_map import (
    GridMap,
    TILE_EMPTY,
    TILE_ROAD,
    TILE_OBSTACLE,
    TILE_LOADING_STATION,
    TILE_DUMPING_STATION,
    TILE_PARKING,
    TILE_NAMES,
)
from environments.pathfinding import find_path
from visualization.controls import UIButton


# Tile Brush Colors
PALETTE_COLORS = {
    TILE_EMPTY: (60, 68, 80),             # Mine floor grey
    TILE_ROAD: (100, 110, 125),           # Paved dark asphalt
    TILE_OBSTACLE: (160, 60, 50),         # Brick rock red
    TILE_LOADING_STATION: (46, 204, 113), # Emerald green
    TILE_DUMPING_STATION: (230, 126, 34), # Crusher orange
    TILE_PARKING: (52, 152, 219),         # Depot blue
}


class MapEditor:
    """
    Visual map editor with click-and-drag drawing, tool palettes, and JSON I/O.
    """

    def __init__(self, map_filepath: str = "maps/starter_map.json", width: int = 24, height: int = 16):
        self.map_filepath = map_filepath
        if os.path.exists(map_filepath):
            self.grid_map = GridMap.load_from_json(map_filepath)
        else:
            self.grid_map = GridMap.create_default_map(width=width, height=height)

        self.current_tool = TILE_ROAD
        self.status_message = "Ready. Select a tool and paint on the grid."
        self.validation_result: Optional[str] = None

        # Display constants
        self.window_width = 1240
        self.window_height = 760
        self.tile_size = 38
        self.grid_origin_x = 30
        self.grid_origin_y = 70

        self.is_running = True
        self.launch_sim_requested = False

        self._init_pygame()
        self._init_buttons()

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Open-Pit Mine Map Editor — AEN-PPO")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.Font(None, 24)
        self.font_body = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 15)
        self.font_bold = pygame.font.Font(None, 18)

    def _init_buttons(self):
        self.tool_buttons = []
        sidebar_x = 970
        curr_y = 80

        tools = [
            ("Haul Road", TILE_ROAD, PALETTE_COLORS[TILE_ROAD]),
            ("Pit Wall / Rock", TILE_OBSTACLE, PALETTE_COLORS[TILE_OBSTACLE]),
            ("Empty Ground", TILE_EMPTY, PALETTE_COLORS[TILE_EMPTY]),
            ("Loading Station", TILE_LOADING_STATION, PALETTE_COLORS[TILE_LOADING_STATION]),
            ("Dumping Station", TILE_DUMPING_STATION, PALETTE_COLORS[TILE_DUMPING_STATION]),
            ("Truck Parking", TILE_PARKING, PALETTE_COLORS[TILE_PARKING]),
        ]

        for text, tile_val, col in tools:
            btn = UIButton(
                rect=(sidebar_x, curr_y, 240, 36),
                text=text,
                callback=lambda t=tile_val: self.select_tool(t),
                bg_color=(32, 38, 52),
                hover_color=(45, 54, 75),
                active_color=col,
            )
            if tile_val == self.current_tool:
                btn.is_active = True
            self.tool_buttons.append((tile_val, btn))
            curr_y += 44

        curr_y += 20
        # Action Buttons
        self.action_buttons = [
            UIButton((sidebar_x, curr_y, 115, 36), "Save Map", callback=self.save_map, bg_color=(25, 120, 80)),
            UIButton((sidebar_x + 125, curr_y, 115, 36), "Load Map", callback=self.load_map, bg_color=(40, 80, 140)),
            UIButton((sidebar_x, curr_y + 44, 115, 36), "Validate Routes", callback=self.validate_routes, bg_color=(120, 90, 30)),
            UIButton((sidebar_x + 125, curr_y + 44, 115, 36), "Reset Default", callback=self.reset_default, bg_color=(100, 40, 40)),
            UIButton((sidebar_x, curr_y + 96, 240, 44), "LAUNCH SIMULATOR", callback=self.launch_sim, bg_color=(0, 150, 100), font_size=16),
        ]

    def select_tool(self, tile_val: int):
        self.current_tool = tile_val
        for t_val, btn in self.tool_buttons:
            btn.is_active = (t_val == tile_val)
        self.status_message = f"Selected tool: {TILE_NAMES.get(tile_val, 'Unknown')}"

    def save_map(self):
        try:
            self.grid_map.save_to_json(self.map_filepath)
            self.status_message = f"Successfully saved to {self.map_filepath}!"
        except Exception as e:
            self.status_message = f"Error saving map: {e}"

    def load_map(self):
        try:
            if os.path.exists(self.map_filepath):
                self.grid_map = GridMap.load_from_json(self.map_filepath)
                self.status_message = f"Loaded {self.map_filepath} successfully."
            else:
                self.status_message = f"File {self.map_filepath} not found."
        except Exception as e:
            self.status_message = f"Error loading map: {e}"

    def reset_default(self):
        self.grid_map = GridMap.create_default_map(width=self.grid_map.width, height=self.grid_map.height)
        self.status_message = "Restored standard open-pit mine haul road layout."

    def validate_routes(self):
        """Validates that all loading shovels can reach all dumping crushers."""
        if not self.grid_map.loading_stations or not self.grid_map.dumping_stations:
            self.status_message = "Map must have at least 1 loading and 1 dumping station!"
            self.validation_result = "FAIL: Missing stations"
            return

        all_ok = True
        for l in self.grid_map.loading_stations:
            for d in self.grid_map.dumping_stations:
                path = find_path(self.grid_map, tuple(l["pos"]), tuple(d["pos"]))
                if path is None:
                    all_ok = False
                    break

        if all_ok:
            self.validation_result = "PASS: All stations connected!"
            self.status_message = "Route validation SUCCESS: All shovels connect to crushers!"
        else:
            self.validation_result = "FAIL: Broken route detected!"
            self.status_message = "Route validation FAILED: Some stations cannot reach each other."

    def launch_sim(self):
        self.save_map()
        self.launch_sim_requested = True
        self.is_running = False

    def handle_mouse_paint(self):
        mouse_buttons = pygame.mouse.get_pressed()
        if not mouse_buttons[0] and not mouse_buttons[2]:
            return

        mx, my = pygame.mouse.get_pos()
        gx = (mx - self.grid_origin_x) // self.tile_size
        gy = (my - self.grid_origin_y) // self.tile_size

        if not self.grid_map.in_bounds(gx, gy):
            return

        # Right click erases to empty ground
        if mouse_buttons[2]:
            self.grid_map.set_tile(gx, gy, TILE_EMPTY)
            return

        # Left click applies selected tool
        if self.current_tool == TILE_LOADING_STATION:
            self.grid_map.add_loading_station(gx, gy)
        elif self.current_tool == TILE_DUMPING_STATION:
            self.grid_map.add_dumping_station(gx, gy)
        elif self.current_tool == TILE_PARKING:
            self.grid_map.add_truck_spawn(gx, gy)
        else:
            self.grid_map.set_tile(gx, gy, self.current_tool)

    def run(self):
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.is_running = False

                for _, btn in self.tool_buttons:
                    btn.handle_event(event)
                for btn in self.action_buttons:
                    btn.handle_event(event)

            self.handle_mouse_paint()
            self.render()
            self.clock.tick(60)

        pygame.quit()
        return self.launch_sim_requested

    def render(self):
        self.screen.fill((16, 20, 30))

        # 1. Top Header
        title = self.font_title.render("OPEN-PIT MINE GRID MAP EDITOR", True, (240, 245, 255))
        self.screen.blit(title, (30, 20))
        sub = self.font_body.render("Left-click / drag to paint tiles. Right-click to erase. Save map to JSON.", True, (140, 155, 180))
        self.screen.blit(sub, (30, 44))

        # 2. Draw Grid Tiles
        for y in range(self.grid_map.height):
            for x in range(self.grid_map.width):
                px = self.grid_origin_x + x * self.tile_size
                py = self.grid_origin_y + y * self.tile_size
                t_val = self.grid_map.get_tile(x, y)

                col = PALETTE_COLORS.get(t_val, (60, 68, 80))
                rect = pygame.Rect(px, py, self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, col, rect)
                pygame.draw.rect(self.screen, (30, 35, 48), rect, width=1)

                # Special icons/markers
                if t_val == TILE_LOADING_STATION:
                    lbl = self.font_small.render("L", True, (255, 255, 255))
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))
                elif t_val == TILE_DUMPING_STATION:
                    lbl = self.font_small.render("D", True, (255, 255, 255))
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))
                elif t_val == TILE_PARKING:
                    lbl = self.font_small.render("P", True, (255, 255, 255))
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))

        # 3. Draw Sidebar & Buttons
        sidebar_rect = pygame.Rect(950, 60, 270, 660)
        pygame.draw.rect(self.screen, (22, 26, 38), sidebar_rect, border_radius=8)
        pygame.draw.rect(self.screen, (40, 48, 68), sidebar_rect, width=1, border_radius=8)

        self.screen.blit(self.font_bold.render("TOOL PALETTE", True, (220, 230, 245)), (970, 75))

        for _, btn in self.tool_buttons:
            btn.draw(self.screen, self.font_bold)
        for btn in self.action_buttons:
            btn.draw(self.screen, self.font_bold)

        # 4. Status Bar
        status_bar = pygame.Rect(30, 700, 900, 36)
        pygame.draw.rect(self.screen, (22, 26, 38), status_bar, border_radius=6)
        pygame.draw.rect(self.screen, (40, 48, 68), status_bar, width=1, border_radius=6)
        stat_surf = self.font_body.render(self.status_message, True, (0, 200, 140))
        self.screen.blit(stat_surf, (45, 708))

        pygame.display.flip()


if __name__ == "__main__":
    editor = MapEditor()
    editor.run()
