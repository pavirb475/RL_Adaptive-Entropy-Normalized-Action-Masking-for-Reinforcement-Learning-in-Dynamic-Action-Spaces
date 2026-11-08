"""
Main Pygame Simulator Window for AEN-PPO Truck Dispatching.
Combines real-time grid animation, route visualization, interactive controls, and research telemetry.
"""

from typing import Optional, Dict, Any, List
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
)
from visualization.controls import SimulationControls, UIButton
from visualization.animation import render_truck_sprite, draw_route_trajectory, STATUS_COLORS
from visualization.dashboard import Dashboard


class SimulatorWindow:
    """
    High-fidelity Pygame visualizer and interactive simulator.
    """

    def __init__(self, env, window_width: int = 1360, window_height: int = 780):
        self.env = env
        self.window_width = window_width
        self.window_height = window_height

        self.viewport_w = 920
        self.viewport_h = window_height
        self.dashboard_w = window_width - self.viewport_w

        # Tile grid scaling
        self.tile_size = min(
            (self.viewport_w - 40) // self.env.grid_map.width,
            (self.viewport_h - 100) // self.env.grid_map.height,
        )
        self.grid_origin_x = 20 + (self.viewport_w - 40 - self.env.grid_map.width * self.tile_size) // 2
        self.grid_origin_y = 20 + (self.viewport_h - 100 - self.env.grid_map.height * self.tile_size) // 2

        # State tracking
        self.controls = SimulationControls()
        self.dashboard = Dashboard(rect=(self.viewport_w, 0, self.dashboard_w, self.window_height))
        
        self.cumulative_reward: float = 0.0
        self.invalid_action_count: int = 0
        self.last_selected_action: Optional[int] = None

        self._init_pygame()
        self._init_ui_buttons()

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("AEN-PPO: Open-Pit Mine Dynamic Truck Dispatch Simulator")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        # Load fonts
        self.fonts = {
            "title": pygame.font.Font(None, 20),
            "body": pygame.font.Font(None, 16),
            "small": pygame.font.Font(None, 14),
            "bold": pygame.font.Font(None, 17),
            "large": pygame.font.Font(None, 24),
        }

    def _init_ui_buttons(self):
        ctrl_y = self.viewport_h - 60
        btn_h = 36

        self.ui_buttons = [
            UIButton((20, ctrl_y, 90, btn_h), "PLAY / PAUSE", callback=self.controls.toggle_pause, bg_color=(35, 120, 80)),
            UIButton((120, ctrl_y, 75, btn_h), "STEP", callback=self.controls.request_single_step, bg_color=(45, 60, 90)),
            UIButton((205, ctrl_y, 65, btn_h), "RESET", callback=self.reset_sim, bg_color=(120, 45, 45)),
            UIButton((280, ctrl_y, 60, btn_h), "1x", callback=lambda: self.controls.set_speed(1), bg_color=(38, 44, 58)),
            UIButton((345, ctrl_y, 60, btn_h), "2x", callback=lambda: self.controls.set_speed(2), bg_color=(38, 44, 58)),
            UIButton((410, ctrl_y, 60, btn_h), "5x", callback=lambda: self.controls.set_speed(5), bg_color=(38, 44, 58)),
            UIButton((475, ctrl_y, 65, btn_h), "MAX", callback=lambda: self.controls.set_speed(30), bg_color=(38, 44, 58)),
            UIButton((550, ctrl_y, 140, btn_h), "CYCLE POLICY", callback=self.controls.cycle_agent, bg_color=(0, 130, 100)),
        ]

    def reset_sim(self):
        self.cumulative_reward = 0.0
        self.invalid_action_count = 0
        self.last_selected_action = None
        self.env.reset()

    def handle_events(self) -> bool:
        """Process Pygame user inputs. Returns False if user closed the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                self.controls.handle_key_event(event)

            for btn in self.ui_buttons:
                btn.handle_event(event)

        return True

    def update_animation(self):
        """Smoothly interpolates trucks for fluid visual movement."""
        anim_speed = 0.25 if self.controls.current_speed <= 5 else 0.8
        for truck in self.env.trucks:
            truck.update_animation(speed=anim_speed)

    def record_step_result(self, action: int, reward: float, info: Dict[str, Any]):
        self.last_selected_action = action
        self.cumulative_reward += float(reward)
        if not info.get("valid_action", True):
            self.invalid_action_count += 1

        self.dashboard.record_step(
            reward=reward,
            valid_actions_count=info.get("num_valid_actions", self.env.num_valid_actions()),
        )

    def render_frame(self):
        self.screen.fill((14, 17, 24))

        # 1. Draw Mine Grid Map
        self._render_grid_map()

        # 2. Draw Route Trails if enabled
        if self.controls.show_paths:
            for truck in self.env.trucks:
                if truck.path:
                    draw_route_trajectory(
                        surface=self.screen,
                        path=[(truck.x, truck.y)] + list(truck.path),
                        origin_x=self.grid_origin_x,
                        origin_y=self.grid_origin_y,
                        tile_size=self.tile_size,
                        color=STATUS_COLORS.get(truck.status, (52, 152, 219)),
                    )

        # 3. Draw Haul Trucks
        font_b = self.fonts["bold"]
        font_s = self.fonts["small"]
        for truck in self.env.trucks:
            render_truck_sprite(
                surface=self.screen,
                truck=truck,
                origin_x=self.grid_origin_x,
                origin_y=self.grid_origin_y,
                tile_size=self.tile_size,
                font=font_b,
                small_font=font_s,
            )

        # 4. Draw UI Buttons
        for btn in self.ui_buttons:
            # Highlight active speed
            if btn.text in ["1x", "2x", "5x", "MAX"]:
                sp_map = {"1x": 1, "2x": 2, "5x": 5, "MAX": 30}
                btn.is_active = (self.controls.current_speed == sp_map.get(btn.text))
            btn.draw(self.screen, self.fonts["bold"])

        # 5. Draw Right Telemetry Dashboard
        mask = self.env.action_masks()
        self.dashboard.draw(
            surface=self.screen,
            current_step=self.env.current_step,
            max_steps=self.env.max_steps,
            completed_jobs=self.env.completed_jobs,
            cumulative_reward=self.cumulative_reward,
            invalid_count=self.invalid_action_count,
            action_mask=mask,
            selected_action=self.last_selected_action,
            trucks=self.env.trucks,
            job_manager=self.env.job_manager,
            active_agent_name=self.controls.selected_agent,
            fonts=self.fonts,
        )

        pygame.display.flip()

    def _render_grid_map(self):
        grid = self.env.grid_map
        for y in range(grid.height):
            for x in range(grid.width):
                px = self.grid_origin_x + x * self.tile_size
                py = self.grid_origin_y + y * self.tile_size
                t_val = grid.get_tile(x, y)
                rect = pygame.Rect(px, py, self.tile_size, self.tile_size)

                # Tile background
                if t_val == TILE_EMPTY:
                    pygame.draw.rect(self.screen, (34, 38, 48), rect)
                    pygame.draw.rect(self.screen, (28, 32, 40), rect, width=1)
                elif t_val == TILE_ROAD:
                    pygame.draw.rect(self.screen, (55, 62, 75), rect)
                    # Dashed road marking
                    if self.tile_size >= 24:
                        pygame.draw.circle(self.screen, (85, 95, 115), rect.center, 2)
                elif t_val == TILE_OBSTACLE:
                    pygame.draw.rect(self.screen, (75, 42, 38), rect, border_radius=3)
                    pygame.draw.rect(self.screen, (95, 52, 48), rect, width=1, border_radius=3)
                elif t_val == TILE_LOADING_STATION:
                    pygame.draw.rect(self.screen, (28, 120, 75), rect, border_radius=4)
                    pygame.draw.rect(self.screen, (46, 204, 113), rect, width=2, border_radius=4)
                    lbl = self.fonts["bold"].render("L", True, (255, 255, 255))
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))
                elif t_val == TILE_DUMPING_STATION:
                    pygame.draw.rect(self.screen, (150, 70, 20), rect, border_radius=4)
                    pygame.draw.rect(self.screen, (230, 126, 34), rect, width=2, border_radius=4)
                    lbl = self.fonts["bold"].render("D", True, (255, 255, 255))
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))
                elif t_val == TILE_PARKING:
                    pygame.draw.rect(self.screen, (30, 60, 100), rect, border_radius=3)
                    pygame.draw.rect(self.screen, (52, 152, 219), rect, width=1, border_radius=3)
                    lbl = self.fonts["small"].render("P", True, (180, 210, 255))
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def close(self):
        pygame.quit()
