"""
UI Controls and Interaction Handler for the Truck Dispatch Simulator.
Provides clickable buttons, keyboard hotkeys, playback states, and agent selectors.
"""

from typing import List, Tuple, Callable, Optional, Dict, Any
import pygame


class UIButton:
    """
    Modern styled clickable button for Pygame interfaces.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        text: str,
        callback: Optional[Callable[[], None]] = None,
        bg_color: Tuple[int, int, int] = (38, 44, 58),
        hover_color: Tuple[int, int, int] = (58, 68, 92),
        active_color: Tuple[int, int, int] = (0, 168, 120),
        text_color: Tuple[int, int, int] = (240, 243, 248),
        font_size: int = 15,
        is_toggle: bool = False,
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.active_color = active_color
        self.text_color = text_color
        self.font_size = font_size
        self.is_toggle = is_toggle
        self.is_active = False
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.is_toggle:
                    self.is_active = not self.is_active
                if self.callback:
                    self.callback()
                return True
        return False

    def draw(self, surface: pygame.Surface, font: Optional[pygame.font.Font] = None):
        if font is None:
            font = pygame.font.Font(None, self.font_size + 4)

        if self.is_active:
            color = self.active_color
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.bg_color

        # Rounded rectangle button with subtle border
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        border_col = (90, 105, 135) if self.is_hovered else (50, 60, 80)
        pygame.draw.rect(surface, border_col, self.rect, width=1, border_radius=6)

        txt_surf = font.render(self.text, True, self.text_color)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class SimulationControls:
    """
    Manages interactive simulation playback settings, speeds, and agent selection.
    """

    def __init__(self):
        self.is_paused: bool = False
        self.step_requested: bool = False
        self.reset_requested: bool = False
        
        # Playback speed multipliers (ticks per second)
        self.speed_options: List[int] = [1, 2, 5, 10, 30]
        self.speed_idx: int = 1
        
        # Agent selection modes
        self.agent_modes: List[str] = ["aen_ppo", "masked_ppo", "ppo", "random", "manual"]
        self.agent_mode_idx: int = 0
        
        # Show overlays
        self.show_paths: bool = True
        self.show_dashboard: bool = True

    @property
    def current_speed(self) -> int:
        return self.speed_options[self.speed_idx]

    @property
    def selected_agent(self) -> str:
        return self.agent_modes[self.agent_mode_idx]

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def request_single_step(self):
        self.step_requested = True
        self.is_paused = True

    def request_reset(self):
        self.reset_requested = True

    def cycle_speed(self):
        self.speed_idx = (self.speed_idx + 1) % len(self.speed_options)

    def set_speed(self, speed_val: int):
        if speed_val in self.speed_options:
            self.speed_idx = self.speed_options.index(speed_val)

    def set_agent(self, agent_name: str):
        if agent_name in self.agent_modes:
            self.agent_mode_idx = self.agent_modes.index(agent_name)

    def cycle_agent(self):
        self.agent_mode_idx = (self.agent_mode_idx + 1) % len(self.agent_modes)

    def handle_key_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.toggle_pause()
            elif event.key == pygame.K_RIGHT:
                self.request_single_step()
            elif event.key == pygame.K_r:
                self.request_reset()
            elif event.key == pygame.K_UP:
                if self.speed_idx < len(self.speed_options) - 1:
                    self.speed_idx += 1
            elif event.key == pygame.K_DOWN:
                if self.speed_idx > 0:
                    self.speed_idx -= 1
            elif event.key == pygame.K_a:
                self.cycle_agent()
            elif event.key == pygame.K_p:
                self.show_paths = not self.show_paths
            elif event.key == pygame.K_d:
                self.show_dashboard = not self.show_dashboard
