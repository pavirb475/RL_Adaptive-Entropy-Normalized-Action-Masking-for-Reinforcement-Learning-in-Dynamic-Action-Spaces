"""
Research Telemetry and Dashboard Panel for Open-Pit Mine RL Dispatch.
Renders real-time action masks, |A_valid| metrics, fleet status, and reward history.
"""

from typing import List, Tuple, Dict, Any, Optional
import pygame
import numpy as np
from environments.truck import Truck, TruckStatus
from environments.jobs import JobManager, Job


class Dashboard:
    """
    Renders high-density telemetry and action mask visualization for RL research.
    """

    def __init__(self, rect: Tuple[int, int, int, int]):
        self.rect = pygame.Rect(rect)
        self.history_rewards: List[float] = []
        self.history_valid_counts: List[int] = []
        self.max_history: int = 40

    def record_step(self, reward: float, valid_actions_count: int):
        self.history_rewards.append(float(reward))
        self.history_valid_counts.append(int(valid_actions_count))
        if len(self.history_rewards) > self.max_history:
            self.history_rewards.pop(0)
            self.history_valid_counts.pop(0)

    def draw(
        self,
        surface: pygame.Surface,
        current_step: int,
        max_steps: int,
        completed_jobs: int,
        cumulative_reward: float,
        invalid_count: int,
        action_mask: np.ndarray,
        selected_action: Optional[int],
        trucks: List[Truck],
        job_manager: JobManager,
        active_agent_name: str,
        fonts: Dict[str, pygame.font.Font],
    ):
        panel_x = self.rect.x
        panel_y = self.rect.y
        panel_w = self.rect.width
        panel_h = self.rect.height

        # Panel background
        pygame.draw.rect(surface, (18, 22, 32), self.rect)
        pygame.draw.line(surface, (40, 48, 68), (panel_x, 0), (panel_x, panel_h), width=2)

        font_title = fonts["title"]
        font_body = fonts["body"]
        font_small = fonts["small"]
        font_bold = fonts["bold"]

        curr_y = panel_y + 16

        # ==================================================================
        # 1. HEADER & ACTIVE ALGORITHM BADGE
        # ==================================================================
        title_surf = font_title.render("AEN-PPO DISPATCH TELEMETRY", True, (245, 248, 252))
        surface.blit(title_surf, (panel_x + 16, curr_y))
        curr_y += 26

        # Agent Badge
        badge_colors = {
            "aen_ppo": (0, 180, 130),
            "masked_ppo": (52, 152, 219),
            "ppo": (231, 76, 60),
            "random": (155, 89, 182),
            "manual": (241, 196, 15),
        }
        b_col = badge_colors.get(active_agent_name.lower(), (100, 110, 130))
        agent_label = f"POLICY: {active_agent_name.upper()}"
        b_rect = pygame.Rect(panel_x + 16, curr_y, font_small.size(agent_label)[0] + 16, 20)
        pygame.draw.rect(surface, b_col, b_rect, border_radius=4)
        surface.blit(font_small.render(agent_label, True, (255, 255, 255)), (b_rect.x + 8, b_rect.y + 3))

        # Simulation Step Counter
        step_text = f"Step: {current_step}/{max_steps}"
        step_surf = font_bold.render(step_text, True, (200, 210, 230))
        surface.blit(step_surf, (panel_x + panel_w - step_surf.get_width() - 16, curr_y))

        curr_y += 30

        # ==================================================================
        # 2. KEY METRICS STATS CARDS
        # ==================================================================
        card_w = (panel_w - 48) // 3
        card_h = 54

        stats = [
            ("REWARD", f"{cumulative_reward:.1f}", (46, 204, 113) if cumulative_reward >= 0 else (231, 76, 60)),
            ("DELIVERIES", f"{completed_jobs}", (52, 152, 219)),
            ("INVALID", f"{invalid_count}", (46, 204, 113) if invalid_count == 0 else (231, 76, 60)),
        ]

        for i, (label, val, col) in enumerate(stats):
            cx = panel_x + 16 + i * (card_w + 8)
            c_rect = pygame.Rect(cx, curr_y, card_w, card_h)
            pygame.draw.rect(surface, (26, 32, 46), c_rect, border_radius=6)
            pygame.draw.rect(surface, (45, 54, 76), c_rect, width=1, border_radius=6)

            surface.blit(font_small.render(label, True, (140, 155, 180)), (cx + 8, curr_y + 6))
            val_surf = font_bold.render(val, True, col)
            surface.blit(val_surf, (cx + 8, curr_y + 24))

        curr_y += card_h + 16

        # ==================================================================
        # 3. ACTION MASK & FEASIBILITY MATRIX
        # ==================================================================
        num_t = len(trucks)
        num_j = job_manager.num_jobs
        total_actions = len(action_mask) if action_mask is not None else (num_t * num_j + 1)
        wait_action_idx = total_actions - 1

        num_valid = int(np.sum(action_mask)) if action_mask is not None else 0
        mask_header = f"DYNAMIC ACTION MASK |A_valid| = {num_valid} / {total_actions}"
        surface.blit(font_bold.render(mask_header, True, (240, 245, 255)), (panel_x + 16, curr_y))
        curr_y += 20

        # Matrix: num_t Trucks x num_j Jobs + 1 WAIT
        row_step = 14 if num_t > 6 else (15 if num_t > 4 else 18)
        matrix_h = max(70, 20 + num_t * row_step + 8)
        matrix_bg = pygame.Rect(panel_x + 16, curr_y, panel_w - 32, matrix_h)
        pygame.draw.rect(surface, (24, 29, 42), matrix_bg, border_radius=6)
        pygame.draw.rect(surface, (45, 54, 76), matrix_bg, width=1, border_radius=6)

        margin_x = 36
        margin_y = 6
        avail_w = matrix_bg.width - margin_x - 90
        cell_spacing = min(22, max(12, avail_w // max(1, num_j)))
        cell_size = min(14, max(9, cell_spacing - 2))

        # Column Labels
        for j in range(num_j):
            lbl_txt = f"{j+1}" if num_j > 8 else f"J{j+1}"
            lbl = font_small.render(lbl_txt, True, (140, 155, 180))
            surface.blit(lbl, (matrix_bg.x + margin_x + j * cell_spacing, matrix_bg.y + margin_y))

        for t in range(num_t):
            t_lbl_txt = f"T{t+1}"
            lbl = font_small.render(t_lbl_txt, True, (140, 155, 180))
            surface.blit(lbl, (matrix_bg.x + 8, matrix_bg.y + margin_y + 16 + t * row_step))

            for j in range(num_j):
                action_idx = t * num_j + j
                is_valid = bool(action_mask[action_idx]) if (action_mask is not None and action_idx < len(action_mask)) else False
                is_sel = selected_action == action_idx

                cell_x = matrix_bg.x + margin_x + j * cell_spacing
                cell_y = matrix_bg.y + margin_y + 16 + t * row_step
                cell_rect = pygame.Rect(cell_x, cell_y, cell_size, cell_size)

                cell_color = (46, 204, 113) if is_valid else (45, 52, 68)
                pygame.draw.rect(surface, cell_color, cell_rect, border_radius=2)
                if is_sel:
                    pygame.draw.rect(surface, (255, 220, 0), cell_rect, width=2, border_radius=2)

        # WAIT Action LED at right of matrix
        wait_valid = bool(action_mask[wait_action_idx]) if (action_mask is not None and len(action_mask) > wait_action_idx) else False
        wait_sel = selected_action == wait_action_idx
        wait_rect = pygame.Rect(matrix_bg.right - 84, matrix_bg.y + max(10, (matrix_h - 36) // 2), 74, 36)
        wait_col = (46, 204, 113) if wait_valid else (45, 52, 68)
        pygame.draw.rect(surface, (20, 24, 34), wait_rect, border_radius=4)
        pygame.draw.rect(surface, wait_col, wait_rect, width=2, border_radius=4)
        if wait_sel:
            pygame.draw.rect(surface, (255, 220, 0), wait_rect, width=2, border_radius=4)

        surface.blit(font_small.render("WAIT ACTION", True, (160, 175, 195)), (wait_rect.x + 5, wait_rect.y + 6))
        status_txt = "VALID" if wait_valid else "BLOCKED"
        surface.blit(font_small.render(status_txt, True, wait_col), (wait_rect.x + 5, wait_rect.y + 20))

        curr_y += matrix_bg.height + 12

        # ==================================================================
        # 4. TRUCK FLEET STATUS
        # ==================================================================
        surface.blit(font_bold.render("FLEET TELEMETRY", True, (240, 245, 255)), (panel_x + 16, curr_y))
        curr_y += 18

        fleet_row_h = 16 if num_t > 6 else (20 if num_t > 4 else 24)
        fleet_h = max(50, 8 + num_t * fleet_row_h)
        fleet_bg = pygame.Rect(panel_x + 16, curr_y, panel_w - 32, fleet_h)
        pygame.draw.rect(surface, (24, 29, 42), fleet_bg, border_radius=6)
        pygame.draw.rect(surface, (45, 54, 76), fleet_bg, width=1, border_radius=6)

        for i, t in enumerate(trucks):
            row_y = fleet_bg.y + 4 + i * fleet_row_h
            t_lbl = font_bold.render(f"T{t.truck_id+1}", True, (245, 185, 20))
            surface.blit(t_lbl, (fleet_bg.x + 8, row_y + 1))

            # Status tag
            status_colors = {
                TruckStatus.IDLE: (46, 204, 113),
                TruckStatus.TO_LOAD: (52, 152, 219),
                TruckStatus.LOADING: (241, 196, 15),
                TruckStatus.TO_DUMP: (230, 126, 34),
                TruckStatus.UNLOADING: (155, 89, 182),
            }
            s_col = status_colors.get(t.status, (150, 150, 150))
            s_surf = font_small.render(t.status, True, s_col)
            surface.blit(s_surf, (fleet_bg.x + 36, row_y + 2))

            # Payload bar
            bar_x = fleet_bg.x + 124
            bar_w = 80
            bar_h = 6 if num_t > 6 else 8
            load_ratio = min(1.0, t.current_load / max(0.1, t.capacity))
            pygame.draw.rect(surface, (45, 52, 68), (bar_x, row_y + 5, bar_w, bar_h), border_radius=2)
            if load_ratio > 0:
                pygame.draw.rect(surface, (245, 185, 20), (bar_x, row_y + 5, int(bar_w * load_ratio), bar_h), border_radius=2)

            # Cap info
            cap_str = f"{t.current_load:.1f}/{t.capacity:.1f}t"
            surface.blit(font_small.render(cap_str, True, (160, 175, 195)), (bar_x + bar_w + 6, row_y + 2))

            # Trips
            trips_str = f"Done: {t.total_jobs_completed}"
            surface.blit(font_small.render(trips_str, True, (140, 155, 175)), (fleet_bg.right - 64, row_y + 2))

        curr_y += fleet_bg.height + 12

        # ==================================================================
        # 5. PENDING JOB QUEUE
        # ==================================================================
        surface.blit(font_bold.render("JOB ORDERS & DEADLINES", True, (240, 245, 255)), (panel_x + 16, curr_y))
        curr_y += 18

        job_row_h = 12 if num_j > 8 else (15 if num_j > 6 else 18)
        job_h = max(55, min(120, 6 + num_j * job_row_h))
        job_bg = pygame.Rect(panel_x + 16, curr_y, panel_w - 32, job_h)
        pygame.draw.rect(surface, (24, 29, 42), job_bg, border_radius=6)
        pygame.draw.rect(surface, (45, 54, 76), job_bg, width=1, border_radius=6)

        for slot in range(job_manager.num_jobs):
            job = job_manager.get_job(slot)
            row_y = job_bg.y + 3 + slot * job_row_h
            if row_y + job_row_h > job_bg.bottom:
                break
            j_lbl = font_small.render(f"J{slot+1}:", True, (200, 210, 230))
            surface.blit(j_lbl, (job_bg.x + 8, row_y))

            if job is not None and job.is_pending:
                rem = job.remaining_deadline(current_step)
                info_str = f"Load: {job.load_requirement:.2f}t | Rem: {rem}s"
                col = (231, 76, 60) if rem <= 5 else ((241, 196, 15) if rem <= 15 else (46, 204, 113))
                surface.blit(font_small.render(info_str, True, col), (job_bg.x + 60, row_y))

                # Mini deadline bar
                d_ratio = min(1.0, float(rem) / 40.0)
                d_bar_x = job_bg.right - 80
                pygame.draw.rect(surface, (45, 52, 68), (d_bar_x, row_y + 3, 70, 5), border_radius=2)
                pygame.draw.rect(surface, col, (d_bar_x, row_y + 3, int(70 * d_ratio), 5), border_radius=2)
            else:
                surface.blit(font_small.render("NO ORDER PENDING", True, (80, 95, 120)), (job_bg.x + 60, row_y))

        curr_y += job_bg.height + 12


        # ==================================================================
        # 6. MINI RESEARCH CURVE (Recent Rewards & |A_valid|)
        # ==================================================================
        surface.blit(font_bold.render("LIVE REWARD DYNAMICS", True, (240, 245, 255)), (panel_x + 16, curr_y))
        curr_y += 18

        chart_bg = pygame.Rect(panel_x + 16, curr_y, panel_w - 32, 80)
        pygame.draw.rect(surface, (24, 29, 42), chart_bg, border_radius=6)
        pygame.draw.rect(surface, (45, 54, 76), chart_bg, width=1, border_radius=6)

        # Baseline line at 0 reward
        zero_y = chart_bg.y + chart_bg.height // 2
        pygame.draw.line(surface, (50, 60, 80), (chart_bg.x + 4, zero_y), (chart_bg.right - 4, zero_y), 1)

        if len(self.history_rewards) >= 2:
            pts = []
            for idx, r in enumerate(self.history_rewards):
                px = chart_bg.x + 8 + int(idx * (chart_bg.width - 16) / max(1, self.max_history - 1))
                # Map reward range [-5, 15] to chart height
                norm_r = (r - (-5.0)) / (15.0 - (-5.0))
                py = chart_bg.bottom - 6 - int(np.clip(norm_r, 0.0, 1.0) * (chart_bg.height - 12))
                pts.append((px, py))
            pygame.draw.lines(surface, (0, 180, 130), False, pts, width=2)

            last_r = self.history_rewards[-1]
            surface.blit(font_small.render(f"Recent: {last_r:.1f}", True, (0, 180, 130)), (chart_bg.right - 85, chart_bg.y + 6))
