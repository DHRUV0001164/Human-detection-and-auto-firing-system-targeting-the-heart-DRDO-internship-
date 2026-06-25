"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Military HUD Overlay — Tactical Display System              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import math
import numpy as np
import time
import config
from ui.colors import *
from core.firing_system import EngagementState


class HUD:
    """Military-style Heads-Up Display overlay."""

    def __init__(self):
        self.frame_times = []
        self.start_time = time.time()
        self.frame_count = 0

    @property
    def fps(self):
        if len(self.frame_times) < 2:
            return 0
        return len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0] + 1e-9)

    def tick(self):
        self.frame_count += 1
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

    def render(self, frame, tracks, engagements, firing_system,
               priority_target, heart_estimator_results, logger, inference_ms):
        """Render full HUD overlay on the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Draw layers
        if config.HUD_SHOW_GRID:
            self._draw_grid(overlay, w, h)
        self._draw_detections(overlay, tracks, engagements, heart_estimator_results)
        self._draw_crosshair(overlay, priority_target, engagements, w, h)
        self._draw_firing_effects(overlay, priority_target, engagements, w, h)
        self._draw_status_bar(overlay, tracks, firing_system, inference_ms, w, h)
        self._draw_target_panel(overlay, priority_target, engagements, w, h)
        self._draw_event_log(overlay, logger, w, h)
        if config.HUD_SHOW_MINIMAP:
            self._draw_minimap(overlay, tracks, w, h)
        self._draw_border(overlay, w, h)

        # Blend overlay
        cv2.addWeighted(overlay, config.HUD_OPACITY, frame,
                        1 - config.HUD_OPACITY, 0, frame)
        # Re-draw critical elements at full opacity on blended frame
        self._draw_crosshair(frame, priority_target, engagements, w, h)
        self._draw_firing_effects(frame, priority_target, engagements, w, h)
        self._draw_status_bar(frame, tracks, firing_system, inference_ms, w, h)
        self._draw_target_panel(frame, priority_target, engagements, w, h)

        return frame

    # ── DETECTIONS ────────────────────────────────────────────────
    def _draw_detections(self, frame, tracks, engagements, heart_results):
        for track in tracks:
            det = track.detection
            eng = engagements.get(track.label)
            state_color = self._state_color(eng)

            # Body bounding box
            if track.box:
                x1, y1, x2, y2 = track.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)
                # Corner brackets
                blen = 15
                for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(frame, (cx, cy), (cx + blen*dx, cy), state_color, 2)
                    cv2.line(frame, (cx, cy), (cx, cy + blen*dy), state_color, 2)
                # Target label
                label = track.label
                if track.distance_m > 0:
                    label += f" {track.distance_m:.1f}m"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1, cv2.LINE_AA)

            # Face box
            if det and det.face_box:
                fx1, fy1, fx2, fy2 = det.face_box
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), COLOR_FACE, 2)
                cv2.putText(frame, f"FACE {det.face_conf:.0%}", (fx1, fy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_FACE, 1, cv2.LINE_AA)

            # Shoulder keypoints
            if det and det.has_shoulders:
                ls, rs = det.left_shoulder, det.right_shoulder
                cv2.circle(frame, (int(ls[0]), int(ls[1])), 5, COLOR_SHOULDER, -1)
                cv2.circle(frame, (int(rs[0]), int(rs[1])), 5, COLOR_SHOULDER, -1)
                cv2.line(frame, (int(ls[0]), int(ls[1])),
                         (int(rs[0]), int(rs[1])), COLOR_SKELETON, 1, cv2.LINE_AA)

            # Hip keypoints
            if det and det.has_hips:
                lh, rh = det.left_hip, det.right_hip
                cv2.circle(frame, (int(lh[0]), int(lh[1])), 5, COLOR_HIP, -1)
                cv2.circle(frame, (int(rh[0]), int(rh[1])), 5, COLOR_HIP, -1)

            # Heart marker
            if track.heart_pos:
                hx, hy = track.heart_pos
                hr = heart_results.get(track.label)
                zone_r = hr.zone_radius if hr else 15

                # Pulsing glow
                pulse = int(5 * math.sin(time.time() * 6)) + zone_r
                cv2.circle(frame, (hx, hy), pulse, COLOR_HEART_GLOW, 1, cv2.LINE_AA)
                cv2.circle(frame, (hx, hy), zone_r, COLOR_HEART_ZONE, 2, cv2.LINE_AA)
                # Inner crosshair
                cs = 8
                cv2.line(frame, (hx - cs, hy), (hx + cs, hy), COLOR_HEART, 2)
                cv2.line(frame, (hx, hy - cs), (hx, hy + cs), COLOR_HEART, 2)
                cv2.putText(frame, "HEART", (hx - 22, hy - zone_r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HEART, 1, cv2.LINE_AA)

            # Trajectory trail
            if config.HUD_SHOW_TRAJECTORY and len(track.center_history) > 1:
                pts = track.center_history
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    color = (0, int(200 * alpha), 0)
                    cv2.line(frame, pts[i-1], pts[i], color, 1, cv2.LINE_AA)

    # ── CROSSHAIR ─────────────────────────────────────────────────
    def _draw_crosshair(self, frame, priority, engagements, w, h):
        if priority and priority.heart_pos:
            cx, cy = priority.heart_pos
            eng = engagements.get(priority.label)
        else:
            cx, cy = w // 2, h // 2
            eng = None

        color = CROSSHAIR_IDLE
        size = config.HUD_CROSSHAIR_SIZE
        thickness = config.HUD_CROSSHAIR_THICKNESS

        if eng:
            if eng.state == EngagementState.LOCKED:
                color = CROSSHAIR_LOCKED
                size = int(size * 0.7)
            elif eng.state == EngagementState.FIRING:
                color = CROSSHAIR_FIRE
                size = int(size * 0.5)
                thickness = 3
            elif eng.state == EngagementState.LOCKING:
                color = CROSSHAIR_TRACKING
                # Animate size based on lock progress
                progress = eng.lock_percentage / 100.0
                size = int(size * (1.0 - 0.4 * progress))

        gap = 6
        # Crosshair lines with gap
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, thickness, cv2.LINE_AA)

        # Lock-on arc
        if eng and eng.state == EngagementState.LOCKING:
            angle = int(360 * eng.lock_percentage / 100)
            cv2.ellipse(frame, (cx, cy), (size + 5, size + 5),
                        -90, 0, angle, CROSSHAIR_TRACKING, 2, cv2.LINE_AA)

        # Lock diamond
        if eng and eng.state in (EngagementState.LOCKED, EngagementState.FIRING):
            d = size - 5
            pts = np.array([[cx, cy-d], [cx+d, cy], [cx, cy+d], [cx-d, cy]], np.int32)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

    # ── FIRING EFFECTS ────────────────────────────────────────────
    def _draw_firing_effects(self, frame, priority, engagements, w, h):
        if not priority or not priority.heart_pos:
            return
        eng = engagements.get(priority.label)
        if not eng or eng.state != EngagementState.FIRING:
            return

        hx, hy = priority.heart_pos
        # Laser line from bottom center
        cv2.line(frame, (w // 2, h), (hx, hy), LASER_COLOR, 2, cv2.LINE_AA)
        # Impact ring
        ring_r = 20 + eng.fire_frame_counter * 3
        cv2.circle(frame, (hx, hy), ring_r, IMPACT_COLOR, 2, cv2.LINE_AA)
        cv2.circle(frame, (hx, hy), 5, MUZZLE_FLASH_COLOR, -1)
        # Flash overlay
        if eng.fire_frame_counter > 5:
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
            cv2.addWeighted(flash, 0.1, frame, 0.9, 0, frame)

    # ── STATUS BAR (top) ──────────────────────────────────────────
    def _draw_status_bar(self, frame, tracks, firing_system, inference_ms, w, h):
        bar_h = 35
        cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.line(frame, (0, bar_h), (w, bar_h), HUD_GREEN_DIM, 1)

        y = 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45
        # System name
        cv2.putText(frame, "DRDO TARGETING SYSTEM v2.0", (10, y),
                    font, fs, HUD_GREEN, 1, cv2.LINE_AA)
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (320, y),
                    font, fs, HUD_CYAN, 1, cv2.LINE_AA)
        # Inference
        cv2.putText(frame, f"INF: {inference_ms:.0f}ms", (420, y),
                    font, fs, HUD_CYAN, 1, cv2.LINE_AA)
        # Targets
        confirmed = len([t for t in tracks if t.is_confirmed])
        cv2.putText(frame, f"TGT: {confirmed}/{len(tracks)}", (540, y),
                    font, fs, HUD_AMBER if confirmed > 0 else HUD_GREEN, 1, cv2.LINE_AA)
        # Firing status
        status = firing_system.get_system_status()
        sc = STATE_SAFE if firing_system.safety_on else (
            CROSSHAIR_LOCKED if firing_system.enabled else HUD_AMBER)
        cv2.putText(frame, status, (660, y), font, fs, sc, 1, cv2.LINE_AA)
        # Rounds
        cv2.putText(frame, f"RND: {firing_system.total_rounds_fired}", (w - 120, y),
                    font, fs, HUD_CYAN, 1, cv2.LINE_AA)
        # Time
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        cv2.putText(frame, f"{mins:02d}:{secs:02d}", (w - 55, y),
                    font, fs, HUD_GREEN_DIM, 1, cv2.LINE_AA)

    # ── TARGET INFO PANEL (right side) ────────────────────────────
    def _draw_target_panel(self, frame, priority, engagements, w, h):
        if not priority:
            return
        eng = engagements.get(priority.label)
        if not eng:
            return

        pw = min(config.HUD_PANEL_WIDTH, 240)
        px = w - pw - 10
        py = 50
        ph = 180
        # Panel background
        panel_bg = frame[py:py+ph, px:px+pw].copy()
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 0, 0), -1)
        cv2.addWeighted(frame[py:py+ph, px:px+pw], 0.8, panel_bg, 0.2, 0,
                        frame[py:py+ph, px:px+pw])
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), HUD_GREEN_DIM, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.42
        lh = 20
        tx = px + 10
        ty = py + 20
        color = self._state_color(eng)

        cv2.putText(frame, f"TARGET: {priority.label}", (tx, ty),
                    font, 0.5, color, 1, cv2.LINE_AA)
        ty += lh
        cv2.putText(frame, f"State: {eng.state}", (tx, ty),
                    font, fs, color, 1, cv2.LINE_AA)
        ty += lh
        dist_str = f"{priority.distance_m:.1f}m" if priority.distance_m > 0 else "N/A"
        cv2.putText(frame, f"Distance: {dist_str}", (tx, ty),
                    font, fs, HUD_CYAN, 1, cv2.LINE_AA)
        ty += lh
        cv2.putText(frame, f"Heart Conf: {priority.heart_confidence:.0%}", (tx, ty),
                    font, fs, HUD_CYAN, 1, cv2.LINE_AA)
        ty += lh
        # Lock progress bar
        cv2.putText(frame, f"Lock: {eng.lock_percentage}%", (tx, ty),
                    font, fs, HUD_CYAN, 1, cv2.LINE_AA)
        ty += 5
        bar_w = pw - 20
        cv2.rectangle(frame, (tx, ty), (tx + bar_w, ty + 8), MID_GRAY, -1)
        fill = int(bar_w * eng.lock_percentage / 100)
        cv2.rectangle(frame, (tx, ty), (tx + fill, ty + 8), color, -1)
        ty += lh
        cv2.putText(frame, f"Fires: {eng.total_fires}  |  V:({priority.velocity[0]:.0f},{priority.velocity[1]:.0f})",
                    (tx, ty), font, 0.35, HUD_GREEN_DIM, 1, cv2.LINE_AA)

    # ── EVENT LOG (bottom left) ───────────────────────────────────
    def _draw_event_log(self, frame, logger, w, h):
        if not logger:
            return
        events = logger.get_recent_events()
        if not events:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.33
        lh = 15
        x = 10
        y = h - 10 - len(events) * lh
        for evt in events:
            cv2.putText(frame, evt, (x, y), font, fs, HUD_GREEN_DIM, 1, cv2.LINE_AA)
            y += lh

    # ── MINIMAP (bottom right) ────────────────────────────────────
    def _draw_minimap(self, frame, tracks, w, h):
        ms = config.HUD_MINIMAP_SIZE
        mx = w - ms - 10
        my = h - ms - 10
        cv2.rectangle(frame, (mx, my), (mx+ms, my+ms), MINIMAP_BG, -1)
        cv2.rectangle(frame, (mx, my), (mx+ms, my+ms), MINIMAP_BORDER, 1)
        # Map target positions
        for track in tracks:
            if track.box:
                cx, cy = track.center
                rx = int(mx + (cx / w) * ms)
                ry = int(my + (cy / h) * ms)
                cv2.circle(frame, (rx, ry), 3, MINIMAP_TARGET, -1)
                cv2.putText(frame, track.label[-3:], (rx+4, ry+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, MINIMAP_TARGET, 1)
        # Camera center marker
        cv2.drawMarker(frame, (mx + ms//2, my + ms//2), MINIMAP_SELF,
                       cv2.MARKER_DIAMOND, 6, 1)

    # ── GRID ──────────────────────────────────────────────────────
    def _draw_grid(self, frame, w, h):
        sp = config.HUD_GRID_SPACING
        for x in range(0, w, sp):
            cv2.line(frame, (x, 0), (x, h), HUD_GREEN_DARK, 1)
        for y in range(0, h, sp):
            cv2.line(frame, (0, y), (w, y), HUD_GREEN_DARK, 1)

    # ── BORDER ────────────────────────────────────────────────────
    def _draw_border(self, frame, w, h):
        cv2.rectangle(frame, (0, 0), (w-1, h-1), HUD_GREEN_DIM, 1)
        # Corner decorations
        cl = 30
        for cx, cy, dx, dy in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + cl*dx, cy), HUD_GREEN, 2)
            cv2.line(frame, (cx, cy), (cx, cy + cl*dy), HUD_GREEN, 2)

    # ── HELPERS ───────────────────────────────────────────────────
    def _state_color(self, eng):
        if eng is None:
            return HUD_GREEN
        mapping = {
            EngagementState.SAFE: STATE_SAFE,
            EngagementState.TRACKING: STATE_TRACKING,
            EngagementState.LOCKING: STATE_LOCKING,
            EngagementState.LOCKED: STATE_LOCKED,
            EngagementState.FIRING: STATE_FIRING,
            EngagementState.COOLDOWN: STATE_COOLDOWN,
        }
        return mapping.get(eng.state, HUD_GREEN)
