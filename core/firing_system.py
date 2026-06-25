"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Firing System — Simulated Auto-Targeting & Engagement       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import config
from utils.geometry import point_distance


class EngagementState:
    """Engagement states for a target."""
    SAFE = "SAFE"
    TRACKING = "TRACKING"
    LOCKING = "LOCKING"
    LOCKED = "LOCKED"
    FIRING = "FIRING"
    COOLDOWN = "COOLDOWN"


class TargetEngagement:
    """Tracks the engagement state for a specific target."""

    def __init__(self, target_id):
        self.target_id = target_id
        self.state = EngagementState.SAFE
        self.lock_progress = 0          # 0 to LOCK_ON_FRAMES
        self.lock_positions = []        # Recent heart positions during lock-on
        self.cooldown_remaining = 0
        self.fire_frame_counter = 0     # Frames the fire animation plays
        self.total_fires = 0
        self.last_fire_time = 0

    @property
    def lock_percentage(self):
        """Lock-on progress as percentage."""
        return min(100, int((self.lock_progress / config.LOCK_ON_FRAMES) * 100))

    @property
    def is_locked(self):
        return self.state == EngagementState.LOCKED

    @property
    def is_firing(self):
        return self.state == EngagementState.FIRING


class FiringSystem:
    """
    Simulated auto-firing system.
    Manages target lock-on, firing simulation, and cooldown.
    """

    def __init__(self, logger=None):
        self.enabled = config.FIRING_ENABLED
        self.safety_on = True          # Safety must be explicitly turned off
        self.engagements = {}          # target_id -> TargetEngagement
        self.logger = logger
        self.total_rounds_fired = 0
        self.fire_animation_frames = 8  # Duration of fire animation

        # Sound control
        self.sound_enabled = config.ENABLE_SOUND
        self._last_sound_time = 0

        print(f"[FIRING] System initialized | Safety: {'ON' if self.safety_on else 'OFF'} | "
              f"Auto-fire: {'ENABLED' if self.enabled else 'DISABLED'}")

    def toggle_auto_fire(self):
        """Toggle auto-fire mode."""
        self.enabled = not self.enabled
        status = "ENABLED" if self.enabled else "DISABLED"
        print(f"[FIRING] Auto-fire {status}")
        self._play_sound(800, 100)
        return self.enabled

    def toggle_safety(self):
        """Toggle safety lock."""
        self.safety_on = not self.safety_on
        status = "ON" if self.safety_on else "OFF"
        print(f"[FIRING] Safety {status}")
        self._play_sound(600 if self.safety_on else 1000, 150)
        return self.safety_on

    def update(self, tracks, priority_target=None):
        """
        Update all engagement states based on current tracks.

        Args:
            tracks: list[Track] — Active tracks from the tracker
            priority_target: Track — The primary target for engagement

        Returns:
            dict: target_id -> TargetEngagement for all active engagements
        """
        active_ids = {t.label for t in tracks}

        # Clean up engagements for lost targets
        lost_ids = [tid for tid in self.engagements if tid not in active_ids]
        for tid in lost_ids:
            if self.logger:
                self.logger.log_event("DISENGAGE", target_id=tid,
                                      state=self.engagements[tid].state,
                                      details="Target lost")
            del self.engagements[tid]

        # Ensure all active tracks have engagement objects
        for track in tracks:
            if track.label not in self.engagements:
                self.engagements[track.label] = TargetEngagement(track.label)

        # Update each engagement
        for track in tracks:
            eng = self.engagements[track.label]
            is_priority = (priority_target is not None and
                           track.label == priority_target.label)

            self._update_engagement(eng, track, is_priority)

        return self.engagements

    def _update_engagement(self, eng, track, is_priority):
        """Update a single target's engagement state."""

        # Handle cooldown countdown
        if eng.cooldown_remaining > 0:
            eng.cooldown_remaining -= 1
            if eng.cooldown_remaining <= 0:
                eng.state = EngagementState.TRACKING
            else:
                eng.state = EngagementState.COOLDOWN
            return

        # Handle fire animation
        if eng.fire_frame_counter > 0:
            eng.fire_frame_counter -= 1
            eng.state = EngagementState.FIRING
            if eng.fire_frame_counter <= 0:
                eng.cooldown_remaining = config.FIRE_COOLDOWN_FRAMES
                eng.state = EngagementState.COOLDOWN
            return

        # Safety check
        if self.safety_on:
            eng.state = EngagementState.SAFE
            eng.lock_progress = 0
            return

        # Not the priority target? Just track.
        if not is_priority:
            eng.state = EngagementState.TRACKING
            eng.lock_progress = 0
            return

        # Priority target — attempt lock-on
        if track.heart_pos is None:
            eng.state = EngagementState.TRACKING
            eng.lock_progress = 0
            return

        # Check if heart position is stable (for lock-on)
        eng.lock_positions.append(track.heart_pos)
        if len(eng.lock_positions) > config.LOCK_ON_FRAMES:
            eng.lock_positions.pop(0)

        # Check stability of recent positions
        is_stable = True
        if len(eng.lock_positions) >= 3:
            recent = eng.lock_positions[-3:]
            for i in range(1, len(recent)):
                if point_distance(recent[i], recent[i-1]) > config.LOCK_ON_STABILITY_PX:
                    is_stable = False
                    break

        if is_stable:
            eng.lock_progress += 1
            eng.state = EngagementState.LOCKING
        else:
            eng.lock_progress = max(0, eng.lock_progress - 2)
            eng.state = EngagementState.TRACKING

        # Check if lock-on is complete
        if eng.lock_progress >= config.LOCK_ON_FRAMES:
            eng.state = EngagementState.LOCKED

            # Auto-fire if enabled
            if self.enabled and not self.safety_on:
                self._fire(eng, track)

    def manual_fire(self, priority_target):
        """
        Manual single-shot fire at the priority target.

        Args:
            priority_target: Track — Target to fire at

        Returns:
            bool: True if fire was executed
        """
        if self.safety_on:
            print("[FIRING] SAFETY ON — Cannot fire!")
            self._play_sound(300, 200)
            return False

        if priority_target is None:
            print("[FIRING] No target to fire at!")
            return False

        eng = self.engagements.get(priority_target.label)
        if eng is None:
            return False

        if eng.cooldown_remaining > 0:
            print("[FIRING] Cooldown active — wait!")
            return False

        self._fire(eng, priority_target)
        return True

    def _fire(self, eng, track):
        """Execute a simulated fire event."""
        eng.state = EngagementState.FIRING
        eng.fire_frame_counter = self.fire_animation_frames
        eng.total_fires += 1
        eng.last_fire_time = time.time()
        eng.lock_progress = 0
        self.total_rounds_fired += 1

        print(f"[FIRING] ██ FIRE ██ → {track.label} @ "
              f"({track.heart_pos[0]}, {track.heart_pos[1]}) | "
              f"Distance: {track.distance_m:.1f}m | "
              f"Round #{self.total_rounds_fired}")

        # Log the event
        if self.logger:
            self.logger.log_event(
                "FIRE",
                target_id=track.label,
                heart_pos=track.heart_pos,
                distance=track.distance_m,
                confidence=track.heart_confidence,
                state="FIRING",
                details=f"Round #{self.total_rounds_fired}"
            )

        # Play firing sound
        self._play_sound(1500, 50)

    def _play_sound(self, frequency, duration_ms):
        """Play a system beep sound (Windows only)."""
        if not self.sound_enabled:
            return
        try:
            import winsound
            winsound.Beep(frequency, duration_ms)
        except Exception:
            pass  # Silently fail on non-Windows systems

    def get_engagement(self, target_id):
        """Get engagement state for a target."""
        return self.engagements.get(target_id)

    def get_system_status(self):
        """Get a status string for HUD display."""
        if self.safety_on:
            return "SAFETY ON"
        elif self.enabled:
            return "AUTO-FIRE ARMED"
        else:
            return "MANUAL MODE"

    def reset(self):
        """Reset all engagements."""
        self.engagements.clear()
        self.total_rounds_fired = 0
