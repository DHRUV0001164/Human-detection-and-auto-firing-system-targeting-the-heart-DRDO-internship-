"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Heart Estimator — Anatomical Heart Position Calculator      ║
╚══════════════════════════════════════════════════════════════════════╝

The heart position is estimated using anatomical proportions:
- When both shoulders AND hips are visible: 
    Heart ≈ 30% down from shoulder midpoint toward hip midpoint, 
    shifted slightly left of center.
- When only shoulders are visible (fallback):
    Heart ≈ shoulder midpoint + (35% of shoulder width) downward,
    shifted slightly left.
"""

import math
import config
from utils.geometry import point_distance, midpoint


class HeartEstimate:
    """Container for a heart position estimate with confidence."""

    def __init__(self, x=0, y=0, confidence=0.0, zone_radius=20, method="none"):
        self.x = int(x)
        self.y = int(y)
        self.confidence = confidence
        self.zone_radius = int(zone_radius)
        self.method = method  # 'full_torso', 'shoulder_only', 'box_estimate'

    @property
    def position(self):
        return (self.x, self.y)

    @property
    def is_valid(self):
        return self.confidence > 0.2 and self.x > 0 and self.y > 0

    def __repr__(self):
        return (f"HeartEstimate(pos=({self.x},{self.y}), "
                f"conf={self.confidence:.2f}, method={self.method})")


class HeartEstimator:
    """
    Estimates the anatomical heart position from body keypoints.
    Uses proportional calculations so it works at any distance/scale.
    """

    def __init__(self):
        # Smoothing history per target
        self._history = {}  # target_id -> list of (x, y)
        self._max_history = 8

    def estimate(self, detection, target_id=None):
        """
        Estimate heart position from a DetectionResult.

        Args:
            detection: DetectionResult object from the detector
            target_id: Optional target ID for smoothing across frames

        Returns:
            HeartEstimate: Estimated heart position with confidence
        """
        estimate = None

        # ── Method 1: Full Torso (shoulders + hips) ──────────────
        if detection.has_shoulders and detection.has_hips:
            estimate = self._estimate_full_torso(detection)

        # ── Method 2: Shoulders Only (fallback) ──────────────────
        elif detection.has_shoulders:
            estimate = self._estimate_from_shoulders(detection)

        # ── Method 3: Bounding Box Only (last resort) ────────────
        elif detection.best_box is not None:
            estimate = self._estimate_from_box(detection)

        # ── No estimation possible ───────────────────────────────
        if estimate is None:
            return HeartEstimate()

        # Apply smoothing if we have a target ID
        if target_id is not None and estimate.is_valid:
            estimate = self._apply_smoothing(target_id, estimate)

        return estimate

    def _estimate_full_torso(self, detection):
        """
        Best method: Uses both shoulders and hips.
        Heart is ~30% down the torso, slightly left of center.
        """
        ls = detection.left_shoulder
        rs = detection.right_shoulder
        lh = detection.left_hip
        rh = detection.right_hip

        # Shoulder midpoint
        shoulder_mid = midpoint(ls, rs)
        # Hip midpoint
        hip_mid = midpoint(lh, rh)

        # Torso vector (from shoulders to hips)
        torso_dx = hip_mid[0] - shoulder_mid[0]
        torso_dy = hip_mid[1] - shoulder_mid[1]
        torso_length = point_distance(shoulder_mid, hip_mid)

        if torso_length < 5:  # Too small to be reliable
            return None

        # Heart position: 30% down the torso
        heart_x = shoulder_mid[0] + torso_dx * config.HEART_VERTICAL_RATIO
        heart_y = shoulder_mid[1] + torso_dy * config.HEART_VERTICAL_RATIO

        # Shift slightly to the left (anatomically, heart is left of center)
        shoulder_width = point_distance(ls, rs)
        heart_x += shoulder_width * config.HEART_HORIZONTAL_OFFSET

        # Zone radius proportional to shoulder width
        zone_radius = shoulder_width * config.HEART_ZONE_RADIUS_RATIO

        # Confidence based on keypoint visibility
        confidence = 0.95

        return HeartEstimate(
            x=heart_x, y=heart_y,
            confidence=confidence,
            zone_radius=max(zone_radius, 12),
            method="full_torso"
        )

    def _estimate_from_shoulders(self, detection):
        """
        Fallback: Only shoulders visible, no hips.
        Uses shoulder span as scale reference.
        """
        ls = detection.left_shoulder
        rs = detection.right_shoulder

        shoulder_mid = midpoint(ls, rs)
        shoulder_width = point_distance(ls, rs)

        if shoulder_width < 10:  # Too small
            return None

        # Heart is below the shoulder midpoint
        heart_x = shoulder_mid[0] + shoulder_width * config.HEART_FALLBACK_LEFT_RATIO
        heart_y = shoulder_mid[1] + shoulder_width * config.HEART_FALLBACK_DOWN_RATIO

        # Zone radius
        zone_radius = shoulder_width * config.HEART_ZONE_RADIUS_RATIO

        # Lower confidence since we don't have hips
        confidence = 0.75

        return HeartEstimate(
            x=heart_x, y=heart_y,
            confidence=confidence,
            zone_radius=max(zone_radius, 10),
            method="shoulder_only"
        )

    def _estimate_from_box(self, detection):
        """
        Last resort: Use bounding box proportions.
        The heart is approximately at the upper-center-left of the body box.
        """
        box = detection.best_box
        if box is None:
            return None

        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < 20 or box_h < 30:
            return None

        # Estimate head takes ~15% of body height, shoulders at ~20%
        # Heart at ~30% from top
        heart_x = x1 + box_w * 0.47  # Slightly left of center
        heart_y = y1 + box_h * 0.30  # 30% from top

        zone_radius = box_w * 0.12

        # Low confidence — this is a rough estimate
        confidence = 0.45

        return HeartEstimate(
            x=heart_x, y=heart_y,
            confidence=confidence,
            zone_radius=max(zone_radius, 8),
            method="box_estimate"
        )

    def _apply_smoothing(self, target_id, estimate):
        """
        Apply temporal smoothing to reduce jitter.
        Uses exponential moving average over recent positions.
        """
        if target_id not in self._history:
            self._history[target_id] = []

        history = self._history[target_id]
        history.append((estimate.x, estimate.y))

        if len(history) > self._max_history:
            history.pop(0)

        # Weighted average (more recent = higher weight)
        if len(history) >= 2:
            alpha = 0.5
            sx, sy = history[0]
            for px, py in history[1:]:
                sx = alpha * px + (1 - alpha) * sx
                sy = alpha * py + (1 - alpha) * sy

            estimate.x = int(sx)
            estimate.y = int(sy)

        return estimate

    def clear_history(self, target_id):
        """Remove smoothing history for a target."""
        if target_id in self._history:
            del self._history[target_id]

    def clear_all(self):
        """Clear all smoothing history."""
        self._history.clear()
