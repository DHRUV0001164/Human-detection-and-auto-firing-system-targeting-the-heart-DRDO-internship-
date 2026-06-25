"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Multi-Target Tracker — IoU-Based Tracking with IDs         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from utils.geometry import calculate_iou, box_center, point_distance
import config


class TrackState:
    """Track lifecycle states."""
    NEW = "NEW"
    TRACKING = "TRACKING"
    LOST = "LOST"
    DELETED = "DELETED"


class Track:
    """
    Represents a single tracked target across frames.
    """
    _next_id = 1

    def __init__(self, box, detection=None):
        self.id = Track._next_id
        Track._next_id += 1
        self.label = f"T-{self.id:03d}"

        # Bounding box
        self.box = box  # (x1, y1, x2, y2)
        self.detection = detection

        # State management
        self.state = TrackState.NEW
        self.age = 0               # Total frames since creation
        self.hits = 1              # Consecutive frames with detection
        self.time_since_update = 0 # Frames since last matched detection

        # Position history for trajectory
        self.center_history = [box_center(box)]
        self.max_history = config.HUD_TRAJECTORY_LENGTH

        # Heart position (updated externally)
        self.heart_pos = None
        self.heart_confidence = 0.0

        # Velocity estimation (pixels per frame)
        self.velocity = (0.0, 0.0)

        # Distance estimate (meters)
        self.distance_m = -1.0

    @property
    def center(self):
        """Get current center of bounding box."""
        return box_center(self.box)

    @property
    def is_confirmed(self):
        """Track is confirmed after minimum hits."""
        return self.hits >= config.TRACKER_MIN_HITS

    @property
    def is_lost(self):
        """Track is lost if not updated for too long."""
        return self.time_since_update > config.TRACKER_MAX_AGE

    def update(self, box, detection=None):
        """
        Update track with new detection.

        Args:
            box: (x1, y1, x2, y2)
            detection: DetectionResult object
        """
        old_center = self.center
        self.box = box
        self.detection = detection
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

        # Update state
        if self.is_confirmed:
            self.state = TrackState.TRACKING

        # Update center history
        new_center = self.center
        self.center_history.append(new_center)
        if len(self.center_history) > self.max_history:
            self.center_history.pop(0)

        # Estimate velocity
        self.velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )

    def mark_lost(self):
        """Mark as not updated this frame."""
        self.time_since_update += 1
        self.age += 1
        self.hits = 0

        if self.time_since_update > 3:
            self.state = TrackState.LOST

        # Predict position using velocity
        if self.velocity != (0, 0):
            cx, cy = self.center
            predicted_cx = int(cx + self.velocity[0])
            predicted_cy = int(cy + self.velocity[1])
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
            self.box = (
                predicted_cx - w // 2,
                predicted_cy - h // 2,
                predicted_cx + w // 2,
                predicted_cy + h // 2
            )


class MultiTargetTracker:
    """
    IoU-based multi-target tracker.
    Associates detections across frames using bounding box overlap.
    Assigns persistent IDs to each tracked person.
    """

    def __init__(self):
        self.tracks = []
        self.total_targets_created = 0
        print("[TRACKER] Multi-target tracker initialized ✓")

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: list[DetectionResult] — Current frame detections

        Returns:
            list[Track]: Active tracks (confirmed + recently lost)
        """
        # Get bounding boxes from detections
        det_boxes = []
        for det in detections:
            box = det.best_box
            if box is not None:
                det_boxes.append((box, det))

        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for box, det in det_boxes:
                self._create_track(box, det)
            return self.get_active_tracks()

        # If no new detections, mark all tracks as lost
        if len(det_boxes) == 0:
            for track in self.tracks:
                track.mark_lost()
            self._cleanup()
            return self.get_active_tracks()

        # ── Hungarian-style IoU matching ──────────────────────────
        # Build IoU matrix
        num_tracks = len(self.tracks)
        num_dets = len(det_boxes)
        iou_matrix = np.zeros((num_tracks, num_dets))

        for t, track in enumerate(self.tracks):
            for d, (box, det) in enumerate(det_boxes):
                iou_matrix[t, d] = calculate_iou(track.box, box)

        # Greedy matching (match highest IoU first)
        matched_tracks = set()
        matched_dets = set()
        matches = []

        # Sort all IoU values and match greedily
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < config.TRACKER_IOU_THRESHOLD:
                break

            t_idx, d_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

            if t_idx in matched_tracks or d_idx in matched_dets:
                iou_matrix[t_idx, d_idx] = 0
                continue

            matches.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            iou_matrix[t_idx, :] = 0
            iou_matrix[:, d_idx] = 0

        # Update matched tracks
        for t_idx, d_idx in matches:
            box, det = det_boxes[d_idx]
            self.tracks[t_idx].update(box, det)

        # Mark unmatched tracks as lost
        for t in range(num_tracks):
            if t not in matched_tracks:
                self.tracks[t].mark_lost()

        # Create new tracks for unmatched detections
        for d in range(num_dets):
            if d not in matched_dets:
                box, det = det_boxes[d]
                self._create_track(box, det)

        # Cleanup dead tracks
        self._cleanup()

        return self.get_active_tracks()

    def _create_track(self, box, detection=None):
        """Create a new track."""
        if len(self.tracks) >= config.TRACKER_MAX_TARGETS:
            return

        track = Track(box, detection)
        self.tracks.append(track)
        self.total_targets_created += 1

    def _cleanup(self):
        """Remove dead tracks."""
        self.tracks = [t for t in self.tracks if not t.is_lost]

    def get_active_tracks(self):
        """Get all active (non-deleted) tracks."""
        return [t for t in self.tracks if t.state != TrackState.DELETED]

    def get_confirmed_tracks(self):
        """Get only confirmed tracks (tracked for minimum frames)."""
        return [t for t in self.tracks if t.is_confirmed and t.state == TrackState.TRACKING]

    def get_priority_target(self):
        """
        Get the highest priority target for engagement.
        Priority: closest confirmed target with valid heart position.
        """
        confirmed = self.get_confirmed_tracks()
        if not confirmed:
            return None

        # Prefer targets with heart positions, then closest
        with_heart = [t for t in confirmed if t.heart_pos is not None]
        if with_heart:
            # Sort by distance (closest first), then by confidence
            with_heart.sort(key=lambda t: (
                t.distance_m if t.distance_m > 0 else 9999,
                -t.heart_confidence
            ))
            return with_heart[0]

        # Fallback: return track with largest bounding box (likely closest)
        confirmed.sort(key=lambda t: -(
            (t.box[2] - t.box[0]) * (t.box[3] - t.box[1])
        ))
        return confirmed[0]

    def reset(self):
        """Reset all tracks."""
        self.tracks.clear()
        Track._next_id = 1
        self.total_targets_created = 0
