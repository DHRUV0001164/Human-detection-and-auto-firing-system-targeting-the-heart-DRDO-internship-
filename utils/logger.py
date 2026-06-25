"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Engagement Logger — CSV & Screenshot Logging                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import csv
import cv2
import json
from datetime import datetime
import config


class EngagementLogger:
    """
    Logs all engagement events to CSV and captures screenshots on firing events.
    Also generates session summary reports.
    """

    def __init__(self):
        os.makedirs(config.LOGS_DIR, exist_ok=True)

        # Session identifier
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(config.LOGS_DIR, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)

        # CSV log file
        self.csv_path = os.path.join(self.session_dir, "engagement_log.csv")
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Timestamp", "Frame", "Event", "Target_ID",
            "Heart_X", "Heart_Y", "Distance_M", "Confidence",
            "State", "Details"
        ])

        # Counters
        self.frame_count = 0
        self.event_count = 0
        self.fire_count = 0
        self.targets_detected_total = 0

        # Recent events (for HUD display)
        self.recent_events = []
        self.max_recent = 8

        print(f"[LOGGER] Session started: {self.session_id}")
        print(f"[LOGGER] Log directory: {self.session_dir}")

    def log_event(self, event_type, target_id=None, heart_pos=None,
                  distance=-1.0, confidence=0.0, state="", details=""):
        """
        Log an engagement event.

        Args:
            event_type: str — 'DETECTION', 'LOCK_ON', 'FIRE', 'LOST', 'NEW_TARGET', etc.
            target_id: str — Target identifier (e.g., 'T-001')
            heart_pos: tuple — (x, y) heart position
            distance: float — Estimated distance in meters
            confidence: float — Detection confidence
            state: str — Current engagement state
            details: str — Additional information
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        hx = heart_pos[0] if heart_pos else -1
        hy = heart_pos[1] if heart_pos else -1

        self.csv_writer.writerow([
            timestamp, self.frame_count, event_type,
            target_id or "N/A", hx, hy,
            f"{distance:.2f}", f"{confidence:.3f}",
            state, details
        ])
        self.csv_file.flush()

        self.event_count += 1

        if event_type == "FIRE":
            self.fire_count += 1

        # Add to recent events for HUD
        event_str = f"[{timestamp}] {event_type}"
        if target_id:
            event_str += f" | {target_id}"
        if distance > 0:
            event_str += f" | {distance:.1f}m"

        self.recent_events.append(event_str)
        if len(self.recent_events) > self.max_recent:
            self.recent_events.pop(0)

    def capture_screenshot(self, frame, event_type="FIRE", target_id=None):
        """
        Save a screenshot on important events (firing, lock-on, etc.)

        Args:
            frame: numpy array — Current video frame
            event_type: str — Type of event
            target_id: str — Target ID
        """
        screenshots_dir = os.path.join(self.session_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        tid = target_id or "unknown"
        filename = f"{event_type}_{tid}_{timestamp}.jpg"
        filepath = os.path.join(screenshots_dir, filename)

        cv2.imwrite(filepath, frame)

    def increment_frame(self):
        """Increment frame counter."""
        self.frame_count += 1

    def get_recent_events(self):
        """Get list of recent event strings for HUD display."""
        return self.recent_events.copy()

    def generate_summary(self):
        """Generate session summary report."""
        summary = {
            "session_id": self.session_id,
            "total_frames": self.frame_count,
            "total_events": self.event_count,
            "total_fires": self.fire_count,
            "total_targets_detected": self.targets_detected_total,
            "duration_frames": self.frame_count,
            "log_file": self.csv_path
        }

        summary_path = os.path.join(self.session_dir, "session_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  SESSION SUMMARY — {self.session_id}")
        print(f"{'='*60}")
        print(f"  Total Frames Processed : {self.frame_count}")
        print(f"  Total Events Logged    : {self.event_count}")
        print(f"  Total Fires            : {self.fire_count}")
        print(f"  Total Targets Detected : {self.targets_detected_total}")
        print(f"  Log File               : {self.csv_path}")
        print(f"  Summary File           : {summary_path}")
        print(f"{'='*60}\n")

        return summary

    def close(self):
        """Close the logger and generate summary."""
        self.generate_summary()
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
        print(f"[LOGGER] Session ended: {self.session_id}")
