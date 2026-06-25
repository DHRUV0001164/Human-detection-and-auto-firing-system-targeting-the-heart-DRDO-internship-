"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Configuration Module                                        ║
║         All tunable parameters in one place                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os

# ─────────────────────────── BASE PATHS ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")

# ─────────────────────────── MODEL CONFIG ─────────────────────────
POSE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-pose.pt")
FACE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-face.pt")
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# Detection confidence thresholds
POSE_CONFIDENCE = 0.45
FACE_CONFIDENCE = 0.40
PERSON_CONFIDENCE = 0.45
KEYPOINT_CONFIDENCE = 0.3     # Minimum confidence for a keypoint to be used

# ─────────────────────────── CAMERA CONFIG ────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ─────────────────────────── HEART ESTIMATION ─────────────────────
# Anatomical ratios (relative to torso dimensions)
# Heart sits ~30% down from shoulders, slightly left of midline
HEART_VERTICAL_RATIO = 0.30        # 30% down from shoulders toward hips
HEART_HORIZONTAL_OFFSET = -0.08    # 8% left of shoulder midline (negative = left)
# Fallback: when hips aren't detected, use shoulder span as reference
HEART_FALLBACK_DOWN_RATIO = 0.35   # % of shoulder width to go down
HEART_FALLBACK_LEFT_RATIO = -0.10  # % of shoulder width to shift left
# Heart zone radius (% of shoulder width)
HEART_ZONE_RADIUS_RATIO = 0.18

# ─────────────────────────── TRACKER CONFIG ───────────────────────
TRACKER_IOU_THRESHOLD = 0.3       # Minimum IoU to associate detections
TRACKER_MAX_AGE = 30              # Frames to keep lost tracks
TRACKER_MIN_HITS = 3              # Minimum detections before track is confirmed
TRACKER_MAX_TARGETS = 20          # Maximum simultaneous targets

# ─────────────────────────── FIRING SYSTEM ────────────────────────
FIRING_ENABLED = False             # Start with safety ON
LOCK_ON_FRAMES = 15               # Frames needed to acquire lock
LOCK_ON_STABILITY_PX = 40         # Max movement (px) during lock-on to count as stable
FIRE_COOLDOWN_FRAMES = 45         # Frames between shots (cooldown)
ENGAGEMENT_RANGE_MIN = 0.5        # Minimum distance (meters) for engagement
ENGAGEMENT_RANGE_MAX = 15.0       # Maximum distance (meters) for engagement
ENABLE_SOUND = True               # Enable firing/lock-on sounds

# ─────────────────────────── HUD CONFIG ───────────────────────────
HUD_OPACITY = 0.7                  # Overlay opacity (0.0 - 1.0)
HUD_FONT_SCALE = 0.5
HUD_CROSSHAIR_SIZE = 30           # Crosshair arm length in pixels
HUD_CROSSHAIR_THICKNESS = 2
HUD_SHOW_GRID = False              # Tactical grid overlay
HUD_GRID_SPACING = 80             # Grid spacing in pixels
HUD_SHOW_MINIMAP = True           # Show minimap
HUD_MINIMAP_SIZE = 150            # Minimap width/height in pixels
HUD_SHOW_TRAJECTORY = True        # Show target trajectory trails
HUD_TRAJECTORY_LENGTH = 20        # Number of past points to draw
HUD_PANEL_WIDTH = 280             # Right-side info panel width

# ─────────────────────────── RECORDING ────────────────────────────
RECORDING_ENABLED = False
RECORDING_CODEC = "XVID"
RECORDING_FPS = 20.0

# ─────────────────────────── YOLO POSE KEYPOINT INDICES ───────────
# Standard COCO pose keypoint indices for YOLOv8-pose
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# ─────────────────────────── DISTANCE ESTIMATION ──────────────────
# Average human face width in cm (used for distance estimation)
AVG_FACE_WIDTH_CM = 14.5
# Average shoulder width in cm
AVG_SHOULDER_WIDTH_CM = 40.0
# Approximate focal length (pixels) — calibrate for your camera
# Default is a reasonable estimate for 720p webcam
FOCAL_LENGTH_PX = 700.0
