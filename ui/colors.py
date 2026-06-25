"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Color Constants — Military HUD Color Scheme                 ║
╚══════════════════════════════════════════════════════════════════════╝

All colors are in BGR format for OpenCV.
"""


# ─────────────────────────── PRIMARY COLORS ───────────────────────
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 40)
MID_GRAY = (100, 100, 100)
LIGHT_GRAY = (180, 180, 180)

# ─────────────────────────── HUD THEME (Military Green) ───────────
HUD_GREEN = (0, 255, 80)            # Primary HUD color — bright green
HUD_GREEN_DIM = (0, 160, 50)        # Dimmed green for secondary elements
HUD_GREEN_DARK = (0, 80, 30)        # Dark green for background fills
HUD_CYAN = (255, 255, 0)            # Cyan/teal for info elements (BGR)
HUD_AMBER = (0, 180, 255)           # Amber for warnings

# ─────────────────────────── DETECTION COLORS ─────────────────────
COLOR_FACE = (0, 255, 255)           # Yellow — face bounding box
COLOR_BODY = (255, 180, 0)           # Blue — body bounding box
COLOR_SHOULDER = (0, 255, 0)         # Green — shoulder keypoints
COLOR_HIP = (255, 165, 0)            # Orange — hip keypoints
COLOR_SKELETON = (180, 180, 0)       # Teal — skeleton lines

# ─────────────────────────── HEART / TARGET COLORS ────────────────
COLOR_HEART = (0, 0, 255)            # Red — heart marker
COLOR_HEART_ZONE = (0, 0, 200)       # Dark red — heart zone circle
COLOR_HEART_GLOW = (80, 80, 255)     # Light red glow around heart

# ─────────────────────────── ENGAGEMENT STATE COLORS ──────────────
STATE_SAFE = (0, 200, 0)             # Green — no threat / safety on
STATE_TRACKING = (0, 220, 255)       # Yellow — actively tracking
STATE_LOCKING = (0, 140, 255)        # Orange — lock-on in progress
STATE_LOCKED = (0, 0, 255)           # Red — target locked
STATE_FIRING = (0, 0, 255)           # Red flash — firing
STATE_COOLDOWN = (200, 100, 0)       # Blue — post-fire cooldown

# ─────────────────────────── CROSSHAIR COLORS ─────────────────────
CROSSHAIR_IDLE = HUD_GREEN           # Green when no target
CROSSHAIR_TRACKING = (0, 220, 255)   # Yellow when tracking
CROSSHAIR_LOCKED = (0, 0, 255)       # Red when locked
CROSSHAIR_FIRE = (0, 0, 255)         # Bright red on fire

# ─────────────────────────── MINIMAP COLORS ───────────────────────
MINIMAP_BG = (20, 20, 20)
MINIMAP_BORDER = HUD_GREEN_DIM
MINIMAP_TARGET = (0, 0, 255)         # Red dot for targets
MINIMAP_SELF = HUD_GREEN             # Green dot for camera position

# ─────────────────────────── LASER / FIRING EFFECTS ───────────────
LASER_COLOR = (0, 0, 255)            # Red laser line
MUZZLE_FLASH_COLOR = (255, 255, 255) # White flash
IMPACT_COLOR = (0, 140, 255)         # Orange impact marker
