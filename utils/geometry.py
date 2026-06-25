"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Geometry Utilities — Distance, Angle, IOU Calculations      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import math
import numpy as np
import config


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box_a: (x1, y1, x2, y2) format
        box_b: (x1, y1, x2, y2) format
    
    Returns:
        float: IoU value between 0.0 and 1.0
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    union = area_a + area_b - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def estimate_distance_from_face(face_width_px):
    """
    Estimate real-world distance (in meters) from face bounding box width.
    Uses the pinhole camera model: distance = (real_width * focal_length) / pixel_width
    
    Args:
        face_width_px: Width of the face bounding box in pixels
    
    Returns:
        float: Estimated distance in meters, or -1 if invalid
    """
    if face_width_px <= 0:
        return -1.0
    
    distance_cm = (config.AVG_FACE_WIDTH_CM * config.FOCAL_LENGTH_PX) / face_width_px
    return distance_cm / 100.0  # Convert to meters


def estimate_distance_from_shoulders(shoulder_width_px):
    """
    Estimate real-world distance from shoulder width.
    
    Args:
        shoulder_width_px: Distance between left and right shoulder in pixels
    
    Returns:
        float: Estimated distance in meters, or -1 if invalid
    """
    if shoulder_width_px <= 0:
        return -1.0
    
    distance_cm = (config.AVG_SHOULDER_WIDTH_CM * config.FOCAL_LENGTH_PX) / shoulder_width_px
    return distance_cm / 100.0


def calculate_bearing(target_x, frame_width):
    """
    Calculate bearing angle (degrees) of target relative to camera center.
    0° = center, negative = left, positive = right
    
    Args:
        target_x: X coordinate of target in pixels
        frame_width: Width of the frame in pixels
    
    Returns:
        float: Bearing angle in degrees (-90 to +90)
    """
    center_x = frame_width / 2
    offset = target_x - center_x
    # Approximate FOV mapping (assumes ~60° horizontal FOV)
    angle = (offset / (frame_width / 2)) * 30.0
    return round(angle, 1)


def calculate_elevation(target_y, frame_height):
    """
    Calculate elevation angle of target relative to camera center.
    0° = center, negative = above, positive = below
    
    Args:
        target_y: Y coordinate of target in pixels
        frame_height: Height of the frame in pixels
    
    Returns:
        float: Elevation angle in degrees
    """
    center_y = frame_height / 2
    offset = target_y - center_y
    angle = (offset / (frame_height / 2)) * 22.5  # ~45° vertical FOV
    return round(angle, 1)


def point_distance(p1, p2):
    """
    Euclidean distance between two 2D points.
    
    Args:
        p1: (x, y) tuple
        p2: (x, y) tuple
    
    Returns:
        float: Euclidean distance
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def midpoint(p1, p2):
    """
    Calculate midpoint between two 2D points.
    
    Args:
        p1: (x, y) tuple
        p2: (x, y) tuple
    
    Returns:
        tuple: (mx, my) midpoint
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def is_in_engagement_zone(distance_m):
    """
    Check if target is within the engagement range.
    
    Args:
        distance_m: Target distance in meters
    
    Returns:
        str: 'IN_RANGE', 'TOO_CLOSE', 'TOO_FAR', or 'UNKNOWN'
    """
    if distance_m < 0:
        return "UNKNOWN"
    if distance_m < config.ENGAGEMENT_RANGE_MIN:
        return "TOO_CLOSE"
    if distance_m > config.ENGAGEMENT_RANGE_MAX:
        return "TOO_FAR"
    return "IN_RANGE"


def smooth_point(history, alpha=0.4):
    """
    Apply exponential moving average smoothing to a series of points.
    Reduces jitter in keypoint positions.
    
    Args:
        history: List of (x, y) points (most recent last)
        alpha: Smoothing factor (higher = more responsive, lower = smoother)
    
    Returns:
        tuple: Smoothed (x, y) point
    """
    if not history:
        return (0, 0)
    if len(history) == 1:
        return history[0]
    
    sx, sy = history[0]
    for px, py in history[1:]:
        sx = alpha * px + (1 - alpha) * sx
        sy = alpha * py + (1 - alpha) * sy
    
    return (int(sx), int(sy))


def box_center(box):
    """
    Get center point of a bounding box.
    
    Args:
        box: (x1, y1, x2, y2) format
    
    Returns:
        tuple: (cx, cy) center point
    """
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))


def box_area(box):
    """
    Get area of a bounding box.
    
    Args:
        box: (x1, y1, x2, y2) format
    
    Returns:
        float: Area in square pixels
    """
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
