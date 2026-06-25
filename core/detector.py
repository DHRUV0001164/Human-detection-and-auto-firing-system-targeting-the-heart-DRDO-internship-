"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Detection Engine — YOLOv8 Pose + Face + Person Detection    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from ultralytics import YOLO
import config


class DetectionResult:
    """
    Container for a single person's detection results from all models.
    """
    def __init__(self):
        self.person_box = None          # (x1, y1, x2, y2) from person model
        self.person_conf = 0.0
        self.face_box = None            # (x1, y1, x2, y2) from face model
        self.face_conf = 0.0
        self.keypoints = None           # (17, 2) array of keypoint positions
        self.keypoint_confs = None      # (17,) array of keypoint confidences
        self.pose_box = None            # (x1, y1, x2, y2) from pose model
        self.pose_conf = 0.0

    @property
    def left_shoulder(self):
        """Get left shoulder position if confident enough."""
        if self.keypoints is not None and self.keypoint_confs is not None:
            if self.keypoint_confs[config.KP_LEFT_SHOULDER] >= config.KEYPOINT_CONFIDENCE:
                return tuple(self.keypoints[config.KP_LEFT_SHOULDER])
        return None

    @property
    def right_shoulder(self):
        """Get right shoulder position if confident enough."""
        if self.keypoints is not None and self.keypoint_confs is not None:
            if self.keypoint_confs[config.KP_RIGHT_SHOULDER] >= config.KEYPOINT_CONFIDENCE:
                return tuple(self.keypoints[config.KP_RIGHT_SHOULDER])
        return None

    @property
    def left_hip(self):
        """Get left hip position if confident enough."""
        if self.keypoints is not None and self.keypoint_confs is not None:
            if self.keypoint_confs[config.KP_LEFT_HIP] >= config.KEYPOINT_CONFIDENCE:
                return tuple(self.keypoints[config.KP_LEFT_HIP])
        return None

    @property
    def right_hip(self):
        """Get right hip position if confident enough."""
        if self.keypoints is not None and self.keypoint_confs is not None:
            if self.keypoint_confs[config.KP_RIGHT_HIP] >= config.KEYPOINT_CONFIDENCE:
                return tuple(self.keypoints[config.KP_RIGHT_HIP])
        return None

    @property
    def nose(self):
        """Get nose position if confident enough."""
        if self.keypoints is not None and self.keypoint_confs is not None:
            if self.keypoint_confs[config.KP_NOSE] >= config.KEYPOINT_CONFIDENCE:
                return tuple(self.keypoints[config.KP_NOSE])
        return None

    @property
    def best_box(self):
        """Get the best available bounding box (prefer pose, then person)."""
        if self.pose_box is not None:
            return self.pose_box
        return self.person_box

    @property
    def has_shoulders(self):
        """Check if both shoulders are detected."""
        return self.left_shoulder is not None and self.right_shoulder is not None

    @property
    def has_hips(self):
        """Check if both hips are detected."""
        return self.left_hip is not None and self.right_hip is not None

    @property
    def has_face(self):
        """Check if face is detected."""
        return self.face_box is not None


class DetectionEngine:
    """
    Multi-model detection engine using YOLOv8 for pose, face, and person detection.
    Fuses results from all three models for comprehensive human detection.
    """

    def __init__(self):
        print("[DETECTOR] Loading models...")

        # Load pose model
        print(f"  → Loading pose model: {config.POSE_MODEL_PATH}")
        self.pose_model = YOLO(config.POSE_MODEL_PATH)

        # Load face model
        print(f"  → Loading face model: {config.FACE_MODEL_PATH}")
        self.face_model = YOLO(config.FACE_MODEL_PATH)

        # Load person model (for robust person detection)
        print(f"  → Loading person model: {config.PERSON_MODEL_PATH}")
        self.person_model = YOLO(config.PERSON_MODEL_PATH)

        print("[DETECTOR] All models loaded successfully ✓")

        # Performance tracking
        self.last_inference_ms = 0

    def detect(self, frame):
        """
        Run all detection models on a frame and fuse results.

        Args:
            frame: numpy array — BGR image from camera

        Returns:
            list[DetectionResult]: List of detection results, one per person
        """
        import time
        t0 = time.time()

        # Run pose model (gives keypoints + person boxes)
        pose_results = self.pose_model(
            frame,
            conf=config.POSE_CONFIDENCE,
            verbose=False
        )

        # Run face model
        face_results = self.face_model(
            frame,
            conf=config.FACE_CONFIDENCE,
            verbose=False
        )

        # Collect pose detections
        detections = []

        for r in pose_results:
            if r.keypoints is None:
                continue

            keypoints_xy = r.keypoints.xy.cpu().numpy()    # (N, 17, 2)
            keypoints_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None  # (N, 17)
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []

            for i, person_kps in enumerate(keypoints_xy):
                det = DetectionResult()
                det.keypoints = person_kps
                det.keypoint_confs = keypoints_conf[i] if keypoints_conf is not None else np.ones(17) * 0.5

                if i < len(boxes):
                    det.pose_box = tuple(map(int, boxes[i]))
                    det.pose_conf = float(confs[i]) if i < len(confs) else 0.5

                detections.append(det)

        # Collect face boxes
        face_boxes = []
        face_confs_list = []
        for fr in face_results:
            if fr.boxes is not None:
                for j, fbox in enumerate(fr.boxes.xyxy.cpu().numpy()):
                    face_boxes.append(tuple(map(int, fbox)))
                    fc = fr.boxes.conf.cpu().numpy()
                    face_confs_list.append(float(fc[j]) if j < len(fc) else 0.5)

        # Associate faces with pose detections using proximity
        used_faces = set()
        for det in detections:
            if det.nose is not None:
                nose_x, nose_y = det.nose
                best_face_idx = -1
                best_dist = float('inf')

                for fi, fbox in enumerate(face_boxes):
                    if fi in used_faces:
                        continue
                    # Check if nose is inside or near the face box
                    fx1, fy1, fx2, fy2 = fbox
                    face_cx = (fx1 + fx2) / 2
                    face_cy = (fy1 + fy2) / 2
                    dist = ((nose_x - face_cx) ** 2 + (nose_y - face_cy) ** 2) ** 0.5

                    # Face box should be close to nose
                    face_size = max(fx2 - fx1, fy2 - fy1)
                    if dist < face_size * 1.5 and dist < best_dist:
                        best_dist = dist
                        best_face_idx = fi

                if best_face_idx >= 0:
                    det.face_box = face_boxes[best_face_idx]
                    det.face_conf = face_confs_list[best_face_idx]
                    used_faces.add(best_face_idx)

            elif det.pose_box is not None:
                # Fallback: associate face with pose box by overlap
                best_face_idx = -1
                best_overlap = 0

                for fi, fbox in enumerate(face_boxes):
                    if fi in used_faces:
                        continue
                    from utils.geometry import calculate_iou
                    iou = calculate_iou(det.pose_box, fbox)
                    # Face should be in the upper portion of the body box
                    fcy = (fbox[1] + fbox[3]) / 2
                    body_top = det.pose_box[1]
                    body_h = det.pose_box[3] - det.pose_box[1]
                    if fcy < body_top + body_h * 0.4 and iou > 0 and iou > best_overlap:
                        best_overlap = iou
                        best_face_idx = fi

                if best_face_idx >= 0:
                    det.face_box = face_boxes[best_face_idx]
                    det.face_conf = face_confs_list[best_face_idx]
                    used_faces.add(best_face_idx)

        # Handle unmatched faces (faces without pose detections)
        for fi, fbox in enumerate(face_boxes):
            if fi not in used_faces:
                det = DetectionResult()
                det.face_box = fbox
                det.face_conf = face_confs_list[fi]
                # Create a rough person box from face (head-to-body ratio ~1:7)
                fx1, fy1, fx2, fy2 = fbox
                face_h = fy2 - fy1
                face_w = fx2 - fx1
                body_x1 = int(fx1 - face_w * 0.5)
                body_y1 = fy1
                body_x2 = int(fx2 + face_w * 0.5)
                body_y2 = int(fy2 + face_h * 5)
                det.person_box = (body_x1, body_y1, body_x2, body_y2)
                det.person_conf = face_confs_list[fi] * 0.6  # Lower confidence for estimated box
                detections.append(det)

        self.last_inference_ms = (time.time() - t0) * 1000

        return detections
