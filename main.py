"""
╔══════════════════════════════════════════════════════════════════════╗
║         DRDO - HUMAN DETECTION & AUTO FIRING SYSTEM                ║
║         Main Entry Point & Control Loop                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import os
import sys
import time
import argparse

# Add parent directory to path to ensure correct package imports when running
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.detector import DetectionEngine
from core.heart_estimator import HeartEstimator
from core.tracker import MultiTargetTracker
from core.firing_system import FiringSystem
from ui.hud import HUD
from utils.logger import EngagementLogger
from utils.geometry import estimate_distance_from_face, estimate_distance_from_shoulders


def parse_args():
    parser = argparse.ArgumentParser(description="DRDO Human Detection & Auto Firing System (Simulation)")
    parser.add_argument("--source", type=str, default=str(config.CAMERA_INDEX),
                        help="Camera index or path to video file")
    parser.add_argument("--record", action="store_true", help="Enable video recording of session")
    parser.add_argument("--no-sound", action="store_true", help="Disable sound effects")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH, help="Camera resolution width")
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT, help="Camera resolution height")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override config with arguments if needed
    if args.no_sound:
        config.ENABLE_SOUND = False
    
    # Initialize Logger
    logger = EngagementLogger()

    # Initialize Detection Engine
    try:
        detector = DetectionEngine()
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO models: {e}")
        print("Please ensure yolov8n-pose.pt, yolov8n-face.pt, and yolov8n.pt are in the project folder.")
        logger.close()
        return

    # Initialize core modules
    heart_estimator = HeartEstimator()
    tracker = MultiTargetTracker()
    firing_system = FiringSystem(logger=logger)
    hud = HUD()

    # Print start message
    print("\n" + "=" * 50)
    print(" SYSTEM ARMED & READY (SIMULATION ONLY)")
    print(" CONTROLS:")
    print("   Q       : Quit application")
    print("   F       : Toggle AUTO-FIRE mode (default: OFF)")
    print("   SPACE   : MANUAL FIRE (requires safety OFF)")
    print("   S       : Toggle SAFETY LOCK (default: LOCKED)")
    print("   G       : Toggle TACTICAL GRID")
    print("   M       : Toggle MINIMAP")
    print("   T       : Toggle TRAJECTORY TRAILS")
    print("   P       : Capture manual SCREENSHOT")
    print("=" * 50 + "\n")

    # Start Video Capture
    source = args.source
    # Convert to int if it's a digit (webcam index)
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        logger.close()
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Get actual frame width/height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.CAMERA_FPS
    print(f"[SYSTEM] Video capture started: {frame_width}x{frame_height} @ {fps:.1f} FPS")

    # Initialize Video Writer if recording is enabled
    out_writer = None
    if args.record or config.RECORDING_ENABLED:
        os.makedirs(config.RECORDINGS_DIR, exist_ok=True)
        rec_filename = os.path.join(
            config.RECORDINGS_DIR, 
            f"record_{logger.session_id}.avi"
        )
        fourcc = cv2.VideoWriter_fourcc(*config.RECORDING_CODEC)
        out_writer = cv2.VideoWriter(
            rec_filename, 
            fourcc, 
            config.RECORDING_FPS, 
            (frame_width, frame_height)
        )
        print(f"[SYSTEM] Session recording enabled: {rec_filename}")

    try:
        while True:
            t_loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[SYSTEM] Video source ended or failed to read frame.")
                break

            # 1. Run Detection
            detections = detector.detect(frame)
            inference_ms = detector.last_inference_ms

            # 2. Update Tracker
            tracks = tracker.update(detections)

            # Update total targets count in logger
            if tracker.total_targets_created > logger.targets_detected_total:
                logger.targets_detected_total = tracker.total_targets_created

            # 3. Estimate Heart Position & Distance for each track
            heart_estimator_results = {}
            for track in tracks:
                if track.detection:
                    # Estimate heart
                    heart_est = heart_estimator.estimate(track.detection, target_id=track.label)
                    if heart_est.is_valid:
                        track.heart_pos = heart_est.position
                        track.heart_confidence = heart_est.confidence
                        heart_estimator_results[track.label] = heart_est

                    # Estimate distance
                    dist_m = -1.0
                    # Method A: Use face box width if available (more precise)
                    if track.detection.has_face:
                        fx1, _, fx2, _ = track.detection.face_box
                        dist_m = estimate_distance_from_face(fx2 - fx1)
                    # Method B: Use shoulder span fallback
                    elif track.detection.has_shoulders:
                        ls = track.detection.left_shoulder
                        rs = track.detection.right_shoulder
                        shoulder_width_px = ((ls[0] - rs[0])**2 + (ls[1] - rs[1])**2)**0.5
                        dist_m = estimate_distance_from_shoulders(shoulder_width_px)
                    
                    track.distance_m = dist_m

            # 4. Get Priority Target
            priority_target = tracker.get_priority_target()

            # 5. Update Firing System
            engagements = firing_system.update(tracks, priority_target)

            # Check if any target state was just set to FIRING to take a screenshot
            for tid, eng in engagements.items():
                if eng.state == "FIRING" and eng.fire_frame_counter == firing_system.fire_animation_frames:
                    # Capture firing screenshot (un-annotated clean frame or HUD frame?)
                    # Let's save a copy of the annotated frame or the original frame
                    logger.capture_screenshot(frame.copy(), event_type="FIRE", target_id=tid)

            # 6. Render HUD Overlay
            hud.tick()
            display_frame = frame.copy()
            hud.render(
                display_frame, 
                tracks, 
                engagements, 
                firing_system, 
                priority_target, 
                heart_estimator_results,
                logger, 
                inference_ms
            )

            # 7. Write to Recording
            if out_writer is not None:
                out_writer.write(display_frame)

            # 8. Show Frame
            cv2.imshow("DRDO Human Detection & Auto Firing System", display_frame)

            # 9. Handle Keyboard Inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("[SYSTEM] Shutdown initiated by user.")
                break
            elif key == ord('f') or key == ord('F'):
                firing_system.toggle_auto_fire()
            elif key == ord('s') or key == ord('S'):
                firing_system.toggle_safety()
            elif key == ord('g') or key == ord('G'):
                config.HUD_SHOW_GRID = not config.HUD_SHOW_GRID
                print(f"[HUD] Grid overlay {'ENABLED' if config.HUD_SHOW_GRID else 'DISABLED'}")
            elif key == ord('m') or key == ord('M'):
                config.HUD_SHOW_MINIMAP = not config.HUD_SHOW_MINIMAP
                print(f"[HUD] Minimap {'ENABLED' if config.HUD_SHOW_MINIMAP else 'DISABLED'}")
            elif key == ord('t') or key == ord('T'):
                config.HUD_SHOW_TRAJECTORY = not config.HUD_SHOW_TRAJECTORY
                print(f"[HUD] Trajectory trails {'ENABLED' if config.HUD_SHOW_TRAJECTORY else 'DISABLED'}")
            elif key == ord('p') or key == ord('P'):
                logger.capture_screenshot(display_frame.copy(), event_type="MANUAL_CAPTURE")
                print("[SYSTEM] Manual screenshot captured.")
            elif key == ord(' '):  # SPACEBAR
                if priority_target:
                    firing_system.manual_fire(priority_target)
                else:
                    print("[FIRING] No active target locked for manual firing.")

            # Increment logger frame count
            logger.increment_frame()

    except KeyboardInterrupt:
        print("[SYSTEM] Shutdown via keyboard interrupt.")

    finally:
        # Clean up resources
        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()
        logger.close()
        print("[SYSTEM] All resources released. Goodbye.")


if __name__ == "__main__":
    main()
