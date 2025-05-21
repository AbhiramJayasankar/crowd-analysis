import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import time

# Attempt to import SORT
try:
    from sort.tracker import SortTracker
except ImportError:
    print("FATAL: Error: Failed to import SortTracker.")
    print("Make sure the SORT library (e.g., filterpy) is installed correctly.")
    sys.exit(1)

# Configuration
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
HEATMAP_DECAY = 0.95
HEATMAP_INTENSITY_SCALE = 0.05 # How much speed contributes to intensity
HEATMAP_RADIUS = 25
BLUR_KERNEL = (35, 35)
OVERLAY_ALPHA = 0.4
AVG_SPEED_FONT_SCALE = 0.6
WINDOW_NAME = f"Speed Heatmap ({DISPLAY_WIDTH}x{DISPLAY_HEIGHT}) (Press 'q' to quit)"
YOLO_CONFIDENCE = 0.3
YOLO_CLASSES = [0] # Track only persons
SORT_MAX_AGE = 20
SORT_MIN_HITS = 3
SORT_IOU_THRESH = 0.3

def load_model(model_path="yolo11n"):
    """Loads the YOLO model."""
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"FATAL: Error loading YOLO model: {e}")
        sys.exit(1)

def initialize_tracker():
    """Initializes the SORT tracker."""
    try:
        tracker = SortTracker(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESH)
        print("SortTracker initialized successfully.")
        return tracker
    except Exception as e:
        print(f"FATAL: Error initializing SortTracker: {e}")
        sys.exit(1)

def open_video_capture(video_path):
    """Opens the video file and gets properties."""
    if not os.path.exists(video_path):
        print(f"FATAL: Video file not found at '{video_path}'")
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FATAL: Could not open video file '{video_path}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not read FPS, defaulting to 25.")
        fps = 25.0 # Default FPS if reading fails
    print(f"Opened video: {video_path} ({width}x{height} @ {fps:.2f} FPS)")
    return cap, width, height, fps

def display_frame_resized(window_name, frame, target_width, target_height):
    """Resizes frame maintaining aspect ratio and displays it centered."""
    try:
        h, w = frame.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        display_canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        display_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        cv2.imshow(window_name, display_canvas)
        return True
    except Exception as e:
        print(f"ERROR resizing/displaying frame: {e}")
        try:
            cv2.imshow(window_name, frame) # Fallback
            return True
        except Exception as e2:
            print(f"ERROR displaying fallback frame: {e2}")
            return False

def main(video_path):
    model = load_model()
    tracker = initialize_tracker()
    cap, original_width, original_height, fps = open_video_capture(video_path)

    heatmap = np.zeros((original_height, original_width), dtype=np.float32)
    prev_positions = {} # Store previous center {obj_id: (x, y)}
    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame_count += 1
            heatmap *= HEATMAP_DECAY # Apply decay

            # YOLO Detection
            try:
                results = model(frame, classes=YOLO_CLASSES, conf=YOLO_CONFIDENCE, verbose=False)
            except Exception as e:
                print(f"ERROR during YOLO detection on frame {frame_count}: {e}")
                continue

            # Prepare Detections for SORT (Using Original 6-element format)
            detections_for_sort_orig = []
            try:
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.xyxy is not None and len(box.xyxy) > 0 and len(box.xyxy[0]) >= 4:
                            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                            # Original format: [x1, y1, x2, y2, 1, 1]
                            detections_for_sort_orig.append([x1, y1, x2, y2, 1, 1])
            except Exception as e:
                print(f"ERROR extracting boxes on frame {frame_count}: {e}")

            # Convert to numpy array, handle empty case
            if detections_for_sort_orig:
                detections_np_orig = np.array(detections_for_sort_orig)
            else:
                 # Original script used float32 here, maintaining consistency
                detections_np_orig = np.empty((0, 6), dtype=np.float32)


            # SORT Tracking (Using Original Call Structure)
            tracked_objects = []
            try:
                 # Original call included the second argument '0'
                tracked_output = tracker.update(detections_np_orig, 0)
                if tracked_output is not None and len(tracked_output) > 0:
                     # Output format expected: x1, y1, x2, y2, obj_id
                    tracked_objects = tracked_output[:, :5]
            except IndexError as ie:
                 # Keep specific index error handling from original script
                 print(f"INDEX ERROR during SORT tracker update on frame {frame_count}: {ie}")
                 print("Input detections shape:", detections_np_orig.shape)
            except Exception as e:
                # Catch potential errors during update (e.g., empty input)
                 print(f"ERROR during SORT tracker update on frame {frame_count}: {e}")


            # Process tracked objects for speed and heatmap
            total_speed = 0.0
            tracked_count = 0
            current_positions = {}
            if len(tracked_objects) > 0:
                for obj in tracked_objects:
                    try:
                        x1_f, y1_f, x2_f, y2_f, obj_id_f = obj
                        obj_id = int(obj_id_f)
                        center_x = (x1_f + x2_f) / 2.0
                        center_y = (y1_f + y2_f) / 2.0
                        current_positions[obj_id] = (center_x, center_y)

                        if obj_id in prev_positions:
                            prev_x, prev_y = prev_positions[obj_id]
                            distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                            speed_px_per_sec = distance * fps # Speed in pixels/second

                            total_speed += speed_px_per_sec
                            tracked_count += 1

                            # Add speed intensity to heatmap
                            intensity = max(0.0, speed_px_per_sec * HEATMAP_INTENSITY_SCALE)
                            center_x_int, center_y_int = int(center_x), int(center_y)
                            if 0 <= center_x_int < original_width and 0 <= center_y_int < original_height:
                                cv2.circle(heatmap, (center_x_int, center_y_int), HEATMAP_RADIUS, intensity, -1)

                    except Exception as e:
                        print(f"ERROR processing tracked object {obj_id if 'obj_id' in locals() else 'unknown'} on frame {frame_count}: {e}")

            prev_positions = current_positions # Update positions for next frame

            # Generate and overlay heatmap
            overlay = frame.copy()
            try:
                max_val = np.max(heatmap)
                if max_val > 0:
                    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    heatmap_norm = np.zeros_like(heatmap, dtype=np.uint8)

                heatmap_blurred = cv2.GaussianBlur(heatmap_norm, BLUR_KERNEL, 0)
                heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 1.0 - OVERLAY_ALPHA, heatmap_colored, OVERLAY_ALPHA, 0)

                avg_speed = total_speed / tracked_count if tracked_count > 0 else 0
                cv2.putText(overlay, f"Avg Speed: {avg_speed:.1f} px/s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, AVG_SPEED_FONT_SCALE,
                            (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"ERROR generating heatmap overlay on frame {frame_count}: {e}")
                # Continue with original frame if overlay fails

            if not display_frame_resized(WINDOW_NAME, overlay, DISPLAY_WIDTH, DISPLAY_HEIGHT):
                break # Exit if display fails critically

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit signal received.")
                break

    except Exception as e:
        print(f"FATAL: An unexpected error occurred in the main loop: {e}")
    finally:
        end_time = time.time()
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

        total_time = end_time - start_time
        if frame_count > 0 and total_time > 0:
            avg_fps = frame_count / total_time
            print(f"Processed {frame_count} frames in {total_time:.2f}s (Avg: {avg_fps:.2f} FPS)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python speed_heatmap_refactored.py <path_to_video>")
        sys.exit(1)
    video_path_arg = sys.argv[1]
    main(video_path_arg)
