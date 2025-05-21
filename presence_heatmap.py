import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import time

# Configuration
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
HEATMAP_DECAY = 0.98
HEATMAP_RADIUS = 15
BLUR_KERNEL = (25, 25)
OVERLAY_ALPHA = 0.4
COUNT_FONT_SCALE = 0.6
WINDOW_NAME = f"Persistent Heatmap ({DISPLAY_WIDTH}x{DISPLAY_HEIGHT}) (Press 'q' to quit)"

def load_model(model_path="yolo11n"):
    """Loads the YOLO model."""
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"FATAL: Error loading YOLO model: {e}")
        sys.exit(1)

def open_video_capture(video_path):
    """Opens the video file."""
    if not os.path.exists(video_path):
        print(f"FATAL: Video file not found at '{video_path}'")
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FATAL: Could not open video file '{video_path}'")
        sys.exit(1)
    print(f"Opened video: {video_path}")
    return cap

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
        # Fallback: try displaying original frame if resizing failed
        try:
            cv2.imshow(window_name, frame)
            return True
        except Exception as e2:
            print(f"ERROR displaying fallback frame: {e2}")
            return False # Indicate display failure

def main(video_path):
    model = load_model()
    cap = open_video_capture(video_path)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {original_width}x{original_height}")

    heatmap = np.zeros((original_height, original_width), dtype=np.float32)
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

            try:
                # Detect objects (adjust classes=[0] if only persons needed)
                results = model(frame, conf=0.2, verbose=False)
            except Exception as e:
                print(f"ERROR during YOLO detection on frame {frame_count}: {e}")
                continue

            people_count = 0
            try:
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            # Optional: Filter for specific classes (e.g., class_id == 0 for person)
                            # class_id = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                            # if class_id != 0: continue # Example: skip if not a person

                            if box.xyxy is not None and len(box.xyxy) > 0 and len(box.xyxy[0]) >= 4:
                                x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                                if 0 <= center_x < original_width and 0 <= center_y < original_height:
                                    cv2.circle(heatmap, (center_x, center_y), HEATMAP_RADIUS, 1, -1)
                                people_count += 1
            except Exception as e:
                print(f"ERROR processing detections on frame {frame_count}: {e}")

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

                cv2.putText(overlay, f"Count: {people_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, COUNT_FONT_SCALE,
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
        print("Usage: python presence_heatmap_refactored.py <path_to_video>")
        sys.exit(1)
    video_path_arg = sys.argv[1]
    main(video_path_arg)
