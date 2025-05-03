import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import time

# --- Configuration ---
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WINDOW_NAME = f"Crowd Speed Heatmap (Fixed: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}) (Press 'q' to quit)"
HEATMAP_DECAY = 0.95
HEATMAP_INTENSITY_SCALE = 0.05
HEATMAP_RADIUS = 25
BLUR_KERNEL = (35, 35)
OVERLAY_ALPHA = 0.4
# Reduced font scale for text drawn on original frame, as it will be resized later
AVG_SPEED_FONT_SCALE = 0.6 # Reduced from 0.8

print("--- Script Start ---")

# --- Load Model ---
try:
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading YOLO model: {e}")
    sys.exit(1)

# --- Initialize SORT tracker ---
try:
    print("Importing SortTracker...")
    from sort.tracker import SortTracker
    print("Initializing SortTracker...")
    tracker = SortTracker(max_age=20, min_hits=3, iou_threshold=0.3)
    print("SortTracker initialized successfully.")
except ImportError:
    print("FATAL: Error: Failed to import SortTracker.")
    print("Make sure the SORT library is installed correctly.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Error initializing SortTracker: {e}")
    sys.exit(1)

# --- Get Video Path from Command Line ---
if len(sys.argv) < 2:
    print("FATAL: No video path provided.")
    print("Usage: python speed2.py <path_to_video>")
    sys.exit(1)
video_path = sys.argv[1]
print(f"Input video path: {video_path}")
if not os.path.exists(video_path):
    print(f"FATAL: Video file not found at '{video_path}'")
    sys.exit(1)

# --- Video Capture Setup ---
print("Opening video capture...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"FATAL: Could not open video file '{video_path}'")
    sys.exit(1)
print("Video capture opened successfully.")

original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video properties: {original_frame_width}x{original_frame_height} @ {fps:.2f} FPS")
if fps <= 0:
    print("Warning: Could not read FPS, defaulting to 25.")
    fps = 25

# --- Heatmap and Position Setup ---
prev_positions = {}
heatmap = np.zeros((original_frame_height, original_frame_width), dtype=np.float32)
print("Heatmap initialized.")

print("Starting main processing loop...")
frame_count = 0
start_time = time.time()

# --- Main Processing Loop ---
try:
    while cap.isOpened():
        frame_count += 1
        loop_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # Apply decay
        heatmap *= HEATMAP_DECAY

        # --- YOLO Detection ---
        try:
            results = model(frame, classes=[0], conf=0.3, verbose=False)
        except Exception as e:
            print(f"ERROR during YOLO detection on frame {frame_count}: {e}")
            continue

        # --- Prepare Detections for SORT ---
        detections_for_sort_orig = []
        try:
            for r in results:
                 if r.boxes is not None:
                    for box in r.boxes:
                        if box.xyxy is not None and len(box.xyxy) > 0 and len(box.xyxy[0]) >= 4:
                           x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                           detections_for_sort_orig.append([x1, y1, x2, y2, 1, 1])
        except Exception as e:
             print(f"ERROR extracting boxes on frame {frame_count}: {e}")

        if detections_for_sort_orig:
            detections_np_orig = np.array(detections_for_sort_orig)
        else:
            detections_np_orig = np.empty((0, 6), dtype=np.float32)

        # --- SORT Tracking ---
        try:
            tracked_objects = tracker.update(detections_np_orig, 0)
        except IndexError as ie:
             print(f"INDEX ERROR during SORT tracker update on frame {frame_count}: {ie}")
             print("Input detections shape:", detections_np_orig.shape)
             tracked_objects = []
        except Exception as e:
            print(f"ERROR during SORT tracker update on frame {frame_count}: {e}")
            tracked_objects = []

        # --- Process Tracked Objects ---
        total_speed = 0.0
        tracked_count = 0
        if tracked_objects is not None and len(tracked_objects) > 0:
            for obj in tracked_objects:
                try:
                    if len(obj) >= 5:
                        x1_f, y1_f, x2_f, y2_f, obj_id_f = obj[:5]
                        obj_id = int(obj_id_f)
                        center_x = (x1_f + x2_f) / 2.0
                        center_y = (y1_f + y2_f) / 2.0

                        speed_px_per_sec = 0.0
                        if obj_id in prev_positions:
                            prev_x, prev_y = prev_positions[obj_id]
                            distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                            speed_px_per_sec = distance * fps

                            total_speed += speed_px_per_sec
                            tracked_count += 1

                            intensity = max(0.0, speed_px_per_sec * HEATMAP_INTENSITY_SCALE)
                            center_x_int, center_y_int = int(center_x), int(center_y)
                            if 0 <= center_x_int < original_frame_width and 0 <= center_y_int < original_frame_height:
                                cv2.circle(heatmap, (center_x_int, center_y_int), HEATMAP_RADIUS, intensity, -1)

                        prev_positions[obj_id] = (center_x, center_y)
                except Exception as e:
                    print(f"ERROR processing tracked object {obj_id if 'obj_id' in locals() else 'unknown'} on frame {frame_count}: {e}")

        # --- Generate Heatmap Overlay ---
        try:
            max_heatmap_val = np.max(heatmap)
            if max_heatmap_val > 0:
                 heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                 heatmap_norm = np.zeros_like(heatmap, dtype=np.uint8)

            heatmap_blurred = cv2.GaussianBlur(heatmap_norm, BLUR_KERNEL, 0)
            heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)
            overlay_base = frame.copy()
            overlay = cv2.addWeighted(overlay_base, 1.0 - OVERLAY_ALPHA, heatmap_colored, OVERLAY_ALPHA, 0)

            # Calculate and display average speed (using reduced font scale)
            avg_speed = total_speed / tracked_count if tracked_count > 0 else 0
            cv2.putText(overlay, f"Avg Crowd Speed: {avg_speed:.1f} px/s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, AVG_SPEED_FONT_SCALE, # Use reduced scale
                        (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"ERROR generating heatmap overlay on frame {frame_count}: {e}")
            overlay = frame

        # --- Resize for Display (Preserving Aspect Ratio) ---
        try:
            # Calculate scaling factor to fit within display dimensions
            scale_w = DISPLAY_WIDTH / original_frame_width
            scale_h = DISPLAY_HEIGHT / original_frame_height
            scale = min(scale_w, scale_h) # Use the smaller scale factor

            # Calculate new dimensions
            new_w = int(original_frame_width * scale)
            new_h = int(original_frame_height * scale)

            # Resize the overlay frame
            resized_overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create a black canvas of the target display size
            display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

            # Calculate top-left position to paste the resized frame (centering it)
            x_offset = (DISPLAY_WIDTH - new_w) // 2
            y_offset = (DISPLAY_HEIGHT - new_h) // 2

            # Paste the resized frame onto the canvas
            display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_overlay

            # Display the final frame with letter/pillar boxing
            cv2.imshow(WINDOW_NAME, display_frame)

        except Exception as e:
            print(f"ERROR resizing or displaying frame {frame_count}: {e}")
            # If resize/display fails, show the original overlay directly (might be wrong size)
            try:
                cv2.imshow(WINDOW_NAME, overlay)
            except Exception as e2:
                 print(f"ERROR displaying fallback frame {frame_count}: {e2}")
                 break # Exit loop if display fails completely

        # --- WaitKey ---
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
           print("Quit signal received.")
           break

        loop_end_time = time.time()

except Exception as e:
    print(f"FATAL: An unexpected error occurred in the main loop: {e}")
finally:
    # --- Cleanup ---
    print("Cleaning up...")
    end_time = time.time()
    cap.release()
    cv2.destroyAllWindows()
    print("Video capture released and windows destroyed.")

    total_time = end_time - start_time
    print("-" * 30)
    print("Processing finished (display only).")
    if frame_count > 0 and total_time > 0:
        avg_fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds (Avg: {avg_fps:.2f} FPS)")
    print("-" * 30)
