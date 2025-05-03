import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import time

# --- Configuration ---
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WINDOW_NAME = f"Persistent Heatmap (Fixed: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}) (Press 'q' to quit)"
HEATMAP_DECAY = 0.98
HEATMAP_RADIUS = 15
BLUR_KERNEL = (25, 25)
OVERLAY_ALPHA = 0.4
# Reduced font scale for text drawn on original frame
COUNT_FONT_SCALE = 0.6 # Reduced from 0.6 (original was 0.6, keeping it same or slightly smaller)

print("--- Script Start ---")

# --- Load Model ---
try:
    print("Loading YOLO model...")
    model = YOLO("yolov8n")
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading YOLO model: {e}")
    sys.exit(1)

# --- Get Video Path from Command Line ---
if len(sys.argv) < 2:
    print("FATAL: No video path provided.")
    print("Usage: python one.py <path_to_video>")
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
# fps = cap.get(cv2.CAP_PROP_FPS) # FPS not needed for this script
print(f"Video properties: {original_frame_width}x{original_frame_height}")


# --- Heatmap Setup ---
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
            # Detect all classes by default, or specify classes=[0] for only persons
            results = model(frame, conf=0.2, verbose=False)
        except Exception as e:
            print(f"ERROR during YOLO detection on frame {frame_count}: {e}")
            continue

        # --- Process Detections for Heatmap ---
        people_count = 0 # Reset count for each frame
        try:
            for r in results:
                 if r.boxes is not None:
                    for box in r.boxes:
                        # Check if it's a person (class 0 in COCO) if you only want to count people
                        # class_id = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                        # if class_id == 0: # Uncomment this line and indent below to count only persons

                        if box.xyxy is not None and len(box.xyxy) > 0 and len(box.xyxy[0]) >= 4:
                           x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                           center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                           # Add heat at center
                           if 0 <= center_x < original_frame_width and 0 <= center_y < original_frame_height:
                               cv2.circle(heatmap, (center_x, center_y), HEATMAP_RADIUS, 1, -1)

                           people_count += 1 # Increment count for each detected object (or person if filtered)

        except Exception as e:
             print(f"ERROR processing detections on frame {frame_count}: {e}")


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

            # Display total people count (using reduced font scale)
            cv2.putText(overlay, f"People Count: {people_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, COUNT_FONT_SCALE, # Use reduced scale
                        (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"ERROR generating heatmap overlay on frame {frame_count}: {e}")
            overlay = frame

        # --- Resize for Display (Preserving Aspect Ratio) ---
        try:
            scale_w = DISPLAY_WIDTH / original_frame_width
            scale_h = DISPLAY_HEIGHT / original_frame_height
            scale = min(scale_w, scale_h)
            new_w = int(original_frame_width * scale)
            new_h = int(original_frame_height * scale)

            resized_overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            x_offset = (DISPLAY_WIDTH - new_w) // 2
            y_offset = (DISPLAY_HEIGHT - new_h) // 2
            display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_overlay

            cv2.imshow(WINDOW_NAME, display_frame)

        except Exception as e:
            print(f"ERROR resizing or displaying frame {frame_count}: {e}")
            try:
                cv2.imshow(WINDOW_NAME, overlay) # Fallback
            except Exception as e2:
                 print(f"ERROR displaying fallback frame {frame_count}: {e2}")
                 break

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
