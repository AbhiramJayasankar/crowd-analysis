import cv2
import numpy as np
from ultralytics import YOLO
from sort.tracker import SortTracker  # SORT tracker for object tracking

# Load YOLO model
model = YOLO("yolov8n.pt")  # Change to 'yolov8m.pt' for better accuracy

# Initialize SORT tracker
tracker = SortTracker()

# Video paths
video_path = "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\video\\big_video.avi"
output_path = (
    "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\video\\crowd_speed_heatmap.avi"
)

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Dictionary to store last known positions
prev_positions = {}
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
heatmap_decay = 0.95  # Adjusted decay for smoother persistence
heatmap_intensity = 3.0  # Higher intensity for better visibility

print("Estimating crowd speed with persistent heatmap...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply decay to smooth out heatmap over time
    heatmap *= heatmap_decay

    # Run YOLO detection
    results = model(frame, conf=0.3)

    detections = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
            detections.append([x1, y1, x2, y2, 1, 1])  # Confidence needed for SORT

    if not detections:
        detections.append(np.empty((0, 5)))

    # Track objects
    tracked_objects = tracker.update(np.array(detections), 0)

    total_speed = 0
    count = 0

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, _, _ = list(map(int, obj))
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate speed if previous position exists
        if obj_id in prev_positions:
            prev_x, prev_y = prev_positions[obj_id]
            speed = (
                (center_x - prev_x) ** 2 + (center_y - prev_y) ** 2
            ) ** 0.5  # Euclidean distance
            speed_per_sec = (
                speed * fps
            ) / 100  # Convert pixels/frame to approx real-world speed
            total_speed += speed_per_sec
            count += 1

            # Apply persistent heatmap effect
            cv2.circle(
                heatmap, (center_x, center_y), 25, speed_per_sec * heatmap_intensity, -1
            )

        # Update last position
        prev_positions[obj_id] = (center_x, center_y)

    # Normalize and colorize heatmap
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Apply Gaussian blur to smooth heatmap further
    heatmap_blurred = cv2.GaussianBlur(heatmap_colored, (35, 35), 15)

    # Blend heatmap overlay with the original frame
    overlay = cv2.addWeighted(frame, 0.6, heatmap_blurred, 0.4, 0)

    # Calculate and display average speed
    avg_speed = total_speed / count if count > 0 else 0
    cv2.putText(
        overlay,
        f"Avg Crowd Speed: {avg_speed:.2f} px/s",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    # Write frame to output video
    out.write(overlay)

    # Show live output
    bigger_frame = cv2.resize(
        overlay, (frame_width * 3, frame_height * 3)
    )  # Triples the size
    cv2.imshow("Persistent Heatmap", bigger_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved crowd speed persistent heatmap video: {output_path}")
