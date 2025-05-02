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
output_path = "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\video\\crowd_speed.avi"

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

print("Estimating crowd speed...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.3)

    detections = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
            detections.append(
                [x1, y1, x2, y2, 1, 1]
            )  # 1 is confidence (needed for SORT)

    if detections is []:
        detections.append(np.empty((0, 5)))

    # Track objects
    tracked_objects = tracker.update(np.array(detections), 0)

    total_speed = 0
    count = 0

    for obj in tracked_objects:
        print(obj)
        x1, y1, x2, y2, obj_id, _, _ = list(
            map(int, obj)
        )  # Extract object ID and position
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

            # Display speed
            cv2.putText(
                frame,
                f"{speed_per_sec:.2f} px/s",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                2,
            )

        # Update last position
        prev_positions[obj_id] = (center_x, center_y)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate and display average speed
    avg_speed = total_speed / count if count > 0 else 0
    cv2.putText(
        frame,
        f"Avg Crowd Speed: {avg_speed:.2f} px/s",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    # Write frame to output video
    out.write(frame)

    # Show live output
    bigger_frame = cv2.resize(frame, (frame_width * 3, frame_height * 3))
    cv2.imshow("Crowd Speed Estimation", bigger_frame)
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved crowd speed video: {output_path}")
