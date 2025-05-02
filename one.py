import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'n' model for speed, change to 'm' or 'x' for accuracy

# Video paths
video_path = "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\video\\big_video.avi"
output_path = (
    "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\video\\persistent_heatmap.avi"
)

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Persistent heatmap (starts as empty)
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

print("Generating persistent heatmap...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.2)

    # Count people detected
    people_count = 0

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # Get center

            # Increase intensity (darker the more people pass)
            cv2.circle(heatmap, (center_x, center_y), 15, 1, -1)  # Add heat

            people_count += 1  # Increase count for each detected person

    # Apply decay to prevent full saturation
    heatmap *= 0.98  # Slowly fades older detections (lower = slower decay)

    # Normalize heatmap (0-255) and apply Gaussian blur
    heatmap_normalized = np.uint8(
        255 * (heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap)
    )
    heatmap_blurred = cv2.GaussianBlur(
        heatmap_normalized, (25, 25), 0
    )  # Smooth transitions

    # Apply colormap (heatmap effect)
    heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

    # Overlay heatmap onto original frame
    overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

    # Display total people count
    cv2.putText(
        overlay,
        f"People Count: {people_count}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Write frame to output video
    out.write(overlay)

    # Show live output (optional)
    bigger_frame = cv2.resize(
        overlay, (frame_width * 3, frame_height * 3)
    )  # Triples the size
    cv2.imshow("Persistent Heatmap", bigger_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved persistent heatmap video: {output_path}")
