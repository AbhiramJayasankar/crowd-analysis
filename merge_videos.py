import cv2
import os

# Folder containing all video files
video_folder = "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\video"
output_video_path = (
    "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\video\\big_video.avi"
)

# Get list of all video files
video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".avi")])

if not video_files:
    print("No videos found!")
    exit()

# Open first video to get properties
first_video = cv2.VideoCapture(os.path.join(video_folder, video_files[0]))
frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(first_video.get(cv2.CAP_PROP_FPS))
first_video.release()

# Create VideoWriter for final merged video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_video = cv2.VideoWriter(
    output_video_path, fourcc, fps, (frame_width, frame_height)
)

# Append all videos
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    print(f"Merging: {video_file}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_video.write(frame)

    cap.release()

output_video.release()
print(f"Saved merged video: {output_video_path}")
