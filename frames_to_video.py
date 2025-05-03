import cv2
import os

# Paths based on your dataset
main_folder = "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\vidf"
output_folder = "C:\\Users\\abhir\\Downloads\\archive\\ucsdpeds\\video"

os.makedirs(output_folder, exist_ok=True)  # Create output directory if it doesn't exist

# Loop through each subfolder in vidf
for subfolder in sorted(os.listdir(main_folder)):
    subfolder_path = os.path.join(main_folder, subfolder)

    if not os.path.isdir(subfolder_path):  # Skip non-folder files
        continue

    images = sorted(
        [img for img in os.listdir(subfolder_path) if img.endswith(".png")]
    )  # Handle .png files
    if not images:
        continue  # Skip empty folders

    # Read the first image to get video dimensions
    first_frame = cv2.imread(os.path.join(subfolder_path, images[0]))
    height, width, layers = first_frame.shape

    # Define video writer
    video_name = os.path.join(output_folder, f"{subfolder}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use "mp4v" for MP4 format
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))  # 10 FPS

    # Write frames into the video
    for image in images:
        frame = cv2.imread(os.path.join(subfolder_path, image))
        video.write(frame)

    video.release()
    print(f"Saved video: {video_name}")

cv2.destroyAllWindows()
