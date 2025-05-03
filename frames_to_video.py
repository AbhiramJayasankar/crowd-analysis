import cv2
import os
import argparse

def frames_to_video(input_folder, output_folder, fps):
    """
    Converts image sequences in subfolders of an input folder into separate videos
    at a specified frame rate.

    Args:
        input_folder (str): Path to the main folder containing subfolders with image sequences.
        output_folder (str): Path to the folder where the output videos will be saved.
        fps (float): Frame rate for the output videos.
    """
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Target FPS: {fps}")

    os.makedirs(output_folder, exist_ok=True)  # Create output directory if it doesn't exist

    # Loop through each item in the input folder
    for item_name in sorted(os.listdir(input_folder)):
        item_path = os.path.join(input_folder, item_name)

        # Process only if it's a directory (subfolder containing frames)
        if not os.path.isdir(item_path):
            print(f"Skipping non-directory item: {item_name}")
            continue

        print(f"Processing subfolder: {item_name}")
        # Find image files (adjust extensions if needed)
        images = sorted(
            [img for img in os.listdir(item_path) if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
        )
        if not images:
            print(f"  No images found in {item_name}. Skipping.")
            continue

        # Read the first image to get video dimensions
        try:
            first_frame_path = os.path.join(item_path, images[0])
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                print(f"  Error reading first frame: {first_frame_path}. Skipping {item_name}.")
                continue
            height, width, layers = first_frame.shape
            print(f"  Detected video dimensions: {width}x{height}")
        except Exception as e:
            print(f"  Error getting dimensions from first frame in {item_name}: {e}. Skipping.")
            continue

        # Define video writer - Use the provided FPS
        video_name = os.path.join(output_folder, f"{item_name}.avi")
        # Use "mp4v" for MP4 format if preferred, ensure correct encoder is installed
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video = cv2.VideoWriter(video_name, fourcc, float(fps), (width, height)) # Use the fps argument

        # Write frames into the video
        print(f"  Writing {len(images)} frames to {video_name} at {fps} FPS...")
        for image_file in images:
            frame_path = os.path.join(item_path, image_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                video.write(frame)
            else:
                print(f"    Warning: Could not read frame {image_file}. Skipping.")

        video.release()
        print(f"  Saved video: {video_name}")

    print("Processing complete for frames_to_video.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image sequences in subfolders to videos.")
    parser.add_argument("input_folder", help="Path to the main folder containing subfolders with image sequences.")
    parser.add_argument("output_folder", help="Path to the folder where output videos will be saved.")
    parser.add_argument("--fps", type=float, default=10.0, help="Frame rate for the output videos (default: 10).") # Added FPS argument
    args = parser.parse_args()

    frames_to_video(args.input_folder, args.output_folder, args.fps) # Pass fps to the function