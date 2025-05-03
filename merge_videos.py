import cv2
import os
import argparse

def merge_videos(input_folder, output_video_path):
    """
    Merges all video files found in a specified folder into a single output video.

    Args:
        input_folder (str): Path to the folder containing the video files to merge.
        output_video_path (str): Path for the final merged output video file.
    """
    print(f"Input video folder: {input_folder}")
    print(f"Output video path: {output_video_path}")

    # Get list of all video files (adjust extensions if needed)
    video_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))])

    if not video_files:
        print(f"No video files found in {input_folder}!")
        return # Use return instead of exit for better integration

    print(f"Found {len(video_files)} videos to merge.")

    # Open first video to get properties (width, height, fps)
    first_video_path = os.path.join(input_folder, video_files[0])
    try:
        first_video = cv2.VideoCapture(first_video_path)
        if not first_video.isOpened():
             print(f"Error opening first video: {first_video_path}. Cannot determine properties.")
             return
        frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = first_video.get(cv2.CAP_PROP_FPS)
        # Handle potential issues reading FPS
        if fps <= 0:
            print("Warning: Could not read FPS from the first video. Defaulting to 25 FPS.")
            fps = 25
        first_video.release()
        print(f"Detected video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")
    except Exception as e:
        print(f"Error getting properties from first video {first_video_path}: {e}")
        if 'first_video' in locals() and first_video.isOpened():
            first_video.release()
        return

    # Create VideoWriter for final merged video
    # Use "mp4v" for MP4 format if preferred, ensure correct encoder is installed
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not output_video.isOpened():
        print(f"Error: Could not open VideoWriter for output file: {output_video_path}")
        return

    # Append all videos
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  Warning: Could not open video {video_file}. Skipping.")
            continue

        print(f"  Merging: {video_file}")
        frame_count = 0
        while True: # Use while True and break
            ret, frame = cap.read()
            if not ret:
                break # End of this video file
            # Optional: Check if frame dimensions match the output video
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                print(f"    Warning: Frame size mismatch in {video_file}. Resizing frame.")
                frame = cv2.resize(frame, (frame_width, frame_height))
            output_video.write(frame)
            frame_count += 1

        cap.release()
        print(f"    -> Merged {frame_count} frames.")

    output_video.release()
    print("-" * 20)
    print(f"Saved merged video: {output_video_path}")
    print("Processing complete for merge_videos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple video files from a folder into one.")
    parser.add_argument("input_folder", help="Path to the folder containing video files to merge.")
    parser.add_argument("output_video", help="Path for the final merged output video file (e.g., merged_output.avi).")
    args = parser.parse_args()

    merge_videos(args.input_folder, args.output_video)