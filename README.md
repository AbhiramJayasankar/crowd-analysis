# Video Analysis and Utility Toolkit

## Description

This project provides a collection of Python scripts designed for video analysis and manipulation. It includes tools for generating heatmaps based on presence and speed, tracking object speed using computer vision techniques (YOLOv8 and SORT), and utility functions for converting image sequences to video and merging multiple videos. Graphical User Interfaces (GUIs) are provided for easier interaction with the scripts.

## Features

* **Presence Heatmap:** Generates a heatmap overlay on a video, visualizing areas with persistent object presence (e.g., people).
* **Speed Tracker:** Detects and tracks objects (specifically configured for persons using YOLOv8) in a video using the SORT algorithm, calculates their speed in pixels per second, displays bounding boxes with IDs and individual speeds, and shows the average crowd speed.
* **Speed Heatmap:** Creates a heatmap overlay visualizing the speed of tracked objects within the video.
* **Frames to Video:** Converts sequences of images stored in separate subfolders into individual video files at a specified frame rate[cite: 1].
* **Merge Videos:** Combines multiple video files located within a specified directory into a single output video file.
* **Analysis GUI (`video_analyzer_gui.py`):** A user-friendly interface to select a video file and run the Presence Heatmap, Speed Tracker, or Speed Heatmap analyses.
* **Utility GUI (`video_util_gui.py`):** A user-friendly interface for the Frames-to-Video and Merge Videos utilities, allowing easy selection of input/output paths and parameters like FPS.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    This will install all necessary libraries listed in the `requirements.txt` file.

## Usage

### Graphical Interfaces (Recommended)

* **Video Analysis:**
    ```bash
    python video_analyzer_gui.py
    ```
    Use the GUI to "Choose Video File" and then click the desired analysis button ("Presence Heatmap", "Speed Tracker", "Speed Heatmap"). Status updates and potential errors will be shown in the GUI's status bar and console.

* **Video Utilities:**
    ```bash
    python video_util_gui.py
    ```
    Use the GUI sections to select input/output directories/files for "Frames to Video" (including setting the FPS) or "Merge Videos" and click the corresponding "Run" button.

### Command Line (Alternative)

The individual analysis and utility scripts can also be run directly from the command line. They accept arguments for input/output paths and parameters.

* **Presence Heatmap:**
    ```bash
    python presence_heatmap.py <path_to_input_video>
    ```
    **
* **Speed Tracker:**
    ```bash
    python speed_tracker.py <path_to_input_video>
    ```
    **
* **Speed Heatmap:**
    ```bash
    python speed_heatmap.py <path_to_input_video>
    ```
    **
* **Frames to Video:**
    ```bash
    python frames_to_video.py <input_folder_with_image_subfolders> <output_folder_for_videos> --fps <frame_rate>
    ```
    *(e.g., `--fps 25`) [cite: 1]*
* **Merge Videos:**
    ```bash
    python merge_videos.py <input_folder_with_videos> <output_video_file_path>
    ```
    **

## Scripts Overview

* **`frames_to_video.py`:** Converts image sequences to videos[cite: 1].
* **`merge_videos.py`:** Merges multiple videos into one.
* **`presence_heatmap.py`:** Creates heatmaps based on object presence using YOLOv8.
* **`speed_tracker.py`:** Tracks objects using YOLOv8+SORT and calculates/displays speed.
* **`speed_heatmap.py`:** Generates heatmaps based on tracked object speed using YOLOv8+SORT.
* **`video_analyzer_gui.py`:** Tkinter GUI for launching analysis scripts.
* **`video_util_gui.py`:** Tkinter GUI for launching utility scripts.
* **`requirements.txt`:** Lists project dependencies.

## Dependencies

Key libraries used in this project include:

* `opencv-python`: For video and image processing.
* `ultralytics`: For YOLOv8 object detection.
* `numpy`: For numerical operations.
* `sort-track`: For object tracking (SORT algorithm implementation).
* `tkinter` (Python standard library): For the GUIs.

*(Refer to `requirements.txt` for the full list and specific versions)*.