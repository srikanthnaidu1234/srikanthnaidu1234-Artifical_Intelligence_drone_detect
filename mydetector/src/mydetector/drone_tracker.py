"""Drone Tracking with Kalman Filter.

This module implements a Kalman filter-based tracking system for drones in videos.
It processes multiple videos from an input directory, applies Kalman filtering
based on provided detections, and generates output videos with tracked drone
trajectories and bounding boxes.

The module requires detection data in JPG format for each input video.

Dependencies:
- OpenCV (cv2)
- NumPy
- filterpy

Usage:
    Update the input_dir, output_dir, and detections_dir in the main() function,
    then run the script.

Author: [Your Name]
Date: [Current Date]
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_kalman_filter():  # noqa: ANN201
    """Set up the Kalman filter for 2D tracking.

    Returns
    -------
        KalmanFilter: Initialized Kalman filter object for 2D tracking.

    The filter is set up with the following parameters:
    - 4D state vector (x, y, dx, dy)
    - 2D measurement vector (x, y)
    - Constant velocity model

    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0  # time step

    kf.F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
    )  # state transition matrix

    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # measurement function

    kf.R = np.eye(2) * 50  # measurement noise
    kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.1)  # process noise

    kf.P *= 1000  # initial state uncertainty
    kf.x = np.zeros((4, 1))  # initial state estimate

    return kf


def process_video(video_path, detections, output_path) -> None:  # noqa: D417, ANN001
    """Process a single video with Kalman filter tracking.

    Args:
    ----
        video_path (str): Path to the input video file.
        detections (dict): Dictionary of detections for each frame.
        output_path (str): Path to save the processed video.

    This function applies Kalman filter tracking to the drone detections,
    draws bounding boxes and trajectories, and saves the processed video.

    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    kf = setup_kalman_filter()
    trajectory = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if str(frame_count) in detections:
            detections_for_frame = detections[str(frame_count)]
            for detection in detections_for_frame:
                x, y, w, h = detection
                center_x, center_y = x + w / 2, y + h / 2

                if not trajectory:  # First detection
                    kf.x = np.array([[center_x], [center_y], [0], [0]])

                z = np.array([[center_x], [center_y]])
                kf.update(z)

                # Draw bounding box
                cv2.rectangle(
                    frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
                )

        kf.predict()

        estimated_x, estimated_y = kf.x[0, 0], kf.x[1, 0]
        trajectory.append((int(estimated_x), int(estimated_y)))

        # Draw trajectory
        if len(trajectory) > 1:
            cv2.polylines(frame, [np.array(trajectory)], False, (0, 0, 255), 2)  # noqa: FBT003

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    logger.info(f"Processed video saved to {output_path}")  # noqa: G004


def process_videos_in_directory(input_dir, output_dir, detections_dir) -> None:  # noqa: D417, ANN001
    """Process all videos in the input directory and save results to the output directory.

    Args:
    ----
        input_dir (str): Path to the directory containing input videos.
        output_dir (str): Path to the directory where processed videos will be saved.
        detections_dir (str): Path to the directory containing detection JPG frames.

    This function iterates through all videos in the input directory, loads corresponding
    detection frames, processes each video with Kalman filter tracking, and saves the
    results in the output directory.

    """
    os.makedirs(output_dir, exist_ok=True)  # noqa: PTH103

    for video_file in os.listdir(input_dir):
        if video_file.endswith(
            (".mp4", ".avi", ".mov")
        ):  # Add more video formats if needed
            video_path = Path(input_dir) / video_file
            output_path = Path(output_dir) / f"tracked_{video_file}"

            # Get the video name without extension
            video_name = video_file.rsplit(".", 1)[0]

            # Path to the directory containing detection frames for this video
            video_detections_dir = Path(detections_dir) / video_name
            logger.info(f"The frames of detection is in :{video_detections_dir}")  # noqa: G004
            if video_detections_dir.is_dir():
                detections = load_detections_from_frames(video_detections_dir)
                if detections:
                    process_video(str(video_path), detections, str(output_path))
                else:
                    logger.warning(
                        f"No valid detections found for {video_file}. Skipping."  # noqa: G004
                    )
            else:
                logger.warning(
                    f"No detection directory found for {video_file}. Skipping."  # noqa: G004
                )


def load_detections_from_frames(detections_dir):  # noqa: ANN201, ANN001
    """Load detections from JPG frames in the given directory.

    Args:
    ----
        detections_dir (Path): Path to the directory containing detection JPG frames.

    Returns:
    -------
        dict: A dictionary of detections, where keys are frame numbers and values are
        ists of [x, y, width, height] for detected objects.

    """
    detections = {}
    for frame_file in sorted(detections_dir.glob("*.jpg")):
        frame_number = int(
            frame_file.stem.split("_")[-1]
        )  # Assuming format like "frame_0001.jpg"

        # Read the frame
        frame = cv2.imread(str(frame_file))

        # Here, you need to implement or call your object detection function
        # This is a placeholder and needs to be replaced with your actual detection logic
        detected_objects = detect_objects_in_frame(frame)

        if detected_objects:
            detections[str(frame_number)] = detected_objects

    return detections


def detect_objects_in_frame(frame):  # noqa: ANN201, ANN001
    """Detect objects in a single frame.

    Args:
    ----
        frame (numpy.ndarray): The input frame as a numpy array.

    Returns:
    -------
        list: A list of [x, y, width, height] for each detected object.

    """
    # This is a placeholder function
    # You need to implement your object detection logic here
    # For example, you might use a pre-trained model like YOLO or SSD
    # Return a list of detections, where each detection is [x, y, width, height]
    # Placeholder implementation (replace with your actual detection code)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # Adjust these thresholds as needed
            detections.append([x, y, w, h])
    return detections


def main() -> None:
    """Main function to run the drone tracking system.

    This function sets up the input, output, and detections directories,
    and initiates the video processing pipeline.
    """  # noqa: D401
    input_dir = "./downloads/videos"
    output_dir = "./downloads/output/videos"
    detections_dir = "./downloads/detections"

    process_videos_in_directory(input_dir, output_dir, detections_dir)


if __name__ == "__main__":
    main()
