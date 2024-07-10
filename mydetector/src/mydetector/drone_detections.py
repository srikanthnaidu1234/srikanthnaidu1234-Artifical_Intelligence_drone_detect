import json
import logging
import os
from pathlib import Path

import cv2
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logger(log_file: str = "drone_detection.log") -> None:
    """Set up the logger for the application.

    Args:
    ----
        log_file (str): The name of the log file. Defaults to 'drone_detection.log'.

    Returns:
    -------
        None

    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def extract_frames(video_path: str, frames_dir: str) -> list[str]:
    """Extract frames from a video and save them to a directory.

    Args:
    ----
        video_path (str): Path to the video file.
        frames_dir (str): Directory to save the extracted frames.

    Returns:
    -------
        List[str]: List of paths to the extracted frames.

    """
    os.makedirs(frames_dir, exist_ok=True)  # noqa: PTH103
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")  # noqa: PTH118
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1

    cap.release()
    logger.info(f"Extracted {frame_count} frames from {video_path}")  # noqa: G004
    return frame_paths


def detect_drones(
    frame_paths: list[str], model, detections_dir: str, detections_file: str
) -> None:
    """Detect drones in the frames and save detections to a JSON file.

    Args:
    ----
        frame_paths (List[str]): List of paths to the frames.
        model: The YOLO model for drone detection.
        detections_dir (str): Directory to save frames with detections.
        detections_file (str): Path to the JSON file to save detections.

    Returns:
    -------
        None

    """
    os.makedirs(detections_dir, exist_ok=True)  # noqa: PTH103
    detections = {}

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        results = model(frame)
        detections_df = results.pandas().xyxy[0]

        if not detections_df.empty:
            detection_path = os.path.join(detections_dir, os.path.basename(frame_path))  # noqa: PTH118, PTH119
            cv2.imwrite(detection_path, frame)
            logger.info(f"Detections found in frame: {frame_path}")  # noqa: G004

            # Get the frame number from the file path
            frame_number = int(
                os.path.splitext(os.path.basename(frame_path))[0].split("_")[-1]  # noqa: PTH122, PTH119
            )

            # Convert the detections DataFrame to a list of [x, y, width, height]
            detections[frame_number] = detections_df[  # noqa: PD011
                ["xmin", "ymin", "xmax", "ymax"]
            ].values.tolist()

    # Save detections to a JSON file
    with open(detections_file, "w") as f:  # noqa: PTH123
        json.dump(detections, f)


def main() -> None:
    """Main function to run the drone detection system.

    This function downloads videos, extracts frames, detects drones, and saves frames
    with detections.

    Returns
    -------
        None

    """  # noqa: D401
    setup_logger()
    logger.info("Starting drone detection system")
    videos_dir = "./downloads/videos"
    frames_dir = "./downloads/frames"
    detections_dir = "./downloads/detections"
    detections_file = "./downloads/detections.json"

    os.makedirs(videos_dir, exist_ok=True)  # noqa: PTH103

    # Load YOLO model
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s"
    )  # Replace with yolov7 or yolov8 if needed

    # Process each video
    for video_file in os.listdir(videos_dir):
        video_path = os.path.join(videos_dir, video_file)  # noqa: PTH118
        video_frames_dir = os.path.join(frames_dir, Path(video_file).stem)  # noqa: PTH118
        video_detections_dir = os.path.join(detections_dir, Path(video_file).stem)  # noqa: PTH118

        # Extract frames
        frame_paths = extract_frames(video_path, video_frames_dir)

        # Detect drones
        detect_drones(frame_paths, model, video_detections_dir, detections_file)

    logger.info("Drone detection system finished")


if __name__ == "__main__":
    main()
