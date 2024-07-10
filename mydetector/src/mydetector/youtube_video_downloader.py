import logging
import os

from pytube import YouTube

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - \
                    %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_logger(log_file: str = "youtube_downloader.log") -> None:
    """Set up the logger for the application.

    Args:
    ----
        log_file (str): The name of the log file. Defaults to 'youtube_downloader.log'.

    Returns:
    -------
        None

    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def download_video(url: str, save_path: str) -> None:
    """Download a YouTube video from the given URL and save it to the specified path.

    This function attempts to download the highest resolution version of the video.

    Args:
    ----
        url (str): The URL of the YouTube video to download.
        save_path (str): The directory path where the video will be saved.

    Raises:
    ------
        Exception: If any error occurs during the download process.

    Returns:
    -------
        None

    """
    try:
        yt = YouTube(url)
        video = yt.streams.get_highest_resolution()
        video.download(output_path=save_path)
        logger.info(f"Successfully downloaded: {yt.title}")  # noqa: G004
    except Exception as e:  # noqa: BLE001
        logger.error(f"An error occurred while downloading the video: {e!s}")  # noqa: TRY400, G004


def main() -> None:
    """Main function to run the YouTube video downloader.

    This function prompts the user for a YouTube video URL, creates a 'downloads'
    directory in the current working directory if it doesn't exist, and then
    calls the download_video function to download the video.

    Returns
    -------
        None

    """  # noqa: D401
    setup_logger()
    logger.info("Starting YouTube video downloader")

    video_url = input("Enter the YouTube video URL: ")
    save_directory = os.path.join(os.getcwd(), "downloads/videos")  # noqa: PTH118, PTH109
    os.makedirs(save_directory, exist_ok=True)  # noqa: PTH103
    logger.info(f"Saving video to: {save_directory}")  # noqa: G004

    download_video(video_url, save_directory)

    logger.info("YouTube video downloader finished")


if __name__ == "__main__":
    main()
