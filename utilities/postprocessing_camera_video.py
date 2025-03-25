import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import time


def load_video_as_numpy(video_path):
    """
    Loads the video from `video_path` into a numpy array of shape (T, N, M)
    where T is the number of frames, N is the height (number of rows), and
    M is the width (number of columns).
    """
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Get the frame rate and number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale if it isn't already
        if len(frame.shape) == 3:  # Color frame (RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame  # Already grayscale

        frames.append(gray_frame)

    cap.release()
    frames = np.array(frames)
    print(f"Video loaded with shape {frames.shape} (T, N, M)")
    return frames, fps


def plot_intensity_vs_time(intensity, fps):
    """
    Plots the sum of pixel intensities per frame over time.
    """
    times = np.arange(0, len(intensity)) / fps  # Time in seconds
    plt.plot(times, intensity)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Intensity (sum of pixel values)")
    plt.title("Pixel Intensity Over Time")

    # Add cursor to help select a region
    cursor = Cursor(plt.gca(), useblit=True, color='red', linewidth=1)
    plt.show()
    return times


def get_time_range_from_user(times, intensity):
    """
    Allows the user to interactively select a time range in the plot
    and returns the corresponding indices in the `times` array.
    """

    def onselect(xmin, xmax):
        global selected_time_range
        selected_time_range = (xmin, xmax)
        print(f"Selected time range: {xmin} - {xmax} seconds")

    # Plot the intensity vs. time and allow the user to select a range
    plt.plot(times, intensity)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Intensity (sum of pixel values)")
    plt.title("Pixel Intensity Over Time")
    plt.gca().set_xlim([min(times), max(times)])  # Ensure full time range
    plt.gca().set_ylim([min(intensity), max(intensity)])

    # Allow user to select a time range
    span = plt.gca().get_axes().add_patch(plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.2))
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    return selected_time_range


def trim_video_by_time_range(video_array, time_range, fps):
    """
    Trims the video to the selected time range (in seconds).
    Returns a new numpy array of the trimmed video.
    """
    start_time, end_time = time_range
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Trim the video array to include only the selected frames
    trimmed_video = video_array[start_frame:end_frame]
    print(f"Trimmed video shape: {trimmed_video.shape}")
    return trimmed_video


def main(video_path):
    # Step 1: Load the video
    video_array, fps = load_video_as_numpy(video_path)

    # Step 2: Sum the pixel intensities per frame to get `intensity_t`
    intensity_t = video_array.sum(axis=(1, 2))  # Sum over the N and M dimensions (height, width)

    # Step 3: Plot intensity vs. time
    times = plot_intensity_vs_time(intensity_t, fps)

    # Step 4: Allow the user to select a time range from the plot
    selected_time_range = get_time_range_from_user(times, intensity_t)

    # Step 5: Trim the video to the selected time range
    trimmed_video = trim_video_by_time_range(video_array, selected_time_range, fps)

    # You can save or work with the `trimmed_video` array here as needed
    # Example: Saving the trimmed video could be done using OpenCV or other methods


if __name__ == "__main__":
    video_path = r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Lab's Dropbox\Laser Phase Plate\Experiments\Results\20250317\Basler_acA2040-90umNIR__24759755__20250317_141749990.mp4"  # Replace with your actual video path
    main(video_path)
