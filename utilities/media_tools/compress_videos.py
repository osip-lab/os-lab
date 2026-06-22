# You need to install ffmpeg to use this script: https://phoenixnap.com/kb/ffmpeg-windows

import subprocess
import os
from send2trash import send2trash
from automations.core.utils import wait_for_path_from_clipboard

def convert_to_h265(input_file, output_file, compression_rate: int):
    """
    Converts a video file to MP4 using H.265 (HEVC) codec.

    :param input_file: Path to the input video file (e.g., "input.mkv").
    :param output_file: Path to the output video file (e.g., "output.mp4").
    """

    command = [
        "ffmpeg", "-y",
        "-i", input_file,  # Input file
        "-c:v", "libx265",  # Use H.265 codec for better compression
        "-crf", str(compression_rate),  # Control quality (lower = better quality, larger file), \in [18, 28]
        "-preset", "slower",  # Encoding speed vs compression tradeoff
        "-pix_fmt", "yuv420p", # "gray",  # Force 1-channel greyscale output
        "-c:a", "copy",  # Keep original audio without re-encoding
        output_file  # Output file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful! Saved as: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False


def convert_all_avi_files_in_folder(folder_path,
                                    compression_rate=23,
                                    delete_original_files=False,
                                    recursive=False):
    if recursive:
        walker = os.walk(folder_path)
    else:
        walker = [(folder_path, [], os.listdir(folder_path))]

    for root, dirs, files in walker:
        for file in files:
            if file.endswith(".avi") or file.endswith(".MOV"):
                input_path = os.path.join(root, file)
                output_path = os.path.splitext(input_path)[0] + ".mp4"
                if os.path.exists(output_path):
                    print("output path already exists, skipping conversion")
                else:
                    success = convert_to_h265(input_path, output_path, compression_rate)

                    # Delete the original file after conversion
                    if os.path.exists(input_path) and delete_original_files and success and os.path.exists(output_path):
                        send2trash(input_path)
                        print(f"Moved to trash: {input_path}")


def input_with_default(prompt, default):
    user_input = input(f"{prompt} [{default}]: ")
    return user_input if user_input.strip() else default


def main():
    folder_path = wait_for_path_from_clipboard('directory')
    compression_rate = int(input_with_default(
        "Compression rate (a number in range [18, 28]. lower = better quality, larger file), use 23 for default", "23"))
    delete_original_files = input_with_default(r"Delete original files? (y/n)", "n").lower() == "y"
    recursive = input_with_default(r"Run recursively in subfolders? (y/n)", "n").lower() == "y"

    convert_all_avi_files_in_folder(
        folder_path=folder_path,
        compression_rate=compression_rate,
        delete_original_files=delete_original_files,
        recursive=recursive
    )


if __name__ == "__main__":
    main()
