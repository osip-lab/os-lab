# You need to install ffmpeg to use this script: https://phoenixnap.com/kb/ffmpeg-windows

import subprocess
import os

def convert_to_h265(input_file, output_file, lossless=False):
    """
    Converts a video file to MP4 using H.265 (HEVC) codec.

    :param input_file: Path to the input video file (e.g., "input.mkv").
    :param output_file: Path to the output video file (e.g., "output.mp4").
    """

    if lossless:
        command = [
            "ffmpeg", "-y"
            "-i", input_file,  # Input file
            "-c:v", "ffv1",  # Use FFV1 codec (lossless)
            "-level", "3",  # FFV1 level 3 (better compression)
            "-coder", "1",  # Range coder for better efficiency
            "-context", "1",  # Context model optimization
            "-g", "1",  # Intra-frame only (no inter-frame compression)
            "-pix_fmt", "gray",  # Force 1-channel greyscale output
            "-c:a", "copy",  # Copy audio as-is
            output_file
        ]
    else:
        command = [
            "ffmpeg", "-y",
            "-i", input_file,  # Input file
            "-c:v", "libx265",  # Use H.265 codec for better compression
            "-crf", "28",  # Control quality (lower = better quality, larger file), \in [18, 28]
            "-preset", "slow",  # Encoding speed vs compression tradeoff
            "-pix_fmt", "gray",  # Force 1-channel greyscale output
            "-c:a", "copy",  # Keep original audio without re-encoding
            output_file  # Output file
        ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful! Saved as: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


# Example usage
# convert_to_h265(r"C:\Users\michaeka\Desktop\Video Games\Basler_acA2040-90umNIR__24759755__20250324_155711560.avi",
#                 r"C:\Users\michaeka\Desktop\Video Games\Basler_acA2040-90umNIR__24759755__20250324_155711560.avi")


def convert_all_avi_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".avi"):
                input_path = os.path.join(root, file)
                output_path = os.path.splitext(input_path)[0] + ".mp4"
                convert_to_h265(input_path, output_path)

                # Delete the original file after conversion
                if os.path.exists(input_path):
                    os.remove(input_path)
                    print(f"Deleted original file: {input_path}")

# %%
# Example usage
convert_all_avi_files_in_folder(r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Lab's Dropbox\Laser Phase Plate\Experiments\Results\20250324")