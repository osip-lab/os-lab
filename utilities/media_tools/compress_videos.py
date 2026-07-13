# You need to install ffmpeg to use this script: https://phoenixnap.com/kb/ffmpeg-windows

import subprocess
import os
import ctypes
from ctypes import wintypes
from send2trash import send2trash
from utilities.utils import wait_for_path_from_clipboard


def copy_file_timestamps(source_file, target_file):
    """
    Copies the "Date Created", "Date Modified" and "Date Accessed" timestamps
    from source_file to target_file so the output keeps the original's
    creation-time signature (used when ordering files by creation time).

    Modified/accessed times are set via os.utime; the creation time is set via
    the Win32 SetFileTime API (os.utime cannot set creation time on Windows).
    """
    stat = os.stat(source_file)

    # Modified and accessed times work on every platform.
    os.utime(target_file, (stat.st_atime, stat.st_mtime))

    # Creation time is Windows-only and requires the Win32 API.
    if os.name != "nt":
        return

    # Convert a Unix timestamp (seconds since 1970) to a Windows FILETIME,
    # which counts 100-nanosecond intervals since 1601-01-01.
    def to_filetime(unix_time):
        ft = int(unix_time * 10_000_000) + 116_444_736_000_000_000
        return wintypes.FILETIME(ft & 0xFFFFFFFF, ft >> 32)

    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

    handle = ctypes.windll.kernel32.CreateFileW(
        str(target_file), GENERIC_WRITE, 0, None,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, None)
    if handle == -1:  # INVALID_HANDLE_VALUE
        print(f"Could not open {target_file} to set creation time")
        return

    try:
        creation_time = to_filetime(stat.st_ctime)
        if not ctypes.windll.kernel32.SetFileTime(
                handle, ctypes.byref(creation_time), None, None):
            print(f"Could not set creation time for {target_file}")
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)

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

                    # Preserve the original "Date Created" (and modified/accessed)
                    # timestamps so files stay ordered by creation time.
                    if success and os.path.exists(output_path):
                        copy_file_timestamps(input_path, output_path)

                    # Delete the original file after conversion
                    if os.path.exists(input_path) and delete_original_files and success and os.path.exists(output_path):
                        send2trash(input_path)
                        print(f"Moved to trash: {input_path}")


def input_with_default(prompt, default):
    user_input = input(f"{prompt} [{default}]: ")
    return user_input if user_input.strip() else default


def main():
    print(os.getcwd())
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
