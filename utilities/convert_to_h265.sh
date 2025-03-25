#!/bin/bash
# Converts a single file from AVI to H.265 (HEVC) using FFmpeg (you need to have FFmpeg installed)
# Function to convert AVI to H.265
convert_to_h265() {
    input_file="$1"
    output_file="$2"
    ffmpeg -y -i "$input_file" -c:v libx265 -crf 28 -preset slow -c:a copy "$output_file"
}

# Get the folder path of the script (current directory)
folder_path=$(dirname "$(realpath "$0")")

# Iterate over all .avi files in the folder (recursively)
find "$folder_path" -type f -name "*.avi" | while read -r input_file; do
    output_file="${input_file%.avi}.mp4"
    
    # Call the conversion function
    convert_to_h265 "$input_file" "$output_file"
    
    # Delete the original file after conversion
    rm "$input_file"
    echo "Deleted original file: $input_file"
done
