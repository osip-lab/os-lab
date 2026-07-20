"""Rewrite a Windows path on the clipboard as a Dropbox-relative path.

See utilities.utils.dropbox_relative_path_from_clipboard for the logic.
Runnable directly: python utilities/convert_path_to_obsidian_embedding.py
"""
import sys
from pathlib import Path

# repo root on sys.path so `utilities.utils` resolves even when this file is
# run directly (Python only puts the script's own folder on sys.path, not
# the repo root the `utilities.*` absolute imports assume).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.utils import convert_path_to_obsidian_embedding_converter  # noqa: E402

if __name__ == '__main__':
    convert_path_to_obsidian_embedding_converter()
