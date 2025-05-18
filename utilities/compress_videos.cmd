@echo off
cd /d %~dp0
python -m media_tools.compress_videos
pause