@echo off
cd /d %~dp0
python -m video_tools.compress_videos
pause