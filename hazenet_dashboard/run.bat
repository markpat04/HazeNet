@echo off
REM HazeNet Mission Control launcher
cd /d %~dp0
echo Starting HazeNet Mission Control...
echo Open http://localhost:8765
C:\Users\mark\miniconda3\Scripts\conda run -n hazenet --no-capture-output python serve.py
pause
