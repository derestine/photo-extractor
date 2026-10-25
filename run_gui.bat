@echo off
rem Launch the Photo Extractor GUI on Windows.
rem Requires uv installed and on PATH: https://docs.astral.sh/uv/getting-started/installation/
cd /d "%~dp0"
uv run python gui.py
if errorlevel 1 (
    echo.
    echo GUI exited with an error. If it says tkinter/tcl is missing, ensure
    echo uv's Python is intact, or install Python from python.org and run:
    echo     python gui.py
    pause
)
