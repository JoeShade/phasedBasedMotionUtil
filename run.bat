@echo off
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m phase_motion_app.app.main
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if not errorlevel 1 (
    python -m phase_motion_app.app.main
    exit /b %ERRORLEVEL%
)

echo Python was not found. Install Python 3 or ensure ^`py^` or ^`python^` is on PATH.
exit /b 1
