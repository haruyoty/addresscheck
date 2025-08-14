@echo off
title Golf Address Analysis Tool

echo ========================================
echo    Golf Address Analysis Tool
echo ========================================
echo.
echo Starting application...

REM Change to script directory
cd /d "%~dp0"

REM Get WSL path
for %%I in (.) do set "WIN_PATH=%%~fI"
set "WSL_PATH=/mnt/c%WIN_PATH:~2%"
set "WSL_PATH=%WSL_PATH:\=/%"

echo Windows Path: %WIN_PATH%
echo WSL Path: %WSL_PATH%
echo.

REM Check Python3 in WSL
wsl python3 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python3 not found in WSL
    echo Please install Python3 in WSL:
    echo   sudo apt update
    echo   sudo apt install python3 python3-pip
    echo.
    pause
    exit /b 1
)

echo Python3 found in WSL
echo.

REM Install dependencies on first run
if not exist ".deps_installed" (
    echo Installing required packages...
    echo Please wait...
    echo.
    
    wsl bash -c "cd '%WSL_PATH%' && pip3 install -r requirements.txt"
    if errorlevel 1 (
        echo ERROR: Failed to install packages
        echo.
        pause
        exit /b 1
    )
    
    echo. > .deps_installed
    echo Package installation completed
    echo.
)

echo Starting Streamlit app...
echo.
echo If browser doesn't open automatically, visit:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Start Streamlit app
wsl bash -c "cd '%WSL_PATH%' && python3 -m streamlit run app.py --browser.gatherUsageStats false"

echo.
echo Application stopped
pause