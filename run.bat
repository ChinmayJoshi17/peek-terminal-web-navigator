@echo off
setlocal
title PEEK · Terminal Web Navigator Launcher

:: Activate Python virtual environment if present
if exist ".\.venv\Scripts\activate.bat" (
    call ".\.venv\Scripts\activate.bat"
) else if exist ".\venv\Scripts\activate.bat" (
    call ".\venv\Scripts\activate.bat"
)

:: Verify Python installation in PATH
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10+ and add it to your system PATH.
    pause
    exit /b 1
)

:menu
cls
echo ==============================================================
echo   🔍 PEEK · Terminal Web Navigator Launcher
echo ==============================================================
echo   [1] Launch PEEK Terminal CLI (Interactive Search)
echo   [2] Launch Web Cockpit (Streamlit Dashboard)
echo   [3] Run Benchmark Evaluation Suite (6/6 Tasks)
echo   [4] Exit
echo ==============================================================
set /p choice="Select an option (1-4): "

if "%choice%"=="1" (
    cls
    python cli.py
    pause
    goto menu
)
if "%choice%"=="2" (
    cls
    echo Starting Streamlit Web Cockpit...
    python -m streamlit run ui/dashboard.py
    goto menu
)
if "%choice%"=="3" (
    cls
    echo Running Evaluation Benchmark Suite in headless mode...
    python eval/runner.py --headless
    echo.
    pause
    goto menu
)
if "%choice%"=="4" (
    exit /b 0
)

echo.
echo Invalid choice. Please enter 1, 2, 3, or 4.
timeout /t 2 >nul
goto menu
