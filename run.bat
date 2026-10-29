@echo off
chcp 65001 >nul
cls

where python >nul 2>&1
if errorlevel 1 goto :no_python

if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat

:menu
cls
echo ==============================================================
echo  ◆ PEEK : Terminal Web Navigator Launcher
echo ==============================================================
echo   [1] Launch PEEK Terminal CLI (Interactive Search)
echo   [2] Launch Web Cockpit (Streamlit Dashboard)
echo   [3] Run Benchmark Evaluation Suite (6/6 Tasks)
echo   [4] Exit
echo ==============================================================
set /p choice="Select an option (1-4): "

if "%choice%"=="1" goto :run_cli
if "%choice%"=="2" goto :run_dashboard
if "%choice%"=="3" goto :run_eval
if "%choice%"=="4" exit /b 0

echo Invalid choice, please enter 1, 2, 3, or 4.
pause
goto menu

:run_cli
cls
python cli.py
pause
goto menu

:run_dashboard
cls
echo Starting Streamlit Cockpit...
python -m streamlit run ui/dashboard.py
goto menu

:run_eval
cls
python eval/runner.py --headless
pause
goto menu

:no_python
echo [ERROR] Python is not installed or not found in PATH.
pause
exit /b 1