@echo off
setlocal EnableDelayedExpansion

echo ╔════════════════════════════════════════╗
echo ║      MRAgent Installer v0.1.0          ║
echo ╚════════════════════════════════════════╝
echo.

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org or the Microsoft Store.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo [OK] Found Python %PYTHON_VER%

:: 2. Setup Virtual Environment
echo.
echo [INFO] Setting up virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo    Created .venv
) else (
    echo    Found existing .venv
)

:: 3. Install Dependencies
echo.
echo [INFO] Installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed!

:: 4. Create Launcher Script
echo.
echo [INFO] Creating 'mragent' launcher...
set "LAUNCHER_DIR=%USERPROFILE%\bin"
if not exist "%LAUNCHER_DIR%" mkdir "%LAUNCHER_DIR%"

set "LAUNCHER_PATH=%LAUNCHER_DIR%\mragent.cmd"
echo @echo off > "%LAUNCHER_PATH%"
echo cd /d "%CD%" >> "%LAUNCHER_PATH%"
echo call .venv\Scripts\activate.bat >> "%LAUNCHER_PATH%"
echo python main.py %%* >> "%LAUNCHER_PATH%"

:: 5. Add to Path
echo [INFO] Adding to PATH...
set "PATH_Check=%PATH%"
echo %PATH_Check% | find /i "%LAUNCHER_DIR%" >nul
if %errorlevel% neq 0 (
    setx PATH "%LAUNCHER_DIR%;%PATH%"
    echo [OK] Added to PATH. Please restart your terminal.
) else (
    echo [OK] Already in PATH.
)

echo.
echo ========================================
echo 🎉 Installation Complete!
echo Type 'mragent' to start your assistant.
echo ========================================
pause
