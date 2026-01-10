@echo off
REM Emotion-Based Music Recommender - Startup Script for Windows
REM This script starts both backend and frontend servers

echo 🎵 Starting Emotion-Based Music Recommender...
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8-3.10
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo ⚠️  Virtual environment not found. Creating one...
    python -m venv venv
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist "venv\.installed" (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    echo. > venv\.installed
    echo ✅ Dependencies installed
)

REM Kill existing processes on ports 8000 and 8501
echo 🧹 Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501') do taskkill /F /PID %%a >nul 2>&1

REM Start backend server
echo 🚀 Starting backend server...
start "Backend Server" cmd /k "cd backend && python main.py"

REM Wait for backend to start
echo ⏳ Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Start frontend
echo 🚀 Starting frontend...
start "Frontend" cmd /k "cd frontend && streamlit run app.py"

REM Wait for frontend to start
timeout /t 3 /nobreak >nul

echo.
echo ================================================
echo 🎉 Application started successfully!
echo ================================================
echo.
echo 📍 Backend API: http://localhost:8000
echo 📍 Frontend UI: http://localhost:8501
echo.
echo 💡 Close the terminal windows to stop the servers
echo.
echo Press any key to exit this launcher...
pause >nul