@echo off
REM ============================================================
REM  AI Job Assistant - Launcher
REM
REM  KEEP THIS FILE ASCII-ONLY.
REM  cmd.exe parses .bat files using the system ANSI codepage
REM  (GBK on Chinese Windows). The previous version was saved as
REM  UTF-8 with Chinese text, which corrupted the commands
REM  themselves (e.g. "if errorlevel" was read as "rrorlevel").
REM ============================================================
cd /d "%~dp0"

echo ============================================================
echo   AI Job Assistant - starting services
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] "python" not found in PATH.
    echo         Try running:  py -m app.main
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('python --version 2^>^&1') do echo   Python : %%v
echo   Project: %CD%
echo.

if not exist ".env" (
    echo [WARN] .env not found - AI features will return 502.
    echo        Run:  copy .env.example .env   then fill in LLM_API_KEY
    echo.
)

echo Starting BACKEND  ^(FastAPI^)   -^> http://127.0.0.1:8000/docs
start "AI-Job-Assistant BACKEND" cmd /k python -m app.main

echo Waiting 4 seconds for the backend to boot...
REM Use ping, not timeout: "timeout" fails with "input redirection is not
REM supported" whenever stdin is redirected (e.g. when run from a script).
ping -n 5 127.0.0.1 >nul

echo Starting FRONTEND ^(Streamlit^) -^> http://localhost:8501
start "AI-Job-Assistant FRONTEND" cmd /k python -m streamlit run ui/app.py

echo.
echo ============================================================
echo   Two windows opened:
echo     BACKEND  : http://127.0.0.1:8000/docs
echo     FRONTEND : http://localhost:8501
echo.
echo   To stop: close those two windows, or press Ctrl+C in each.
echo ============================================================
ping -n 7 127.0.0.1 >nul
