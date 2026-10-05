@echo off
REM ============================================================
REM  Environment self-check
REM
REM  KEEP THIS FILE ASCII-ONLY (see note in 启动项目.bat:
REM  cmd.exe parses .bat using the system ANSI codepage).
REM ============================================================
cd /d "%~dp0.."

echo ============================================================
echo   Environment self-check - AI Job Assistant
echo ============================================================
echo.

echo [1/5] Python interpreter
echo ------------------------------------------------------------
where python
python --version
if errorlevel 1 (
    echo   [ERROR] python is not available. Try:  py --version
)
echo.

echo [2/5] Required packages
echo ------------------------------------------------------------
python -c "import fastapi, streamlit, openai, jieba, pydantic, httpx, pypdf, pytest, dotenv, uvicorn, multipart; print('  all packages import OK')"
if errorlevel 1 echo   [ERROR] some packages are missing
echo.

echo [3/5] .env configuration
echo ------------------------------------------------------------
if exist ".env" (
    python -c "from app.config import get_settings, mask_secret; s=get_settings(); print('  base_url =', s.llm_base_url); print('  model    =', s.llm_model); print('  api_key  =', mask_secret(s.llm_api_key))"
) else (
    echo   [WARN] .env not found - AI features will return 502
    echo          Run:  copy .env.example .env   then fill in LLM_API_KEY
)
echo.

echo [4/5] Unit tests
echo ------------------------------------------------------------
python -m pytest -q -p no:cacheprovider
echo.

echo [5/5] Agent example (offline mode, no API key needed)
echo ------------------------------------------------------------
python examples\01_add_agent.py "3 + 5" --mock
echo.

echo ============================================================
echo   Done. If anything failed above, send the output to the AI.
echo ============================================================
pause
