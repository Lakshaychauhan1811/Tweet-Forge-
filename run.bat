@echo off
setlocal
echo ========================================
echo Marketing Tweet Generator (LangChain)
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
  echo Error: Python is not installed or not in PATH
  echo Please install Python 3.9+ from https://python.org
  pause
  exit /b 1
)

if not exist .env (
  echo Creating .env from env_example.txt
  copy env_example.txt .env >nul
)

if not exist env (
  py -3.11 -m venv env
)
call env\Scripts\activate
pip install --upgrade pip >nul
pip install -r requirements.txt

python start.py

endlocal
