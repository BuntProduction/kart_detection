@echo off
setlocal
set HERE=%~dp0
set PY=%HERE%.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [ERROR] Python venv not found: %PY%
  echo Create it and install requirements first.
  exit /b 1
)
"%PY%" -m streamlit run "%HERE%streamlit_app.py"
