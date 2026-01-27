$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$python = Join-Path $here '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python venv not found at $python. Create it and install requirements first."
}

& $python -m streamlit run "$here\streamlit_app.py"
