$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m streamlit run dashboard/app.py
