@echo off
setlocal
cd /d "%~dp0"
if not exist .venv ( py -m venv .venv )
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
streamlit run app/app.py
