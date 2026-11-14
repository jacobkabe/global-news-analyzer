@echo off
rem Double-click to open the Global News Analyzer dashboard.
rem The first time it makes a small Python environment (.venv-dashboard) and installs streamlit and pandas.
rem It needs the pipeline outputs in data\raw\outputs\ (from OneDrive: Semestral project - TA\dashboard-data).

cd /d "%~dp0"

set MISSING=
for %%F in (articles_clean.csv sentiment.csv sentiment_explain.csv entities.csv) do (
    if not exist "data\raw\outputs\%%F" set MISSING=1
)
if defined MISSING (
    echo.
    echo  The dashboard data is missing.
    echo  Copy the 4 CSV files from OneDrive  ^(TA2\Semestral project - TA\dashboard-data^)
    echo  into this folder:  %~dp0data\raw\outputs\
    echo.
    if not exist "data\raw\outputs" mkdir "data\raw\outputs"
    explorer "data\raw\outputs"
    pause
    exit /b 1
)

if not exist ".venv-dashboard\Scripts\python.exe" (
    echo First start: installing the dashboard packages, this takes a minute...
    python -m venv .venv-dashboard || py -3 -m venv .venv-dashboard
    if not exist ".venv-dashboard\Scripts\python.exe" (
        echo Python was not found. Install Python 3.11 or newer from python.org and try again.
        pause
        exit /b 1
    )
    ".venv-dashboard\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements-dashboard.txt
)

echo Starting the dashboard... it opens in your browser. Close this window to stop it.
".venv-dashboard\Scripts\python.exe" -m streamlit run app\streamlit_app.py --browser.gatherUsageStats false
pause
