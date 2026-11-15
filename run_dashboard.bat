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
    if errorlevel 1 (
        rmdir /s /q .venv-dashboard
        echo.
        echo  The installation failed. If the message above says the file name is too long,
        echo  move the project to a folder with a shorter path, e.g.  C:\Users\%USERNAME%\gna
        echo  and double-click run_dashboard.bat again.
        pause
        exit /b 1
    )
)

echo Starting the dashboard... it opens in your browser in a few seconds: http://localhost:8501
echo Close this window to stop the dashboard.
rem headless: Streamlit would otherwise ask for an e-mail address on its first start and wait;
rem so the browser is opened here, 5 seconds later
start "" /min cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"
".venv-dashboard\Scripts\python.exe" -m streamlit run app\streamlit_app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
pause
