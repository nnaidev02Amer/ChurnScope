@echo off
REM Starts the ChurnScope app at http://localhost:8501 (Windows, local only).
REM Uses "python" from PATH. To use a specific interpreter that already has the packages:
REM     set CHURN_PYTHON=C:\path\to\python.exe
cd /d "%~dp0"
if not defined CHURN_PYTHON set CHURN_PYTHON=python
"%CHURN_PYTHON%" -c "import streamlit, plotly, sklearn, pandas, joblib" 2>nul
if errorlevel 1 (
  echo The packages needed by the app are missing for "%CHURN_PYTHON%".
  echo Install them with:  "%CHURN_PYTHON%" -m pip install -r requirements.txt
  echo or set CHURN_PYTHON to an interpreter that already has them.
  pause
  exit /b 1
)
REM open the browser a few seconds after the server starts (skipped when NO_BROWSER is set)
if not defined NO_BROWSER start "" /b cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"
"%CHURN_PYTHON%" -m streamlit run app.py --server.port 8501
