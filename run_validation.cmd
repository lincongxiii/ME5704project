@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" validation\run_validation.py --open
) else if exist "..\_venv\Scripts\python.exe" (
  "..\_venv\Scripts\python.exe" validation\run_validation.py --open
) else (
  py -3 validation\run_validation.py --open
)
set "VALIDATION_EXIT=%ERRORLEVEL%"
if not "%VALIDATION_EXIT%"=="0" echo Validation failed or dependencies are missing. See validation\README.md and validation\output\pytest.log.
pause
exit /b %VALIDATION_EXIT%
