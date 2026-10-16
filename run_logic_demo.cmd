@echo off
setlocal
pushd "%~dp0"
set "VISIONDRIVE_PYTHON=python"
if exist "%~dp0.venv-model\Scripts\python.exe" set "VISIONDRIVE_PYTHON=%~dp0.venv-model\Scripts\python.exe"
if exist "%~dp0.venv-inference\Scripts\python.exe" set "VISIONDRIVE_PYTHON=%~dp0.venv-inference\Scripts\python.exe"
echo Starting the model-driven logic viewer.
echo Model: Laptop YOLO11n + metric depth. Camera input: 0.
echo Use --detector esp for the original tiny model, or --esp-host IP for Wi-Fi instructions.
"%VISIONDRIVE_PYTHON%" "%~dp0logic_demo.py" %*
set "VISIONDRIVE_EXIT=%ERRORLEVEL%"
popd
exit /b %VISIONDRIVE_EXIT%
