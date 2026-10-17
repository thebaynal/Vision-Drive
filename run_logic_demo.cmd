@echo off
setlocal
pushd "%~dp0"
set "VISIONDRIVE_PYTHON=python"
if exist "%~dp0.venv-inference\Scripts\python.exe" set "VISIONDRIVE_PYTHON=%~dp0.venv-inference\Scripts\python.exe"
echo Starting the model-driven logic viewer.
echo Default: laptop YOLO11n + metric depth. Use --detector coco for YOLO11n only.
echo The application prints the active model after loading it.
"%VISIONDRIVE_PYTHON%" "%~dp0logic_demo.py" %*
set "VISIONDRIVE_EXIT=%ERRORLEVEL%"
popd
exit /b %VISIONDRIVE_EXIT%
