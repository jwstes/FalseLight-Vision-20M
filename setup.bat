@echo off
REM Create virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Install required packages
pip install tensorflow==2.16.1 numpy pillow opencv-python fastapi uvicorn python-multipart

REM Change directory to dlib to install the appropriate dlib wheel
cd dlib

REM Get the Python version and store it in the variable "pyver"
for /f "tokens=2 delims= " %%a in ('python --version') do set pyver=%%a
echo Detected Python version: %pyver%

REM Install the appropriate dlib wheel based on the detected Python version
echo %pyver% | findstr "3.7" >nul && (
    echo Installing dlib for Python 3.7...
    python -m pip install dlib-19.22.99-cp37-cp37m-win_amd64.whl
    goto AfterDlib
)
echo %pyver% | findstr "3.8" >nul && (
    echo Installing dlib for Python 3.8...
    python -m pip install dlib-19.22.99-cp38-cp38-win_amd64.whl
    goto AfterDlib
)
echo %pyver% | findstr "3.9" >nul && (
    echo Installing dlib for Python 3.9...
    python -m pip install dlib-19.22.99-cp39-cp39-win_amd64.whl
    goto AfterDlib
)
echo %pyver% | findstr "3.10" >nul && (
    echo Installing dlib for Python 3.10...
    python -m pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
    goto AfterDlib
)
echo %pyver% | findstr "3.11" >nul && (
    echo Installing dlib for Python 3.11...
    python -m pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
    goto AfterDlib
)
echo %pyver% | findstr "3.12" >nul && (
    echo Installing dlib for Python 3.12...
    python -m pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
    goto AfterDlib
)

:AfterDlib
REM Return to the parent directory
cd ..

REM Install the face-recognition package
pip install face-recognition

pause
