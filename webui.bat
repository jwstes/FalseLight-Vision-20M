@echo off
REM Activate the virtual environment
call venv\Scripts\activate

REM Start the Uvicorn server with reload enabled
uvicorn server:app --reload

pause
