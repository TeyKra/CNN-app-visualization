@echo off
echo [INFO] Activating the virtual environment...
call venv\Scripts\activate

echo [INFO] Launching Streamlit UI...
streamlit run app.py

echo [INFO] Streamlit UI launched. You can now access the application in your browser.
pause