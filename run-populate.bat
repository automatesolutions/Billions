@echo off
echo Populating BILLIONS database with test data...
echo.

.venv\Scripts\python.exe populate-test-data.py

echo.
echo Done! Press any key to exit...
pause

