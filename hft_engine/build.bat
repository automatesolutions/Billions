@echo off
REM HFT Trading Engine Build Script for Windows
REM This script builds the C++ HFT engine and Python bindings

echo Building HFT Trading Engine...

REM Check if we're in the right directory
if not exist "CMakeLists.txt" (
    echo [ERROR] CMakeLists.txt not found. Please run this script from the hft_engine directory.
    exit /b 1
)

REM Create build directory
echo [INFO] Creating build directory...
if not exist "build" mkdir build
cd build

REM Check for required dependencies
echo [INFO] Checking dependencies...

REM Check for CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake is required but not installed.
    echo Please install CMake from https://cmake.org/download/
    exit /b 1
)

REM Check for Visual Studio or MinGW
where cl >nul 2>&1
if errorlevel 1 (
    where g++ >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] C++ compiler is required but not installed.
        echo Please install Visual Studio or MinGW.
        exit /b 1
    )
)

REM Download third-party dependencies
echo [INFO] Downloading third-party dependencies...

REM Create third_party directory
if not exist "..\third_party" mkdir ..\third_party
cd ..\third_party

REM Download WebSocket++
if not exist "websocketpp" (
    echo [INFO] Downloading WebSocket++...
    git clone https://github.com/zaphoyd/websocketpp.git
    if errorlevel 1 (
        echo [ERROR] Failed to download WebSocket++
        exit /b 1
    )
)

REM Download nlohmann/json
if not exist "nlohmann" (
    echo [INFO] Downloading nlohmann/json...
    git clone https://github.com/nlohmann/json.git nlohmann
    if errorlevel 1 (
        echo [ERROR] Failed to download nlohmann/json
        exit /b 1
    )
)

cd ..\build

REM Configure CMake
echo [INFO] Configuring CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    exit /b 1
)

REM Build the project
echo [INFO] Building HFT engine...
cmake --build . --config Release
if errorlevel 1 (
    echo [ERROR] Build failed
    exit /b 1
)

echo [INFO] Build successful!

REM Check if Python bindings were built
if exist "Release\hft_python_bindings.pyd" (
    echo [INFO] Python bindings built successfully!
    
    REM Test Python bindings
    echo [INFO] Testing Python bindings...
    python -c "import sys; sys.path.insert(0, '.'); import hft_python_bindings; print('Python bindings imported successfully!')"
    if errorlevel 1 (
        echo [WARNING] Python bindings test failed, but C++ library was built.
    ) else (
        echo [INFO] Python bindings installed and tested successfully!
    )
) else (
    echo [WARNING] Python bindings not built. Install pybind11 to enable Python integration.
)

REM Run example if it exists
if exist "Release\hft_example.exe" (
    echo [INFO] Running example...
    Release\hft_example.exe
)

echo [INFO] Build completed successfully!
echo [INFO] You can now use the HFT Trading Engine in your BILLIONS system.

REM Create installation script
echo @echo off > ..\install_hft.bat
echo REM Installation script for HFT Trading Engine >> ..\install_hft.bat
echo. >> ..\install_hft.bat
echo echo Installing HFT Trading Engine... >> ..\install_hft.bat
echo. >> ..\install_hft.bat
echo REM Copy library to system location >> ..\install_hft.bat
echo copy "Release\libhft_engine.lib" "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\lib\x64\" >> ..\install_hft.bat
echo copy /E "..\include\" "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\include\hft\" >> ..\install_hft.bat
echo. >> ..\install_hft.bat
echo REM Copy Python bindings if they exist >> ..\install_hft.bat
echo if exist "Release\hft_python_bindings.pyd" copy "Release\hft_python_bindings.pyd" "C:\Python*\Lib\site-packages\" >> ..\install_hft.bat
echo. >> ..\install_hft.bat
echo echo Installation complete! >> ..\install_hft.bat

echo [INFO] Installation script created: install_hft.bat
echo [INFO] Run 'install_hft.bat' to install the engine system-wide.

pause
