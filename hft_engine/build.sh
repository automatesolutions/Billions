#!/bin/bash

# HFT Trading Engine Build Script
# This script builds the C++ HFT engine and Python bindings

set -e

echo "Building HFT Trading Engine..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found. Please run this script from the hft_engine directory."
    exit 1
fi

# Create build directory
print_status "Creating build directory..."
mkdir -p build
cd build

# Check for required dependencies
print_status "Checking dependencies..."

# Check for CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake is required but not installed."
    exit 1
fi

# Check for C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    print_error "C++ compiler (g++ or clang++) is required but not installed."
    exit 1
fi

# Check for required libraries
missing_deps=()

# Check for libcurl
if ! pkg-config --exists libcurl; then
    missing_deps+=("libcurl")
fi

# Check for OpenSSL
if ! pkg-config --exists openssl; then
    missing_deps+=("openssl")
fi

# Check for Boost
if ! pkg-config --exists boost; then
    missing_deps+=("boost")
fi

if [ ${#missing_deps[@]} -ne 0 ]; then
    print_warning "Missing dependencies: ${missing_deps[*]}"
    print_status "Installing dependencies..."
    
    # Detect package manager and install dependencies
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y libcurl4-openssl-dev libssl-dev libboost-all-dev
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install -y libcurl-devel openssl-devel boost-devel
    elif command -v brew &> /dev/null; then
        # macOS
        brew install curl openssl boost
    else
        print_error "Cannot detect package manager. Please install dependencies manually."
        exit 1
    fi
fi

# Download third-party dependencies
print_status "Downloading third-party dependencies..."

# Create third_party directory
mkdir -p ../third_party
cd ../third_party

# Download WebSocket++
if [ ! -d "websocketpp" ]; then
    print_status "Downloading WebSocket++..."
    git clone https://github.com/zaphoyd/websocketpp.git
fi

# Download nlohmann/json
if [ ! -d "nlohmann" ]; then
    print_status "Downloading nlohmann/json..."
    git clone https://github.com/nlohmann/json.git nlohmann
fi

cd ../build

# Configure CMake
print_status "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_STANDARD=17 \
         -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native"

# Build the project
print_status "Building HFT engine..."
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    print_status "Build successful!"
    
    # Check if Python bindings were built
    if [ -f "hft_python_bindings.so" ] || [ -f "hft_python_bindings.pyd" ]; then
        print_status "Python bindings built successfully!"
        
        # Install Python bindings
        print_status "Installing Python bindings..."
        python3 -c "
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
try:
    import hft_python_bindings
    print('Python bindings imported successfully!')
except ImportError as e:
    print(f'Error importing Python bindings: {e}')
    sys.exit(1)
"
        
        if [ $? -eq 0 ]; then
            print_status "Python bindings installed and tested successfully!"
        else
            print_warning "Python bindings installation failed, but C++ library was built."
        fi
    else
        print_warning "Python bindings not built. Install pybind11 to enable Python integration."
    fi
    
    # Run example if it exists
    if [ -f "hft_example" ]; then
        print_status "Running example..."
        ./hft_example
    fi
    
else
    print_error "Build failed!"
    exit 1
fi

print_status "Build completed successfully!"
print_status "You can now use the HFT Trading Engine in your BILLIONS system."

# Create installation script
cat > ../install_hft.sh << 'EOF'
#!/bin/bash
# Installation script for HFT Trading Engine

echo "Installing HFT Trading Engine..."

# Copy library to system location
sudo cp build/libhft_engine.a /usr/local/lib/
sudo cp -r include/ /usr/local/include/hft/

# Copy Python bindings if they exist
if [ -f "build/hft_python_bindings.so" ]; then
    sudo cp build/hft_python_bindings.so /usr/local/lib/python3.*/site-packages/
fi

echo "Installation complete!"
EOF

chmod +x ../install_hft.sh

print_status "Installation script created: install_hft.sh"
print_status "Run './install_hft.sh' to install the engine system-wide."
