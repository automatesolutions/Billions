cmake_minimum_required(VERSION 3.16)
project(HFTTradingEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find OpenSSL for HTTPS requests
find_package(OpenSSL REQUIRED)

# Find libcurl
find_package(CURL REQUIRED)

# Include directories
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Add third-party libraries
# WebSocket++ (header-only)
set(WEBSOCKETPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/websocketpp)
include_directories(${WEBSOCKETPP_INCLUDE_DIR})

# nlohmann/json (header-only)
set(NLOHMANN_JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann)
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

# Boost libraries
find_package(Boost REQUIRED COMPONENTS system thread chrono)

# Source files
set(SOURCES
    src/market_data_ingestion.cpp
    src/signal_processor.cpp
    src/order_executor.cpp
    src/hft_engine.cpp
    src/alpaca_websocket_client.cpp
)

# Header files
set(HEADERS
    include/market_data_ingestion.h
    include/signal_processor.h
    include/order_executor.h
    include/hft_engine.h
    include/alpaca_websocket_client.h
)

# Create the library
add_library(hft_engine STATIC ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(hft_engine
    ${CURL_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    Boost::system
    Boost::thread
    Boost::chrono
    Threads::Threads
)

# Compiler flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(hft_engine PRIVATE
        -O3
        -march=native
        -mtune=native
        -flto
        -DNDEBUG
    )
    target_link_options(hft_engine PRIVATE -flto)
endif()

# Compiler flags for debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hft_engine PRIVATE
        -g
        -O0
        -DDEBUG
    )
endif()

# Create example executable
add_executable(hft_example examples/hft_example.cpp)
target_link_libraries(hft_example hft_engine)

# Create Python bindings (optional)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(hft_python_bindings python_bindings/bindings.cpp)
    target_link_libraries(hft_python_bindings PRIVATE hft_engine)
    
    # Create Alpaca WebSocket Python bindings
    pybind11_add_module(alpaca_websocket src/alpaca_websocket_python.cpp)
    target_link_libraries(alpaca_websocket PRIVATE hft_engine)
endif()

# Installation
install(TARGETS hft_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Create package
include(CPack)
set(CPACK_PACKAGE_NAME "HFTTradingEngine")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION "High-Frequency Trading Engine")
set(CPACK_PACKAGE_CONTACT "your-email@example.com")
