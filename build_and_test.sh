#!/bin/bash
set -e

# Check if pybind11 is present, if not clone it
if [ ! -d "pybind11" ]; then
    echo "Cloning pybind11 repository..."
    git clone https://github.com/pybind/pybind11.git
fi

# Check if Eigen is present, if not clone it
if [ ! -d "include/eigen" ]; then
    echo "Cloning Eigen repository..."
    mkdir -p include
    git clone https://gitlab.com/libeigen/eigen.git include/eigen
fi

# Create build directory if it doesn't exist
mkdir -p build

# Build the project
cd build
echo "Configuring CMake..."
cmake ..
echo "Building the project..."
make -j4

# Return to the root directory
cd ..

# Run the test
#echo "Running the test..."
#python python/test_inference.py 