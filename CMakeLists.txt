cmake_minimum_required(VERSION 3.10)
project(xbar_simulator_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position independent code for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add pybind11
add_subdirectory(pybind11)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/eigen)

# Define Eigen as a system include to suppress warnings
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include/eigen)

# Add OpenMP definitions for Eigen
add_definitions(-DEIGEN_USE_THREADS)
add_definitions(-DEIGEN_USE_OPENMP)

# Source files for the crossbar simulator
file(GLOB CORE_SRC "src/core/*.cpp")
file(GLOB CROSSBAR_SRC "src/crossbar_model/*.cpp")
file(GLOB MEMRISTOR_SRC "src/memristor_model/*.cpp")
#file(GLOB TEST_SRC "tests/*.cpp")

# Separate main file from core sources
set(MAIN_SRC "src/core/xbar_simulator.cpp")
list(REMOVE_ITEM CORE_SRC ${MAIN_SRC})

# Create a library for the core simulator functionality
add_library(xbar_core STATIC
    ${CORE_SRC}
    ${CROSSBAR_SRC}
    ${MEMRISTOR_SRC}
)
target_compile_options(xbar_core PRIVATE 
    -fPIC 
    -O3 
    -march=native 
    -ffast-math 
    -fopenmp 
    
)
target_link_libraries(xbar_core PRIVATE OpenMP::OpenMP_CXX)

# Create the main simulator executable
add_executable(xbar_simulator_exe ${MAIN_SRC})
target_link_libraries(xbar_simulator_exe PRIVATE xbar_core OpenMP::OpenMP_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries(xbar_simulator_exe PRIVATE stdc++fs)
endif()
set_target_properties(xbar_simulator_exe PROPERTIES
    OUTPUT_NAME xbar_simulator
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

# Create the pybind11 module
pybind11_add_module(xbar_simulator 
    src/python/xbar_wrapper.cpp
)
target_link_libraries(xbar_simulator PRIVATE xbar_core OpenMP::OpenMP_CXX)
target_compile_options(xbar_simulator PRIVATE 
    -fPIC 
    -O3 
    -march=native  
    -ffast-math 
    -fopenmp  
)

# Link against filesystem library if needed (for C++17 std::filesystem)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries(xbar_simulator PRIVATE stdc++fs)
endif()

# Set output directory
set_target_properties(xbar_simulator PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)

# Installation
install(TARGETS xbar_simulator DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python)

# Add test executables
foreach(test_file ${TEST_SRC})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE xbar_core OpenMP::OpenMP_CXX)
    
    # Link against filesystem library if needed
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_link_libraries(${test_name} PRIVATE stdc++fs)
    endif()
    
    # Set output directory for test executables
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    )
endforeach() 