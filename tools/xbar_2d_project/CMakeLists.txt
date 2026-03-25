cmake_minimum_required(VERSION 3.10)
project(xbar_simulator_2d_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Repo root from this file location: tools/xbar_2d_project
set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_subdirectory(${REPO_ROOT}/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11_build)

include_directories(${REPO_ROOT})
include_directories(${REPO_ROOT}/include)
include_directories(${REPO_ROOT}/include/eigen)
include_directories(SYSTEM ${REPO_ROOT}/include/eigen)

add_definitions(-DEIGEN_USE_THREADS)
add_definitions(-DEIGEN_USE_OPENMP)

file(GLOB CORE_SRC ${REPO_ROOT}/src/core/*.cpp)
file(GLOB CROSSBAR_SRC ${REPO_ROOT}/src/crossbar_model/*.cpp)
file(GLOB MEMRISTOR_SRC ${REPO_ROOT}/src/memristor_model/*.cpp)

set(MAIN_SRC ${REPO_ROOT}/src/core/xbar_simulator.cpp)
list(REMOVE_ITEM CORE_SRC ${MAIN_SRC})

add_library(xbar_core_2d STATIC
    ${CORE_SRC}
    ${CROSSBAR_SRC}
    ${MEMRISTOR_SRC}
)

target_compile_options(xbar_core_2d PRIVATE
    -fPIC
    -O3
    -march=native
    -ffast-math
    -fopenmp
)

target_link_libraries(xbar_core_2d PRIVATE OpenMP::OpenMP_CXX)

pybind11_add_module(xbar_simulator_2d
    ${REPO_ROOT}/src/python/xbar_wrapper_2d.cpp
)

target_link_libraries(xbar_simulator_2d PRIVATE xbar_core_2d OpenMP::OpenMP_CXX)

target_compile_options(xbar_simulator_2d PRIVATE
    -fPIC
    -O3
    -march=native
    -ffast-math
    -fopenmp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries(xbar_simulator_2d PRIVATE stdc++fs)
endif()

set_target_properties(xbar_simulator_2d PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${REPO_ROOT}/python
)
