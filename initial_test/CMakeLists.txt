cmake_minimum_required(VERSION 3.18)
project(KVCacheBandwidthTester LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find Kvikio
find_path(KVIKIO_INCLUDE_DIR 
    NAMES kvikio/file_handle.hpp
    PATHS 
        /usr/local/include
        /opt/conda/include
        $ENV{CONDA_PREFIX}/include
        /usr/include
)

if(NOT KVIKIO_INCLUDE_DIR)
    message(FATAL_ERROR "Kvikio headers not found. Please install kvikio.")
endif()

find_library(KVIKIO_LIBRARY
    NAMES kvikio
    PATHS
        /usr/local/lib
        /opt/conda/lib
        $ENV{CONDA_PREFIX}/lib
        /usr/lib
)

if(NOT KVIKIO_LIBRARY)
    message(FATAL_ERROR "Kvikio library not found. Please install kvikio.")
endif()


# Create executable
add_executable(kv_cache_bandwidth_tester
    ${CMAKE_SOURCE_DIR}/kv_cache_bandwidth_tester.cpp
)

# Set target properties
set_target_properties(kv_cache_bandwidth_tester PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"  # Support various GPU architectures
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(kv_cache_bandwidth_tester PRIVATE
    ${KVIKIO_INCLUDE_DIR}
    ${MATPLOTLIBCPP_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(kv_cache_bandwidth_tester PRIVATE
    ${KVIKIO_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cufile
    ${Python3_LIBRARIES}
    pthread
)

# Compiler flags for optimization
target_compile_options(kv_cache_bandwidth_tester PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -mtune=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Linker flags
target_link_options(kv_cache_bandwidth_tester PRIVATE
    -Wl,--as-needed
)

# Add custom target for installation script
add_custom_target(install_dependencies
    COMMAND ${CMAKE_COMMAND} -E echo "Installing dependencies..."
    COMMAND conda install -c rapidsai -c conda-forge kvikio python=3.9
    COMMAND pip install numpy
    COMMENT "Installing required dependencies"
)

# Print build information
message(STATUS "Build configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  KVIKIO_INCLUDE_DIR: ${KVIKIO_INCLUDE_DIR}")
message(STATUS "  KVIKIO_LIBRARY: ${KVIKIO_LIBRARY}")
message(STATUS "  MATPLOTLIBCPP_INCLUDE_DIR: ${MATPLOTLIBCPP_INCLUDE_DIR}")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  Python Version: ${Python3_VERSION}")

# Installation
install(TARGETS kv_cache_bandwidth_tester
    RUNTIME DESTINATION bin
)

# Create a simple run script
configure_file(
    ${CMAKE_SOURCE_DIR}/run_benchmark.sh.in
    ${CMAKE_BINARY_DIR}/run_benchmark.sh
    @ONLY
)

# Create run script template
file(WRITE ${CMAKE_SOURCE_DIR}/run_benchmark.sh.in
"#!/bin/bash
# KV Cache Bandwidth Benchmark Runner
# Optimized for AMD Threadripper Pro WX3995

# Set environment variables for optimal performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=64
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Kvikio optimizations for COMAT mode
export KVIKIO_BOUNCE_BUFFER_SIZE=16777216
export KVIKIO_GDS_THRESHOLD=1048576
export KVIKIO_THREAD_POOL_NTHREADS=32

# System optimizations
echo 'Setting system optimizations...'
sudo sh -c 'echo mq-deadline > /sys/block/md*/queue/scheduler' 2>/dev/null || true
sudo sh -c 'echo 0 > /sys/block/md*/queue/rotational' 2>/dev/null || true
sudo sh -c 'echo 1024 > /sys/block/md*/queue/nr_requests' 2>/dev/null || true

# Set CPU governor to performance
sudo cpupower frequency-set -g performance 2>/dev/null || echo 'cpupower not available'

# Clear filesystem caches
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo 'Starting KV Cache bandwidth benchmark...'
echo 'Target: AMD Threadripper Pro WX3995 with 11-NVME RAID0'
echo '================================================='

# Run the benchmark
./kv_cache_bandwidth_tester \$@

echo 'Benchmark completed. Check results:'
echo '  - kv_cache_bandwidth_results.png (performance graph)'
echo '  - kv_cache_benchmark_results.csv (raw data)'
"
)