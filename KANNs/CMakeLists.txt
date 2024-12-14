cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(KANNs_cpp LANGUAGES CXX CUDA)

# to locate libtorch (MODIFY ACCORDINGLY)
set(CMAKE_PREFIX_PATH "/cluster/home/doezev/Semester Project/libtorch")
find_package(Torch REQUIRED)

find_package(Python3 3.9 EXACT COMPONENTS Interpreter Development REQUIRED)

find_package(CUDA REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O2")

# the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(KANNs_cpp KANNs.cpp)

# Link all libraries
target_link_libraries(KANNs_cpp 
    ${TORCH_LIBRARIES}
    Python3::Python
    ${CUDA_CUDA_LIBRARY}
)
