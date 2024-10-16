cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ODEsolvecpp_torch)

#set torch and python
find_package(Torch REQUIRED)
set(Python3_ROOT_DIR "/opt/homebrew/opt/python@3.9")
find_package(Python3 3.9 EXACT COMPONENTS Interpreter Development REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# the executable from the C++ source file
add_executable(ODEsolvecpp_torch ODEsolvecpp_torch.cpp)

#link Torch and Python
target_link_libraries(ODEsolvecpp_torch "${TORCH_LIBRARIES}" Python3::Python)
set_property(TARGET ODEsolvecpp_torch PROPERTY CXX_STANDARD 17)


# Handle Windows DLL copying, if necessary'
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ODEsolvecpp_torchexample-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:ODEsolvecpp_torch>)
endif (MSVC)

# run the following to cmake
# cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..