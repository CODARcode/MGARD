cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_STANDARD 11)
# ENABLE ADIOS2_ROOT
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(mgard_cuda C CXX)
find_package(mgard)
find_package(CUDA)

add_executable(mgard_check_cuda_and_cpu mgard_check_cuda_and_cpu.cpp)

target_link_libraries(mgard_check_cuda_and_cpu mgard::mgard-library ${CUDA_CUDART_LIBRARY} ${CUDA_cudadevrt_LIBRARY})
