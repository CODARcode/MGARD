cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_STANDARD 11)
# ENABLE ADIOS2_ROOT
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(mgard_cuda C CXX)
find_package(mgard)

add_executable(mgard_check_cuda_and_cpu mgard_check_cuda_and_cpu.cpp)

target_link_libraries(mgard_check_cuda_and_cpu mgard::mgard)
