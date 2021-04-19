cmake_minimum_required(VERSION 3.19)

project(mgard_cuda_example CXX)
find_package(mgard)

add_executable(mgard_cuda_double_test mgard_cuda_double_test.cpp)
add_executable(mgard_cuda_float_test mgard_cuda_float_test.cpp)

target_link_libraries(mgard_cuda_double_test mgard::mgard)
target_link_libraries(mgard_cuda_float_test mgard::mgard)