cmake_minimum_required(VERSION 3.19)

project(mgard_cuda_example CXX)
find_package(mgard)

add_executable(SimpleExample SimpleExample.cpp)
target_link_libraries(SimpleExample mgard::mgard)
add_executable(TestXGCAbsoluteError TestXGCAbsoluteError.cpp)
target_link_libraries(TestXGCAbsoluteError adios2::adios2 mgard::mgard MPI::MPI_CXX)
