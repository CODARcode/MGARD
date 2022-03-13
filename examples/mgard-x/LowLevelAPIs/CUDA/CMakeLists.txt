cmake_minimum_required(VERSION 3.19)

project(Example CUDA)
find_package(mgard)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(Example Example.cu)
target_link_libraries(Example mgard::mgard)
