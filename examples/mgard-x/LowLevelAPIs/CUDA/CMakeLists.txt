cmake_minimum_required(VERSION 3.19)

project(Example CUDA)
find_package(mgard)

add_executable(Example Example.cu)
target_link_libraries(Example mgard::mgard)
