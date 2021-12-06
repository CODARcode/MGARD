cmake_minimum_required(VERSION 3.14)

project(mgard_x_cpu_compare CXX)
find_package(mgard REQUIRED)

add_executable(BatchTests BatchTests.cpp)
target_link_libraries(BatchTests mgard::mgard)
