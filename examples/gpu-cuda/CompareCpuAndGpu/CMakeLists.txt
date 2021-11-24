cmake_minimum_required(VERSION 3.19)

project(mgard_x_cpu_compare CXX C)
find_package(mgard)

add_executable(BatchTests BatchTests.cpp)
target_link_libraries(BatchTests mgard::mgard)
