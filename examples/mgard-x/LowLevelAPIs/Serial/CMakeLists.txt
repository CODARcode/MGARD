cmake_minimum_required(VERSION 3.19)

project(Example CXX)
find_package(mgard)

add_executable(Example Example.cpp)
target_link_libraries(Example mgard::mgard)
