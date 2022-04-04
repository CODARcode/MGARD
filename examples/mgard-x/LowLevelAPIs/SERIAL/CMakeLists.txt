cmake_minimum_required(VERSION 3.19)

project(Example CXX)
find_package(mgard)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(Example Example.cpp)
target_link_libraries(Example mgard::mgard)
