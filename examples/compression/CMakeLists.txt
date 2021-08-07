cmake_minimum_required(VERSION 3.19)

project(compression LANGUAGES CXX)

add_executable(main src/main.cpp)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../cmake")

find_package(mgard)

target_link_libraries(main mgard::mgard)
