cmake_minimum_required(VERSION 3.22)

project(Example HIP)
find_package(mgard REQUIRED)
find_package(MPI REQUIRED)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set_source_files_properties(Example.cpp PROPERTIES LANGUAGE HIP)
add_executable(Example Example.cpp)
target_link_libraries(Example mgard::mgard MPI::MPI_CXX)
