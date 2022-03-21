cmake_minimum_required(VERSION 3.22)

project(Example HIP)
find_package(mgard)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set_source_files_properties(refactor.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(reconstructor.cpp PROPERTIES LANGUAGE HIP)
add_executable(refactor refactor.cpp)
target_link_libraries(refactor mgard::mgard)
add_executable(reconstructor reconstructor.cpp)
target_link_libraries(reconstructor mgard::mgard)
