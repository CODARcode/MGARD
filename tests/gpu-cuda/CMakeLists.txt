cmake_minimum_required(VERSION 3.19)

project(mgard_cuda_example CXX)
find_package(mgard)

add_executable(SimpleExample SimpleExample.cpp)
add_executable(TestDouble TestDouble.cpp)
add_executable(TestFloat TestFloat.cpp)


target_link_libraries(SimpleExample mgard::mgard)
target_link_libraries(TestDouble mgard::mgard)
target_link_libraries(TestFloat mgard::mgard)