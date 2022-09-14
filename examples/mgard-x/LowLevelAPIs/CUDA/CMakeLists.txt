cmake_minimum_required(VERSION 3.19)

project(Example CUDA CXX)
find_package(mgard)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")
add_executable(Example Example.cu)
target_link_libraries(Example mgard::mgard)
