cmake_minimum_required(VERSION 3.19)

project(MDR CUDA CXX)
find_package(mgard)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")
add_executable(refactor refactor.cu)
target_link_libraries(refactor mgard::mgard)
# add_executable(reconstructor reconstructor.cu)
# target_link_libraries(reconstructor mgard::mgard)
