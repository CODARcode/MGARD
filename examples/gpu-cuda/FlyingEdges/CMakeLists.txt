cmake_minimum_required(VERSION 3.19)

project(FlyingEdges CXX C CUDA)
find_package(mgard)
find_package(VTKm REQUIRED)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_VERBOSE_MAKEFILE OFF)

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")

add_executable(FlyingEdges FlyingEdges.cu)
target_link_libraries(FlyingEdges vtkm_cont vtkm_filter vtkm_worklet vtkm_rendering mgard::mgard)
set_property(TARGET FlyingEdges PROPERTY CUDA_ARCHITECTURES "70;72;75")
set_target_properties(FlyingEdges PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

