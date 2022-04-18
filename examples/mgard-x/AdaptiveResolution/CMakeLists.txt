cmake_minimum_required(VERSION 3.19)

project(TestAdaptiveResolution CUDA CXX)
find_package(mgard REQUIRED)
find_package(VTKm REQUIRED)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")

add_executable(TestAdaptiveResolution TestAdaptiveResolution.cu)
target_link_libraries(TestAdaptiveResolution mgard::mgard vtkm_rendering vtkm_filter vtkm_io)
add_executable(TestFlyingEdges TestFlyingEdges.cu)
target_link_libraries(TestFlyingEdges mgard::mgard vtkm_rendering vtkm_filter vtkm_io)
