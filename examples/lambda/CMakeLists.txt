cmake_minimum_required (VERSION 3.20)

project (xgc-postprocessing)

set (CMAKE_CXX_STANDARD 17)

find_package (MPI COMPONENTS CXX)
find_package (ADIOS2 REQUIRED COMPONENTS CXX MPI)
find_package (OpenMP)
#find_package (MGARD REQUIRED)

add_executable (xgc_postprocess TestXGCPostProcessing.cpp LagrangeOptimizer.cpp KmeansMPI.cpp)
add_executable (xgc_decompress TestXGCDecompress.cpp LagrangeOptimizer.cpp KmeansMPI.cpp)
target_link_libraries (xgc_postprocess adios2::adios2 MPI::MPI_C)
target_link_libraries (xgc_postprocess mgard-library)
target_link_libraries (xgc_decompress adios2::adios2 MPI::MPI_C)
target_link_libraries (xgc_decompress mgard-library)
#include_directories(${CMAKE_SOURCE_DIR})
#include_directories("/gpfs/alpine/csc143/proj-shared/tania/rnd100/MGARD/install-cuda-summit/include")
