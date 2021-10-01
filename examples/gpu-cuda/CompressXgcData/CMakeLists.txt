cmake_minimum_required(VERSION 3.19)

project(mgard_cuda_xgc CXX C)
find_package(mgard)
find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)

add_executable(TestXGCAbsoluteError TestXGCAbsoluteError.cpp)
target_link_libraries(TestXGCAbsoluteError adios2::adios2 mgard::mgard MPI::MPI_CXX)
