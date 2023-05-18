cmake_minimum_required(VERSION 3.19)

project(adios2-test CXX C)
find_package(mgard)
find_package(ADIOS2 REQUIRED)
find_package(MPI REQUIRED)

add_executable(adios2-test adios2-test.cpp)
target_link_libraries(adios2-test mgard::mgard adios2::adios2 MPI::MPI_CXX)
