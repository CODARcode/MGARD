cmake_minimum_required(VERSION 3.19)

project(adios2-test CXX C)
find_package(mgard)
find_package(ADIOS2 REQUIRED)
find_package(HDF5 REQUIRED)

find_package(MPI REQUIRED)

find_package(CUSZ REQUIRED)

find_package(zfp REQUIRED)

add_executable(adios2-test adios2-test.cpp)
target_link_libraries(adios2-test mgard::mgard CUSZ::cusz zfp::zfp adios2::adios2 MPI::MPI_CXX)

enable_language(CUDA)
set_source_files_properties(adios2-test-external-compressor.cpp PROPERTIES LANGUAGE CUDA)
add_executable(adios2-test-external-compressor adios2-test-external-compressor.cpp)
target_link_libraries(adios2-test-external-compressor mgard::mgard CUSZ::cusz zfp::zfp adios2::adios2 hdf5::hdf5 MPI::MPI_CXX)