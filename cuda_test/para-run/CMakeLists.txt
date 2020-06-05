cmake_minimum_required(VERSION 3.0)

project(test)


include("FindMGARD.cmake")
find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
add_executable(test test.cxx)

target_link_libraries(test MGARD::MGARD ${MPI_C_LIBRARIES})
