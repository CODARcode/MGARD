cmake_minimum_required(VERSION 3.9)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(smc_xgc)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)
find_package(MGARD REQUIRED)

add_executable(TestXGCBasic TestXGCBasic.cpp)
target_link_libraries(TestXGCBasic adios2::adios2 mgard::mgard MPI::MPI_C)

add_executable(TestXGC4DNonuniform TestXGC4DNonuniform.cpp)
target_link_libraries(TestXGC4DNonuniform adios2::adios2 mgard::mgard)

add_executable(CalcSnormDensity CalcSnormDensity.cpp)
target_link_libraries(CalcSnormDensity adios2::adios2 mgard::mgard)

add_executable(TestXGC5DTemporal TestXGC5DTemporal.cpp)
target_link_libraries(TestXGC5DTemporal adios2::adios2 mgard::mgard)

add_executable(TestXGCDensity TestXGCDensity.cpp)
target_link_libraries(TestXGCDensity adios2::adios2 mgard::mgard)
