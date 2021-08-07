cmake_minimum_required(VERSION 3.19)

project(linking LANGUAGES CXX)

add_executable(main src/main.cpp)

# A project using MGARD will inherit a subset of the dependencies of MGARD.
# MGARD provides find modules for some of its dependencies in `cmake/` (in the root of the repository).
# If CMake can't find one of these dependencies when building your project, you can copy the corresponding find module to a location in `CMAKE_MODULE_PATH` (or modify `CMAKE_MODULE_PATH`, as in the next line).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../cmake")

# A config-file package is installed to `${CMAKE_INSTALL_PREFIX}/lib/cmake/mgard` (by default) as part of MGARD.
# If CMake can't find MGARD, make sure that that location is included in `CMAKE_MODULE_PATH`.
find_package(mgard)

target_link_libraries(main mgard::mgard)
