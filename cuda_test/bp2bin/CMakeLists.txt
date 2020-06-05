cmake_minimum_required(VERSION 3.0)

project(test)



find_package(ADIOS2 REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)


# create our example 
add_executable(bp2bin bp2bin.cxx)

target_link_libraries(bp2bin adios2::adios2)
