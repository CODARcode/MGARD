
cmake_minimum_required(VERSION 3.6)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported.  Please create a build directory "
    "separate from the source directory")
endif()

project(MGARD VERSION 0.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Default to a Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Dependencies
find_package(ZLIB REQUIRED)

# Set library
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_capi.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_nuni.cpp
   )
   
add_library(mgard ${SOURCE_FILES})
#target_include_directories(mgard
#  PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#)
set(CMAKE_INSTALL_BINDIR bin CACHE STRING "Installation runtime subdirectory")
set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Installation library subdirectory")
set(CMAKE_INSTALL_INCLUDEDIR include
  CACHE STRING "Installation include subdirectory")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
 
target_include_directories(mgard
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(mgard PRIVATE ZLIB::ZLIB)

install(
  TARGETS mgard EXPORT mgard
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  FILES include/mgard.h 
        include/mgard_capi.h
        include/mgard_nuni.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

message("-- Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message("        bin: ${CMAKE_INSTALL_BINDIR}")
message("        lib: ${CMAKE_INSTALL_LIBDIR}")
message("    include: ${CMAKE_INSTALL_INCLUDEDIR}")
message("-- Build Type:   ${CMAKE_BUILD_TYPE}")

# Create executables under build/bin
enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(mgard_test ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_test.c)
target_link_libraries(mgard_test mgard)

add_test(NAME mgard_test1
    COMMAND $<TARGET_FILE:mgard_test> 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513_orig 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513.mgard  
    513 513 1e-2
)
  
add_test(NAME mgard_test2
    COMMAND $<TARGET_FILE:mgard_test> 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400_orig 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400.mgard 
    600 400 1e-3
)

add_executable(mgard_sirius_test ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_sirius_test.c)
target_link_libraries(mgard_sirius_test mgard)

add_test(NAME mgard_test3
    COMMAND $<TARGET_FILE:mgard_sirius_test> 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400_orig 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400_coarse.mgard 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400_fine.mgard  
    600 400 1e-2 1e-3
)
