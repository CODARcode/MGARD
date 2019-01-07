cmake_minimum_required(VERSION 3.6)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported.  Please create a build directory "
    "separate from the source directory")
endif()

project(MGARD VERSION 0.1)

# Some boilerplate to setup nice output directories
set(CMAKE_INSTALL_BINDIR bin CACHE STRING "Installation runtime subdirectory")
set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Installation library subdirectory")
set(CMAKE_INSTALL_INCLUDEDIR include
  CACHE STRING "Installation include subdirectory")
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
endif()

# Setup shared library defaults.  If explicitly specified somehow, then default
# to that.  Otherwise base the default on whether or not shared libs are even
# supported (oddities of the CrayPE).
include(CMakeDependentOption)
get_property(SHARED_LIBS_SUPPORTED GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
cmake_dependent_option(BUILD_SHARED_LIBS
  "Build shared libraries (so/dylib/dll)." ${SHARED_LIBS_SUPPORTED}
  "SHARED_LIBS_SUPPORTED" OFF
)
mark_as_advanced(BUILD_SHARED_LIBS)

# Always build with PIC support if available
if(SHARED_LIBS_SUPPORTED)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Default to a release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
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

# Make sure we require C++11.  Use meta-compile features if available,
# otherwise use specific language features
if(NOT (CMAKE_VERSION VERSION_LESS 3.9))
  target_compile_features(mgard PUBLIC cxx_std_11)
else()
  target_compile_features(mgard PUBLIC cxx_auto_type cxx_nullptr)
endif()

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

# Create executables under build/bin
include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_executable(mgard_test ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_test.c)
  target_link_libraries(mgard_test mgard)

  add_test(NAME mgard_test1
    COMMAND mgard_test
      ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513_orig
      ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513.mgard
      513 513 1 1e-2
  )
  
  add_test(NAME mgard_test2
    COMMAND mgard_test
      ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400_orig
      ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400.mgard
      600 400 1 1e-2
  )

  add_test(NAME mgard_test3
    COMMAND mgard_test
      ${CMAKE_CURRENT_SOURCE_DIR}/data/zort_111_160_15.dat
      ${CMAKE_CURRENT_SOURCE_DIR}/data/zort_111_160_15.mgard
      11 160 15 1e-3
  )
endif()

message("-- Configuration summary:")
message("")
message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message("         bin: ${CMAKE_INSTALL_BINDIR}")
message("         lib: ${CMAKE_INSTALL_LIBDIR}")
message("     include: ${CMAKE_INSTALL_INCLUDEDIR}")
message("")
message("  Build Type: ${CMAKE_BUILD_TYPE}")
message("  Shared Lib: ${BUILD_SHARED_LIBS}")
message("     Testing: ${BUILD_TESTING}")

