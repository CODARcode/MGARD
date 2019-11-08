cmake_minimum_required(VERSION 3.6)


if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported.  Please create a build directory "
    "separate from the source directory")
endif()

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Extract the version string from the header file and setup the project version
# accordingly
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/include/mgard.h MGARD_VERSION
  REGEX "// version: "
  LIMIT_COUNT 1)
string(REGEX MATCH "[0-9]+(\.[0-9])+" MGARD_VERSION "${MGARD_VERSION}")
project(MGARD VERSION ${MGARD_VERSION} LANGUAGES C CXX CUDA)

# Some boilerplate to setup nice output directories
set(CMAKE_INSTALL_BINDIR bin CACHE STRING "Installation runtime subdirectory")
set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Installation library subdirectory")
set(CMAKE_INSTALL_INCLUDEDIR include
  CACHE STRING "Installation include subdirectory")
if(WIN32 AND NOT CYGWIN)
  set(CMAKE_INSTALL_CMAKEDIR CMake
    CACHE STRING "Installation CMake subdirectory")
else()
  set(CMAKE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/mgard
    CACHE STRING "Installation CMake subdirectory")
endif()

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
find_package(ADIOS2)


# Set library
add_library(mgard
  src/mgard.cpp       include/mgard.h
  src/mgard_api.cpp  include/mgard_api.h
  src/mgard_nuni.cpp  include/mgard_nuni.h
  src/mgard_float.cpp       include/mgard_float.h
  src/mgard_api_float.cpp  include/mgard_api_float.h
  src/mgard_nuni_float.cpp  include/mgard_nuni_float.h
  src/mgard_cuda.cu       include/mgard_cuda.h
  src/mgard_nuni_2d_cuda.cu       include/mgard_nuni_2d_cuda.h
  src/mgard_api_cuda.cpp  include/mgard_api_cuda.h
  src/mgard_cuda_compacted.cu       include/mgard_cuda_compacted.h
  src/mgard_cuda_helper.cu          include/mgard_cuda_helper.h
  src/mgard_cuda_helper_internal.cu include/mgard_cuda_helper_internal.h
)
target_include_directories(mgard
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mgard PRIVATE ZLIB::ZLIB  ${CMAKE_DL_LIBS})


# Make sure we require C++11.  Use meta-compile features if available,
# otherwise use specific language features
if(NOT (CMAKE_VERSION VERSION_LESS 3.9))
  target_compile_features(mgard PUBLIC cxx_std_11)
else()
  target_compile_features(mgard PUBLIC cxx_auto_type cxx_nullptr)
endif()

# Set library version information
set_target_properties(mgard PROPERTIES
  VERSION ${MGARD_VERSION}
  SOVERSION ${MGARD_VERSION_MAJOR}
  CUDA_SEPARABLE_COMPILATION ON
)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DCPU=0 -DGPU=1")
SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DCPU=0 -DGPU=1")

install(
  TARGETS mgard EXPORT mgard-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  FILES include/mgard.h 
        include/mgard_api.h
        include/mgard_nuni.h
	include/mgard_float.h 
        include/mgard_api_float.h
        include/mgard_nuni_float.h
  	include/mgard_api_cuda.h
        include/mgard_cuda.h
	include/mgard_cuda_compacted.h
	include/mgard_cuda_helper.h
	include/mgard_cuda_helper_internal.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Add all targets to the build-tree export set
export(TARGETS mgard NAMESPACE mgard::
  FILE "${PROJECT_BINARY_DIR}/mgard-targets.cmake"
)

# Create the mgard-config.cmake and mgard-config-version files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/mgard-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
configure_file(mgard-config.cmake.in
  "${PROJECT_BINARY_DIR}/mgard-config.cmake" @ONLY
)

# Install the mgard-config.cmake and mgard-config-version.cmake
install(
  FILES
    "${PROJECT_BINARY_DIR}/mgard-config.cmake"
    "${PROJECT_BINARY_DIR}/mgard-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" COMPONENT dev
)

# Install the export set for use with the install-tree
install(EXPORT mgard-targets NAMESPACE mgard::
  DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" COMPONENT dev
)

# Install pkg-config files for non-cmake projects
configure_file(mgard.pc.in "${PROJECT_BINARY_DIR}/mgard.pc" @ONLY)
install(
  FILES "${PROJECT_BINARY_DIR}/mgard.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig" COMPONENT dev
)

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_executable(mgard_test_cuda ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_test_cuda.cpp)
  target_link_libraries(mgard_test_cuda mgard  ${CMAKE_DL_LIBS})
  set_target_properties(mgard_test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  
  add_executable(mgard_test ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_test.cpp)
  target_link_libraries(mgard_test mgard  ${CMAKE_DL_LIBS})
  set_target_properties(mgard_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(mgard_check ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_check.cpp)
  target_link_libraries(mgard_check mgard  ${CMAKE_DL_LIBS})
  set_target_properties(mgard_check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(mgard_check_cuda ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_check_cuda.cpp)
  target_link_libraries(mgard_check_cuda mgard  ${CMAKE_DL_LIBS})
  set_target_properties(mgard_check_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  if(ADIOS2_FOUND)
    add_subdirectory(tests)
  endif()

  add_test(NAME mgard_test1
    COMMAND mgard_test 0
      ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513_orig
      ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513.mgard
      513 513 1 1e-2 0 
  )
  
  add_test(NAME mgard_test2
    COMMAND mgard_test 0
      ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400_orig
      ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400.mgard
      600 400 1 1e-2 1 
  )

  add_test(NAME mgard_test3
    COMMAND mgard_test 0
      ${CMAKE_CURRENT_SOURCE_DIR}/data/zort_111_160_15.dat
      ${CMAKE_CURRENT_SOURCE_DIR}/data/zort_111_160_15.mgard
      11 160 15 1e-3 2 
  )

  add_test(NAME mgard_test1_cuda
    COMMAND mgard_test_cuda 0
      ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513_orig
      ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513.mgard
      513 513 1 1e-2 0 
  )
  
  add_test(NAME mgard_test2_cuda
    COMMAND mgard_test_cuda 0
      ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400_orig
      ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400.mgard
      600 400 1 1e-2 1 
  )

  add_test(NAME mgard_test3_cuda
    COMMAND mgard_test_cuda 0
      ${CMAKE_CURRENT_SOURCE_DIR}/data/zort_111_160_15.dat
      ${CMAKE_CURRENT_SOURCE_DIR}/data/zort_111_160_15.mgard
      11 160 15 1e-3 2 
  )
  add_test(NAME mgard_correctness_check
    COMMAND mgard_check 
	${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513_orig
	${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513.mgard
	513 513 1 0.001 0
  )
  add_test(NAME mgard_correctness_check_cuda
    COMMAND mgard_check_cuda 
        ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513_orig
        ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513.mgard
        513 513 1 0.001 0
  )




endif()


message("-- Configuration summary:")
message("")
message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message("         bin: ${CMAKE_INSTALL_BINDIR}")
message("         lib: ${CMAKE_INSTALL_LIBDIR}")
message("     include: ${CMAKE_INSTALL_INCLUDEDIR}")
message("       cmake: ${CMAKEf_INSTALL_CMAKEDIR}")
message("")
message("  Build Type: ${CMAKE_BUILD_TYPE}")
message("  Shared Lib: ${BUILD_SHARED_LIBS}")
message("     Testing: ${BUILD_TESTING}")

