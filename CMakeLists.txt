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
project(MGARD VERSION ${MGARD_VERSION})

#-----------------------------------------------------------------------------
# mgard_option(variable doc [initial])
#   Provides an option if it is not already defined.
# This can be replaced when CMake 3.13 is our cmake_minimum_required
macro (mgard_option variable)
  if (NOT DEFINED "${variable}")
    option("${variable}" ${ARGN})
  endif ()
endmacro ()

#Configurable Options
mgard_option(MGARD_ENABLE_CUDA "Enable Cuda support" ON)
mgard_option(MGARD_ENABLE_CUDA_DEBUG "Enable Cuda support with debug" OFF)

# Some boilerplate to setup nice output directories
include(GNUInstallDirs)
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
#if(SHARED_LIBS_SUPPORTED)
#  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#endif()

# Default to a release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

if(MGARD_ENABLE_CUDA_DEBUG)
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_DEBUG")
  SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_DEBUG")
endif()

# Dependencies
find_package(ZLIB REQUIRED)
find_package(ADIOS2)

if(NOT MGARD_ENABLE_CUDA)

  # Set library
  add_library(mgard
    src/mgard.cpp       include/mgard.h
    src/mgard_mesh.cpp  include/mgard_mesh.hpp
    src/mgard_compress.cpp  include/mgard_compress.hpp
    src/mgard_api.cpp  include/mgard_api.h
    src/mgard_nuni.cpp  include/mgard_nuni.h
  )
else()
  enable_language(CUDA)
  add_library(mgard
    src/mgard.cpp       include/mgard.h
    src/mgard_mesh.cpp  include/mgard_mesh.hpp
    src/mgard_compress.cpp  include/mgard_compress.hpp
    src/mgard_api.cpp  include/mgard_api.h
    src/mgard_nuni.cpp  include/mgard_nuni.h
  # CUDA part  
    src/cuda/mgard_cuda.cu                  include/cuda/mgard_cuda.h
    src/cuda/mgard_cuda_add_level.cu        include/cuda/mgard_cuda_add_level.h
    src/cuda/mgard_cuda_assign_num_level.cu include/cuda/mgard_cuda_assign_num_level.h
    src/cuda/mgard_cuda_copy_level.cu       include/cuda/mgard_cuda_copy_level.h
    src/cuda/mgard_cuda_mass_multiply.cu    include/cuda/mgard_cuda_mass_multiply.h
    src/cuda/mgard_cuda_pi_Ql.cu            include/cuda/mgard_cuda_pi_Ql.h
    src/cuda/mgard_cuda_prolongate.cu       include/cuda/mgard_cuda_prolongate.h
    src/cuda/mgard_cuda_subtract_level.cu   include/cuda/mgard_cuda_subtract_level.h
    src/cuda/mgard_cuda_restriction.cu      include/cuda/mgard_cuda_restriction.h
    src/cuda/mgard_cuda_solve_tridiag.cu    include/cuda/mgard_cuda_solve_tridiag.h
    src/cuda/mgard_cuda_linear_quantize.cu  include/cuda/mgard_cuda_linear_quantize.h

    src/cuda/mgard_cuda_handle.cu          include/cuda/mgard_cuda_handle.h
    src/cuda/mgard_api_cuda.cu             include/mgard_api_cuda.h
    src/cuda/mgard_cuda_helper.cu          include/cuda/mgard_cuda_helper.h
    src/cuda/mgard_cuda_common_internal.cu include/cuda/mgard_cuda_common_internal.h
    src/cuda/mgard_cuda_compact_helper.cu  include/cuda/mgard_cuda_compact_helper.h

    src/cuda/mgard_cuda_prep_2d.cu          include/cuda/mgard_cuda_prep_2d.h
    src/cuda/mgard_cuda_refactor_2d.cu      include/cuda/mgard_cuda_refactor_2d.h
    src/cuda/mgard_cuda_recompose_2d.cu     include/cuda/mgard_cuda_recompose_2d.h
    src/cuda/mgard_cuda_postp_2d.cu         include/cuda/mgard_cuda_postp_2d.h
    src/cuda/mgard_cuda_prep_3d.cu          include/cuda/mgard_cuda_prep_3d.h
    src/cuda/mgard_cuda_refactor_3d.cu      include/cuda/mgard_cuda_refactor_3d.h
    src/cuda/mgard_cuda_recompose_3d.cu     include/cuda/mgard_cuda_recompose_3d.h
    src/cuda/mgard_cuda_postp_3d.cu         include/cuda/mgard_cuda_postp_3d.h
    include/cuda/mgard_cuda_kernels.h
)
endif()

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
if(NOT MGARD_ENABLE_CUDA)
  set_target_properties(mgard PROPERTIES
    VERSION ${MGARD_VERSION}
    SOVERSION ${MGARD_VERSION_MAJOR}
  )
else()
  set_target_properties(mgard PROPERTIES
    VERSION ${MGARD_VERSION}
    SOVERSION ${MGARD_VERSION_MAJOR}
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

install(
  TARGETS mgard EXPORT mgard-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  FILES include/mgard.h 
        include/mgard_mesh.hpp
        include/mgard_compress.hpp
        include/mgard_api.h
        include/mgard_nuni.h
	"${PROJECT_BINARY_DIR}/include/MGARDConfig.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create executables under build/bin
include(CTest)
if(BUILD_TESTING)
#  enable_testing()
  add_executable(mgard_test ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard_test.cpp)
  target_link_libraries(mgard_test mgard  ${CMAKE_DL_LIBS})

  add_executable(mgard_check_cuda_and_cpu ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/mgard_check_cuda_and_cpu.cpp)
  target_link_libraries(mgard_check_cuda_and_cpu mgard  ${CMAKE_DL_LIBS})
  set_target_properties(mgard_check_cuda_and_cpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  
  add_subdirectory(tests)

  add_test(NAME mgard_test1
    COMMAND mgard_test double
      ${CMAKE_CURRENT_SOURCE_DIR}/data/u3_513x513_orig
      513 513 1 1e-2 0 
  )
  
  add_test(NAME mgard_test2
    COMMAND mgard_test double
      ${CMAKE_CURRENT_SOURCE_DIR}/data/data_600x400_orig
      600 400 1 1e-2 1 
  )

  add_test(NAME mgard_test3
    COMMAND mgard_test double
      ${CMAKE_CURRENT_SOURCE_DIR}/data/zort_111_160_15.dat
      11 160 15 1e-3 2 
  )
endif()

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

configure_file(
  include/MGARDConfig.h.in
  ${PROJECT_BINARY_DIR}/include/MGARDConfig.h.in
)

configure_file(
  ${PROJECT_BINARY_DIR}/include/MGARDConfig.h.in
  ${PROJECT_BINARY_DIR}/include/MGARDConfig.h
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

message("-- Configuration summary:")
message("")
message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message("         bin: ${CMAKE_INSTALL_BINDIR}")
message("         lib: ${CMAKE_INSTALL_LIBDIR}")
message("     include: ${CMAKE_INSTALL_INCLUDEDIR}")
message("       cmake: ${CMAKE_INSTALL_CMAKEDIR}")
message("")
message("  Build Type: ${CMAKE_BUILD_TYPE}")
message("  Shared Lib: ${BUILD_SHARED_LIBS}")
message("     Testing: ${BUILD_TESTING}")
message("        CUDA: ${MGARD_ENABLE_CUDA}")
message("  CUDA-Debug: ${MGARD_ENABLE_CUDA_DEBUG}")

