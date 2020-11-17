cmake_minimum_required(VERSION 3.6)


if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported.  Please create a build directory "
    "separate from the source directory")
endif()

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Extract the version string from the header file and setup the project version
# accordingly
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/include/mgard.hpp MGARD_VERSION
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
mgard_option(MGARD_ENABLE_CUDA "Enable Cuda support" OFF)
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

if(MGARD_ENABLE_CUDA)
  enable_language(CUDA)
  set(MGARD_CUDA_SRC
      src/cuda/mgard_cuda.cu                 
      src/cuda/mgard_cuda_add_level.cu       
      src/cuda/mgard_cuda_assign_num_level.cu
      src/cuda/mgard_cuda_copy_level.cu      
      src/cuda/mgard_cuda_mass_multiply.cu   
      src/cuda/mgard_cuda_pi_Ql.cu           
      src/cuda/mgard_cuda_prolongate.cu      
      src/cuda/mgard_cuda_subtract_level.cu  
      src/cuda/mgard_cuda_restriction.cu     
      src/cuda/mgard_cuda_solve_tridiag.cu   
      src/cuda/mgard_cuda_linear_quantize.cu 
      src/cuda/mgard_cuda_handle.cu          
      src/cuda/mgard_api_cuda.cu             
      src/cuda/mgard_cuda_helper.cu          
      src/cuda/mgard_cuda_common_internal.cu 
      src/cuda/mgard_cuda_compact_helper.cu  
      src/cuda/mgard_cuda_prep_2d.cu         
      src/cuda/mgard_cuda_refactor_2d.cu     
      src/cuda/mgard_cuda_recompose_2d.cu    
      src/cuda/mgard_cuda_postp_2d.cu        
      src/cuda/mgard_cuda_prep_3d.cu         
      src/cuda/mgard_cuda_refactor_3d.cu     
      src/cuda/mgard_cuda_recompose_3d.cu    
      src/cuda/mgard_cuda_postp_3d.cu)

  set(MGARD_CUDA_HEADER
      include/cuda/mgard_cuda.h
      include/cuda/mgard_cuda_add_level.h
      include/cuda/mgard_cuda_assign_num_level.h
      include/cuda/mgard_cuda_copy_level.h
      include/cuda/mgard_cuda_mass_multiply.h
      include/cuda/mgard_cuda_pi_Ql.h
      include/cuda/mgard_cuda_prolongate.h
      include/cuda/mgard_cuda_subtract_level.h
      include/cuda/mgard_cuda_restriction.h
      include/cuda/mgard_cuda_solve_tridiag.h
      include/cuda/mgard_cuda_linear_quantize.h
      include/cuda/mgard_cuda_handle.h
      include/mgard_api_cuda.h
      include/cuda/mgard_cuda_helper.h
      include/cuda/mgard_cuda_common_internal.h
      include/cuda/mgard_cuda_compact_helper.h
      include/cuda/mgard_cuda_prep_2d.h
      include/cuda/mgard_cuda_refactor_2d.h
      include/cuda/mgard_cuda_recompose_2d.h
      include/cuda/mgard_cuda_postp_2d.h
      include/cuda/mgard_cuda_prep_3d.h
      include/cuda/mgard_cuda_refactor_3d.h
      include/cuda/mgard_cuda_recompose_3d.h
      include/cuda/mgard_cuda_postp_3d.h
      include/cuda/mgard_cuda_kernels.h)

  set (MGARD_CUDA_SEPARATE_COMPILE_COMPILATION_OPTION 
       CUDA_SEPARABLE_COMPILATION ON)
  if(MGARD_ENABLE_CUDA_DEBUG)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_DEBUG")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_DEBUG")
  endif()
endif()

# Dependencies
find_package(ZLIB REQUIRED)
find_package(ADIOS2)
find_package(ZSTD)
find_package(MOAB)

if(MOAB_FOUND)
	set(MGARD_MOAB_HEADER
include/blas.hpp
include/utilities.hpp
include/data.hpp
include/UniformEdgeFamilies.hpp
include/LinearQuantizer.hpp
include/SituatedCoefficientRange.hpp
include/MultilevelCoefficientQuantizer.hpp

include/measure.hpp
include/LinearOperator.hpp
include/pcg.hpp
include/MassMatrix.hpp
include/MeshLevel.hpp
include/MeshHierarchy.hpp
include/MeshRefiner.hpp
include/UniformMeshRefiner.hpp
include/UniformMeshHierarchy.hpp
include/UniformRestriction.hpp
include/norms.hpp
include/estimators.hpp
include/EnumeratedMeshRange.hpp
include/indicators.hpp
include/IndicatorInput.hpp
	)
	set(MGARD_MOAB_TEMPLATE_DEFINITIONS
include/blas.tpp
include/utilities.tpp
include/data.tpp
include/UniformEdgeFamilies.tpp
include/LinearQuantizer.tpp
include/SituatedCoefficientRange.tpp
include/MultilevelCoefficientQuantizer.tpp
	)
	set(MGARD_MOAB_SOURCE
src/measure.cpp
src/LinearOperator.cpp
src/pcg.cpp
src/MassMatrix.cpp
src/MeshLevel.cpp
src/MeshHierarchy.cpp
src/MeshRefiner.cpp
src/UniformMeshRefiner.cpp
src/UniformMeshHierarchy.cpp
src/UniformRestriction.cpp
src/norms.cpp
src/estimators.cpp
src/EnumeratedMeshRange.cpp
src/indicators.cpp
src/IndicatorInput.cpp
	)
endif()

# Set library
add_library(mgard-library STATIC
  src/mgard_compress.cpp  include/mgard_compress.hpp
  src/TensorMeshHierarchyIteration.cpp  include/TensorMeshHierarchyIteration.hpp include/TensorMeshHierarchyIteration.tpp
  ${MGARD_CUDA_SRC}   ${MGARD_CUDA_HEADER}
  ${MGARD_MOAB_HEADER} ${MGARD_MOAB_TEMPLATE_DEFINITIONS} ${MGARD_MOAB_SOURCE}
)
set_target_properties(mgard-library PROPERTIES OUTPUT_NAME mgard)

target_include_directories(mgard-library
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/mgard>
  ${MOAB_INCLUDE_DIRS}
)
target_link_libraries(mgard-library PRIVATE ZLIB::ZLIB ${CMAKE_DL_LIBS} PUBLIC ${MOAB_LIBRARIES})
if(ZSTD_FOUND)
  target_link_libraries(mgard-library PRIVATE ZSTD::ZSTD ${CMAKE_DL_LIBS})
endif()

option(DEFINE_MGARD_TIMING "Enable/disable MGARD timing" OFF)
if(DEFINE_MGARD_TIMING)
  add_definitions(-DMGARD_TIMING)
endif()

# Make sure we require C++11.  Use meta-compile features if available,
# otherwise use specific language features
if(NOT (CMAKE_VERSION VERSION_LESS 3.9))
  target_compile_features(mgard-library PUBLIC cxx_std_17)
else()
  target_compile_features(mgard-library PUBLIC cxx_auto_type cxx_nullptr)
endif()

set_target_properties(mgard-library PROPERTIES
  VERSION ${MGARD_VERSION}
  SOVERSION ${MGARD_VERSION_MAJOR}
  ${MGARD_CUDA_SEPARATE_COMPILE_COMPILATION_OPTION}
)

install(
  TARGETS mgard-library EXPORT mgard-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  FILES include/mgard.hpp
        include/mgard_compress.hpp
        include/mgard_api.h
	include/TensorMeshHierarchy.hpp
        ${MGARD_CUDA_HEADER}
	"${PROJECT_BINARY_DIR}/include/MGARDConfig.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
  DIRECTORY
    include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create executables under build/bin
include(CTest)
if(BUILD_TESTING)
#  enable_testing()
  add_subdirectory(tests)
endif()

# Add all targets to the build-tree export set
export(TARGETS mgard-library NAMESPACE mgard::
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

#Google Benchmark for benchmarks. See <https://github.com/google/benchmark>.
find_package(benchmark)

add_executable(mgard-benchmarks src/benchmarks.cpp)
target_link_libraries(mgard-benchmarks benchmark::benchmark benchmark::benchmark_main mgard-library)
set_target_properties(mgard-benchmarks PROPERTIES OUTPUT_NAME benchmarks)

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
message("        MOAB: ${MOAB_FOUND}")
