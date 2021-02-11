cmake_minimum_required(VERSION 3.19)

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

#Configurable Options
option(MGARD_ENABLE_CUDA "Enable Cuda support" OFF)
option(MGARD_ENABLE_CUDA_DEBUG "Enable Cuda support with debug" OFF)

# Some boilerplate to setup nice output directories
include(GNUInstallDirs)
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
find_package(ZSTD)
find_package(MOAB)

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
set(MGARD_MOAB_TEMPLATE_DEFINITION
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

# Set library
add_library(mgard-library STATIC
  src/mgard_compress.cpp  include/mgard_compress.hpp
  src/TensorMeshHierarchyIteration.cpp  include/TensorMeshHierarchyIteration.hpp include/TensorMeshHierarchyIteration.tpp
  ${MGARD_CUDA_SRC}   ${MGARD_CUDA_HEADER}
)
set_target_properties(mgard-library PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(mgard-library PROPERTIES OUTPUT_NAME mgard)

#target_compile_options(mgard-library PUBLIC -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	#target_compile_options(mgard-library PUBLIC -fsanitize=address -fsanitize=undefined)
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")
	# target_link_libraries(mgard-library PUBLIC -fsanitize=address -fsanitize=undefined)
else()
	#target_compile_options(mgard-library PUBLIC -march=native -ffast-math -fno-finite-math-only)
  execute_process(COMMAND "arch" OUTPUT_VARIABLE ARCH)
  string(REGEX REPLACE "\n$" "" ARCH "${ARCH}")
  if ("${ARCH}" STREQUAL "x86_64")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -ffast-math -fno-finite-math-only")
  elseif("${ARCH}" STREQUAL "ppc64le")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mcpu=native -mtune=native -ffast-math -fno-finite-math-only")
  endif()
endif()

target_include_directories(mgard-library
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mgard-library PUBLIC ZLIB::ZLIB)
if (MOAB_FOUND)
	target_sources(mgard-library PRIVATE ${MGARD_MOAB_HEADER} ${MGARD_MOAB_TEMPLATE_DEFINITION} ${MGARD_MOAB_SOURCE})
	target_link_libraries(mgard-library PUBLIC MOAB::MOAB)
endif()
if(ZSTD_FOUND)
	target_link_libraries(mgard-library PUBLIC ZSTD::ZSTD)
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
	target_link_libraries(mgard-library PUBLIC OpenMP::OpenMP_CXX)
endif()

option(DEFINE_MGARD_TIMING "Enable/disable MGARD timing" OFF)
if(DEFINE_MGARD_TIMING)
  add_definitions(-DMGARD_TIMING)
endif()

target_compile_features(mgard-library PUBLIC cxx_std_17)

set_target_properties(mgard-library PROPERTIES
  EXPORT_NAME mgard
  VERSION ${MGARD_VERSION}
  SOVERSION ${MGARD_VERSION_MAJOR}
  ${MGARD_CUDA_SEPARATE_COMPILE_COMPILATION_OPTION}
)

install(TARGETS mgard-library EXPORT mgard-targets)

install(FILES "${PROJECT_BINARY_DIR}/include/MGARDConfig.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard")

install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard")

# Create executables under build/bin
include(CTest)
if(BUILD_TESTING)
#  enable_testing()
  add_subdirectory(tests)
endif()

#It would be better if these took `MOAB_FOUND` into account.
set(MGARD_TESTS_HEADER
tests/include/testing_random.hpp
tests/include/testing_utilities.hpp
)
set(MGARD_TESTS_TEMPLATE_DEFINITION
tests/include/testing_random.tpp
tests/include/testing_utilities.tpp
)
set(MGARD_TESTS_SOURCE
tests/src/testing_utilities.cpp
tests/src/test_blas.cpp
tests/src/test_estimators.cpp
tests/src/test_indicators.cpp
tests/src/test_norms.cpp
tests/src/test_data.cpp
tests/src/test_pcg.cpp
tests/src/test_measure.cpp
tests/src/test_utilities.cpp
tests/src/test_MeshLevel.cpp
tests/src/test_MeshHierarchy.cpp
tests/src/test_MeshRefiner.cpp
tests/src/test_LinearOperator.cpp
tests/src/test_MassMatrix.cpp
tests/src/test_LinearQuantizer.cpp
tests/src/test_UniformMeshHierarchy.cpp
tests/src/test_UniformMeshRefiner.cpp
tests/src/test_UniformEdgeFamilies.cpp
tests/src/test_UniformRestriction.cpp
tests/src/test_EnumeratedMeshRange.cpp
tests/src/test_SituatedCoefficientRange.cpp
tests/src/test_IndicatorInput.cpp
tests/src/test_MultilevelCoefficientQuantizer.cpp
tests/src/test_shuffle.cpp
tests/src/test_TensorMeshHierarchy.cpp
tests/src/test_TensorMeshHierarchyIteration.cpp
tests/src/test_TensorLinearOperator.cpp
tests/src/test_TensorMassMatrix.cpp
tests/src/test_TensorProlongation.cpp
tests/src/test_TensorRestriction.cpp
tests/src/test_TensorMultilevelCoefficientQuantizer.cpp
tests/src/test_TensorNorms.cpp
tests/src/test_TensorQuantityOfInterest.cpp
tests/src/test_MultidimensionalArray.cpp
tests/src/test_mgard_api.cpp
tests/src/test_mgard.cpp
)

find_package(Catch2)

if(Catch2_FOUND)
	add_executable(mgard-tests
${MGARD_TESTS_HEADER}
${MGARD_TESTS_TEMPLATE_DEFINITION}
${MGARD_TESTS_SOURCE}
	)
	target_include_directories(mgard-tests PRIVATE tests/include)
	target_link_libraries(mgard-tests PRIVATE mgard-library Catch2::Catch2WithMain)
	set_target_properties(mgard-tests PROPERTIES OUTPUT_NAME tests)
else()
	message("-- Will not build tests (dependencies not met).")
endif()

# Add all targets to the build-tree export set
export(
	TARGETS mgard-library
	NAMESPACE mgard::
	FILE "${PROJECT_BINARY_DIR}/mgard-targets.cmake"
)

# Create the mgard-config.cmake and mgard-config-version files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/mgard-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)
configure_file(
	"${PROJECT_SOURCE_DIR}/cmake/mgard-config.cmake.in"
	"${PROJECT_BINARY_DIR}/mgard-config.cmake"
	@ONLY
)

configure_file(
	"${PROJECT_SOURCE_DIR}/include/MGARDConfig.h.in"
	"${PROJECT_BINARY_DIR}/include/MGARDConfig.h"
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
configure_file(
	"${PROJECT_SOURCE_DIR}/cmake/mgard.pc.in"
	"${PROJECT_BINARY_DIR}/mgard.pc"
	@ONLY
)
install(
  FILES "${PROJECT_BINARY_DIR}/mgard.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig" COMPONENT dev
)

#Google Benchmark for benchmarks. See <https://github.com/google/benchmark>.
find_package(benchmark)

if(benchmark_FOUND)
	add_executable(mgard-benchmarks src/benchmarks.cpp)
	target_link_libraries(mgard-benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main mgard-library)
	set_target_properties(mgard-benchmarks PROPERTIES OUTPUT_NAME benchmarks)
else()
	message("-- Will not build benchmarks (dependencies not met).")
endif()

find_package(yaml-cpp)
find_package(archive)
find_package(TCLAP)

if (yaml-cpp_FOUND AND archive_FOUND AND TCLAP_FOUND)
	add_library(
		mgard-cli STATIC
		src/subcommand_arguments.cpp include/subcommand_arguments.hpp
		src/metadata.cpp include/metadata.hpp
	)
	target_include_directories(mgard-cli PUBLIC include/)
	target_link_libraries(mgard-cli PUBLIC TCLAP::TCLAP)

	add_executable(mgard-executable src/executable.cpp)
	target_link_libraries(mgard-executable PRIVATE mgard-library mgard-cli yaml-cpp::yaml-cpp archive::archive TCLAP::TCLAP)
	set_target_properties(mgard-executable PROPERTIES OUTPUT_NAME mgard)
else()
	message("-- Will not build executable (dependencies not met).")
endif()

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
message("      OpenMP: ${OpenMP_FOUND}")
message("        CUDA: ${MGARD_ENABLE_CUDA}")
message("  CUDA-Debug: ${MGARD_ENABLE_CUDA_DEBUG}")
message("        MOAB: ${MOAB_FOUND}")
message("      Catch2: ${Catch2_FOUND}")
message("    yaml-cpp: ${yaml-cpp_FOUND}")
message("     archive: ${archive_FOUND}")
message("       TCLAP: ${TCLAP_FOUND}")
message("   benchmark: ${benchmark_FOUND}")
