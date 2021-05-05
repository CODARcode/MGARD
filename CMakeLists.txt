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

macro (mgard_option variable)
  if (NOT DEFINED "${variable}")
    option("${variable}" ${ARGN})
  endif ()
endmacro ()

#Configurable Options
mgard_option(MGARD_ENABLE_CUDA "Enable Cuda support" OFF)

#For performance optimization
mgard_option(MGARD_ENABLE_CUDA_FMA "Enable Cuda support with fused multpily-add instruction" OFF)
mgard_option(MGARD_ENABLE_CUDA_OPTIMIZE_VOLTA "Optimized for Volta GPUs" OFF)
mgard_option(MGARD_ENABLE_CUDA_OPTIMIZE_TURING "Optimized for Turing GPUs" OFF)

#For debug only
mgard_option(MGARD_ENABLE_CUDA_DEBUG "Enable Cuda support with debug" OFF )
mgard_option(MGARD_ENABLE_CUDA_KERNEL_PROFILE "Profile kernel configuration" OFF)

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
  find_package(NVCOMP)
  find_package(CUDAToolkit)
  set (CMAKE_CXX_STANDARD 14)
  set(MGARD_CUDA_SRC
      src/cuda/Array.cu
      src/cuda/Message.cpp
      src/cuda/CompressionWorkflow.cu
      src/cuda/Handle.cu
      src/cuda/MemoryManagement.cu                 
      src/cuda/CommonInternal.cu    
      src/cuda/PrecomputeKernels.cu  
      src/cuda/DataRefactoring.cu   
      src/cuda/GridProcessingKernels/InterpCalcCoeff.cu
      src/cuda/GridProcessingKernels/InterpCalcCoeff_3D.cu
      src/cuda/GridProcessingKernels/Interpolation.cu
      src/cuda/GridProcessingKernels/CalcCoefficients.cu
      src/cuda/GridProcessingKernels/Reorder.cu
      src/cuda/GridProcessingKernels/InterpCoeffRestore.cu
      src/cuda/GridProcessingKernels/InterpCoeffRestore_3D.cu
      src/cuda/GridProcessingKernels/InterpolationRev.cu
      src/cuda/GridProcessingKernels/CoefficientRestore.cu
      src/cuda/GridProcessingKernels/ReorderRestore.cu
      src/cuda/LinearProcessingKernels/MassTrans1.cu
      src/cuda/LinearProcessingKernels/MassTrans2.cu
      src/cuda/LinearProcessingKernels/MassTrans3.cu
      src/cuda/LinearProcessingKernels/MassTrans1_3D.cu
      src/cuda/LinearProcessingKernels/MassTrans2_3D.cu
      src/cuda/LinearProcessingKernels/MassTrans3_3D.cu
      src/cuda/IterativeProcessingKernels/SolveTridiag1.cu
      src/cuda/IterativeProcessingKernels/SolveTridiag2.cu
      src/cuda/IterativeProcessingKernels/SolveTridiag3.cu
      src/cuda/IterativeProcessingKernels/SolveTridiag1_3D.cu
      src/cuda/IterativeProcessingKernels/SolveTridiag2_3D.cu
      src/cuda/IterativeProcessingKernels/SolveTridiag3_3D.cu
      src/cuda/LevelwiseProcessingKernels/CopyLevel.cu
      src/cuda/LevelwiseProcessingKernels/AddLevel.cu
      src/cuda/LevelwiseProcessingKernels/SubtractLevel.cu
      src/cuda/LinearQuantization/LevelwiseLinearQuantization.cu
      src/cuda/LinearQuantization/LevelwiseLinearDequantization.cu
      src/cuda/LosslessCompression.cu
      src/cuda/ParallelHuffman/canonical.cu
      src/cuda/ParallelHuffman/constants.cc
      src/cuda/ParallelHuffman/cuda_mem.cu
      src/cuda/ParallelHuffman/format.cc
      src/cuda/ParallelHuffman/histogram.cu
      src/cuda/ParallelHuffman/huffman_codec.cu
      src/cuda/ParallelHuffman/huffman_workflow.cu
      src/cuda/ParallelHuffman/huffman.cu
      src/cuda/ParallelHuffman/par_huffman.cu
      src/cuda/ParallelHuffman/par_huffman_sortbyfreq.cu
      src/cuda/ParallelHuffman/par_merge.cu
      src/cuda/ParallelHuffman/types.cc)

  set(MGARD_CUDA_HEADER
      include/mgard_cuda_api.h
      include/cuda/Array.h
      include/cuda/Message.h
      include/cuda/Handle.h
      include/cuda/MemoryManagement.h                 
      include/cuda/Common.h
      include/cuda/CommonInternal.h   
      include/cuda/PrecomputeKernels.h   
      include/cuda/DataRefactoring.h   
      include/cuda/GridProcessingKernel.hpp   
      include/cuda/GridProcessingKernel.h
      include/cuda/GridProcessingKernel3D.hpp   
      include/cuda/GridProcessingKernel3D.h
      include/cuda/LinearProcessingKernel.hpp   
      include/cuda/LinearProcessingKernel.h
      include/cuda/LinearProcessingKernel3D.hpp   
      include/cuda/LinearProcessingKernel3D.h
      include/cuda/IterativeProcessingKernel.hpp   
      include/cuda/IterativeProcessingKernel.h
      include/cuda/IterativeProcessingKernel3D.hpp   
      include/cuda/IterativeProcessingKernel3D.h
      include/cuda/LevelwiseProcessingKernel.hpp   
      include/cuda/LevelwiseProcessingKernel.h
      include/cuda/LinearQuantization.hpp
      include/cuda/LinearQuantization.h
      include/cuda/LosslessCompression.h
      include/cuda/ParallelHuffman/canonical.cuh
      include/cuda/ParallelHuffman/constants.hh
      include/cuda/ParallelHuffman/cuda_error_handling.cuh
      include/cuda/ParallelHuffman/cuda_mem.cuh
      include/cuda/ParallelHuffman/dbg_gpu_printing.cuh
      include/cuda/ParallelHuffman/format.hh
      include/cuda/ParallelHuffman/histogram.cuh
      include/cuda/ParallelHuffman/huffman_codec.cuh
      include/cuda/ParallelHuffman/huffman_workflow.cuh
      include/cuda/ParallelHuffman/io.hh
      include/cuda/ParallelHuffman/par_huffman.cuh
      include/cuda/ParallelHuffman/par_merge.cuh
      include/cuda/ParallelHuffman/timer.hh
      include/cuda/ParallelHuffman/types.hh)

  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA")
  SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA")
  set (MGARD_CUDA_SEPARATE_COMPILE_COMPILATION_OPTION 
       CUDA_SEPARABLE_COMPILATION ON)

  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")

  if(MGARD_ENABLE_CUDA_DEBUG)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_DEBUG")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_DEBUG")
  endif()
  if(MGARD_ENABLE_CUDA_FMA)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_FMA")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_FMA")
  endif()
  if(MGARD_ENABLE_CUDA_KERNEL_PROFILE)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_KERNEL_PROFILE")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_KERNEL_PROFILE")
  endif()
  if(MGARD_ENABLE_CUDA_OPTIMIZE_TURING)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_OPTIMIZE_TURING")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_OPTIMIZE_TURING")
    set(CUDA_ARCH_STRING "${CUDA_ARCH_STRING};70;72;75")
  endif()
  if(MGARD_ENABLE_CUDA_OPTIMIZE_VOLTA)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_OPTIMIZE_VOLTA")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_OPTIMIZE_VOLTA")
    set(CUDA_ARCH_STRING "${CUDA_ARCH_STRING};60;61;62")
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

if(MGARD_ENABLE_CUDA)
  target_link_libraries(mgard-library PUBLIC NVCOMP::NVCOMP CUDA::cudart ${CUDAToolkit_LIBRARY_DIR}/libcudadevrt.a)
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
tests/src/test_mgard_api.cpp
tests/src/test_mgard.cpp
)

find_package(Catch2)

if(Catch2_FOUND)
	configure_file("tests/include/testing_paths.hpp.in" "include/testing_paths.hpp")

	add_executable(mgard-tests
${MGARD_TESTS_HEADER}
${MGARD_TESTS_TEMPLATE_DEFINITION}
${MGARD_TESTS_SOURCE}
	)
	target_include_directories(mgard-tests PRIVATE tests/include "${CMAKE_CURRENT_BINARY_DIR}/include")
	target_link_libraries(mgard-tests PRIVATE mgard-library Catch2::Catch2WithMain)
	set_target_properties(mgard-tests PROPERTIES OUTPUT_NAME tests)
else()
	message("-- Will not build tests (dependencies not met).")
endif()

if(MGARD_ENABLE_CUDA)
    add_executable(TestDoubleRelativeError ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/TestDoubleRelativeError.cpp)
    target_link_libraries(TestDoubleRelativeError mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(TestDoubleRelativeError PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    add_executable(TestFloatRelativeError ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/TestFloatRelativeError.cpp)
    target_link_libraries(TestFloatRelativeError mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(TestFloatRelativeError PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    add_executable(TestDoubleAbsoluteError ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/TestDoubleAbsoluteError.cpp)
    target_link_libraries(TestDoubleAbsoluteError mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(TestDoubleAbsoluteError PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    add_executable(TestFloatAbsoluteError ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/TestFloatAbsoluteError.cpp)
    target_link_libraries(TestFloatAbsoluteError mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(TestFloatAbsoluteError PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
endif()

if(MGARD_ENABLE_CUDA)
  cmake_policy(SET CMP0104 NEW)
  cmake_policy(SET CMP0105 NEW)
  set_property(TARGET mgard-library PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_STRING})
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
	add_executable(
		mgard-executable
		src/executable.cpp
		src/subcommand_arguments.cpp
		include/subcommand_arguments.hpp
		src/metadata.cpp
		include/metadata.hpp
	)
	target_include_directories(mgard-executable PRIVATE include/)
	target_link_libraries(mgard-executable PRIVATE mgard-library yaml-cpp::yaml-cpp archive::archive TCLAP::TCLAP)
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
message("    CUDA-FMA: ${MGARD_ENABLE_CUDA_FMA}")
message("   CUDA-PROF: ${MGARD_ENABLE_CUDA_KERNEL_PROFILE}")
message("  CUDA-VOLTA: ${MGARD_ENABLE_CUDA_OPTIMIZE_VOLTA}")
message(" CUDA-TURING: ${MGARD_ENABLE_CUDA_OPTIMIZE_TURING}")
message("        MOAB: ${MOAB_FOUND}")
message("      Catch2: ${Catch2_FOUND}")
message("    yaml-cpp: ${yaml-cpp_FOUND}")
message("     archive: ${archive_FOUND}")
message("       TCLAP: ${TCLAP_FOUND}")
message("   benchmark: ${benchmark_FOUND}")
