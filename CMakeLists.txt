cmake_minimum_required(VERSION 3.19)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
	message(
		FATAL_ERROR
		"In-source builds are not supported. Please create a build directory "
		"separate from the source directory."
	)
endif()

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(MGARD_VERSION_MAJOR "1")
set(MGARD_VERSION_MINOR "2")
set(MGARD_VERSION_PATCH "0")

set(MGARD_FILE_VERSION_MAJOR "1")
set(MGARD_FILE_VERSION_MINOR "0")
set(MGARD_FILE_VERSION_PATCH "0")

project(
  MGARD
  HOMEPAGE_URL "https://github.com/CODARcode/MGARD"
  VERSION "${MGARD_VERSION_MAJOR}.${MGARD_VERSION_MINOR}.${MGARD_VERSION_PATCH}"
  #CUDA will be enabled below if `MGARD_ENABLE_CUDA` is `ON`.
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Configurable Options
option(MGARD_ENABLE_BENCHMARKS "Build benchmarks." OFF)

option(MGARD_ENABLE_DOCS "Build documentation." OFF)

option(MGARD_ENABLE_OPENMP "Enable OpenMP support." OFF)
option(MGARD_ENABLE_CUDA "Enable CUDA support" OFF)
option(MGARD_ENABLE_SERIAL "Enable SERIAL support" OFF)
option(MGARD_ENABLE_HIP "Enable HIP support" OFF)
option(MGARD_ENABLE_KOKKOS "Enable Kokkos support" OFF)

#For performance optimization
option(MGARD_ENABLE_CUDA_FMA "Enable CUDA support with fused multiply–add instruction." OFF)
option(MGARD_ENABLE_CUDA_OPTIMIZE_VOLTA "Optimize for Volta GPUs." OFF)
option(MGARD_ENABLE_CUDA_OPTIMIZE_TURING "Optimize for Turing GPUs." OFF)

option(MGARD_ENABLE_CLI "Build executable." OFF)

include(CTest)
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

# Default to a release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

if (MGARD_ENABLE_SERIAL OR 
    MGARD_ENABLE_CUDA   OR
    MGARD_ENABLE_HIP    OR
    MGARD_ENABLE_KOKKOS)
  add_subdirectory (src/mgard-x)
  add_subdirectory (include/mgard-x)
  add_subdirectory (include/mgard-x/MDR)
  find_package(zstd REQUIRED)
  add_executable(mgard-x ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard-x/Executables/mgard-x.cpp)
  target_link_libraries(mgard-x mgard-library  ${CMAKE_DL_LIBS})
  target_include_directories(mgard-x PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

  add_executable(mgard-x-autotuner ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard-x/Executables/mgard-x-autotuner.cpp)
  target_link_libraries(mgard-x-autotuner mgard-library  ${CMAKE_DL_LIBS})
  target_include_directories(mgard-x-autotuner PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

  add_executable(MgardSerialExec ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard-x/Executables/MgardSerialExec.cpp)
  target_link_libraries(MgardSerialExec mgard-library  ${CMAKE_DL_LIBS})
  target_include_directories(MgardSerialExec PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
endif()

if(MGARD_ENABLE_SERIAL)
	set (CMAKE_CXX_STANDARD 14)
  set (CMAKE_CXX_STANDARD_REQUIRED ON)
  set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_ENABLE_SERIAL -w")
endif()

if(MGARD_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # Mininum common non-deprecated architecture
      set(CMAKE_CUDA_ARCHITECTURES 52)
  endif()

  find_package(nvcomp REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  set (CMAKE_CXX_STANDARD 14)
  set (CMAKE_CXX_STANDARD_REQUIRED ON)

  # Legacy CUDA version
  add_subdirectory (src/cuda)
  add_subdirectory (include/cuda)
  option(MGARD_ENABLE_CUDA_FMA "Enable CUDA support with fused multiply–add instruction" OFF)
  add_executable(MgardCudaExec ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MgardCudaExec.cpp)
    target_link_libraries(MgardCudaExec mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(MgardCudaExec PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

  set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_ENABLE_CUDA")
  set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMGARD_ENABLE_CUDA")
  set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")
  
  if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  endif()

  set (MGARD_X_SEPARATE_COMPILE_COMPILATION_OPTION 
       CUDA_SEPARABLE_COMPILATION ON)

  if(MGARD_ENABLE_CUDA_FMA)
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_X_FMA")
    set (CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_X_FMA")
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_FMA")
    set (CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_FMA")
  endif()
endif()


if (MGARD_ENABLE_HIP) 
  find_package(HIP REQUIRED)
  enable_language(HIP)
  set (CMAKE_CXX_STANDARD 14)
  set (CMAKE_CXX_STANDARD_REQUIRED ON)
  set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_ENABLE_HIP -w")
endif()

if (MGARD_ENABLE_KOKKOS)
  find_package(Kokkos REQUIRED)
  set (MGARD_X_SEPARATE_COMPILE_COMPILATION_OPTION 
         CUDA_SEPARABLE_COMPILATION ON)
endif()

find_package(ZLIB REQUIRED)

find_package(PkgConfig)

# pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)

find_package(zstd)

find_package(MOAB)
#We run into this when building MOAB 5.3.0 with `./configure && make`.
if(MOAB_FOUND AND NOT TARGET MOAB)
	message(STATUS "MOAB found, but configuration file ")
	list(APPEND CMAKE_MESSAGE_INDENT "    ")
	message(STATUS "${MOAB_CONFIG}")
	list(POP_BACK CMAKE_MESSAGE_INDENT)
	message(STATUS "did not define a library target.")
	set(MOAB_FOUND FALSE)
endif()

set(
  MGARD_MOAB_HPP
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
set(
  MGARD_MOAB_TPP
  include/blas.tpp
  include/utilities.tpp
  include/data.tpp
  include/UniformEdgeFamilies.tpp
  include/LinearQuantizer.tpp
  include/SituatedCoefficientRange.tpp
  include/MultilevelCoefficientQuantizer.tpp
)
set(
  MGARD_MOAB_CPP
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
set(
	MGARD_LIBRARY_CPP
        src/compress.cpp
        src/compress_internal.cpp
	src/compressors.cpp
	src/format.cpp
)

set(MAXIMUM_DIMENSION 16 CACHE STRING "Maximum supported dimension for self-describing decompression.")

set(PYTHON3_NAMES "python3" "python")
find_program(PYTHON3 NAMES ${PYTHON3_NAMES} REQUIRED)
if(NOT PYTHON3 STREQUAL "PYTHON3-NOTFOUND")
	execute_process(
		COMMAND "${PYTHON3}" "--version"
		OUTPUT_VARIABLE PYTHON3_VERSION_OUTPUT
		ERROR_VARIABLE PYTHON3_VERSION_OUTPUT
	)
	string(
		REGEX REPLACE
		"^Python ([0-9]+)\.([0-9]+)\.([0-9]+)\n$"
		"\\1"
		PYTHON3_VERSION_MAJOR
		"${PYTHON3_VERSION_OUTPUT}"
	)
	if(PYTHON3_VERSION_MAJOR LESS 3)
		#Mimic the error message we'd have gotten had `find_program` failed.
		list(JOIN PYTHON3_NAMES ", " PYTHON3_NAMES_JOINED)
		message(FATAL_ERROR "Could not find PYTHON3 ('python>=3') using the following names: ${PYTHON3_NAMES_JOINED}")
	endif()
endif()

set(MGARD_FILE_FORMAT_PROTO "src/mgard.proto")
set(MGARD_FILE_FORMAT_INCLUDEDIR "${CMAKE_CURRENT_BINARY_DIR}/include")
set(MGARD_FILE_FORMAT_SRCDIR "${CMAKE_CURRENT_BINARY_DIR}/src")
#This is hardcoded in a bunch of source files.
set(MGARD_FILE_FORMAT_INCLUDESUBDIR_COMPONENT "proto")
set(MGARD_FILE_FORMAT_DIR "${MGARD_FILE_FORMAT_INCLUDEDIR}/${MGARD_FILE_FORMAT_INCLUDESUBDIR_COMPONENT}")
set(MGARD_FILE_FORMAT_HPP "${MGARD_FILE_FORMAT_DIR}/mgard.pb.h")
set(MGARD_FILE_FORMAT_CPP "${MGARD_FILE_FORMAT_DIR}/mgard.pb.cc")
set(MGARD_GENERATE_DIMENSION_SWITCH "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_dimension_switch.py")
set(MGARD_COMPRESS_INTERNAL_CPP "${MGARD_FILE_FORMAT_SRCDIR}/compress_internal_generated.cpp")

pkg_search_module(protobuf REQUIRED IMPORTED_TARGET GLOBAL protobuf>=3.0.0)
find_program(PROTOC NAMES "protoc" REQUIRED)

file(MAKE_DIRECTORY "${MGARD_FILE_FORMAT_DIR}")
file(MAKE_DIRECTORY "${MGARD_FILE_FORMAT_SRCDIR}")
get_filename_component(PROTO_PATH "${MGARD_FILE_FORMAT_PROTO}" DIRECTORY)
add_custom_command(
	OUTPUT
	"${MGARD_FILE_FORMAT_HPP}"
	"${MGARD_FILE_FORMAT_CPP}"
	COMMAND "${PROTOC}"
	ARGS
	"--cpp_out=${MGARD_FILE_FORMAT_DIR}"
	"--proto_path=${PROTO_PATH}"
	"${PROTO_PATH_OPTION}"
	"${MGARD_FILE_FORMAT_PROTO}"
	DEPENDS "${MGARD_FILE_FORMAT_PROTO}"
	WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_command(
	OUTPUT
	"${MGARD_COMPRESS_INTERNAL_CPP}"
	COMMAND "${PYTHON3}"
	ARGS
	"${MGARD_GENERATE_DIMENSION_SWITCH}"
	"--decompress"
	"${MAXIMUM_DIMENSION}"
	"${MGARD_COMPRESS_INTERNAL_CPP}"
)

list(APPEND MGARD_LIBRARY_CPP "${MGARD_FILE_FORMAT_CPP}" "${MGARD_COMPRESS_INTERNAL_CPP}")

add_library(
  mgard-library
  ${MGARD_LIBRARY_CPP}
  ${MGARD_CUDA_SRC}
  ${MGARD_X_SRC}
)
set_target_properties(mgard-library PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(mgard-library PROPERTIES OUTPUT_NAME mgard)

target_link_libraries(mgard-library PUBLIC PkgConfig::protobuf)
target_compile_definitions(mgard-library PUBLIC MGARD_PROTOBUF)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(SANITIZER_DEFAULT ON)
else()
  set(SANITIZER_DEFAULT OFF)
endif()
option(MGARD_ENABLE_ASAN "Compile with address sanitizer support." ${SANITIZER_DEFAULT})
option(MGARD_ENABLE_UBSAN "Compile with undefined behavior sanitizer support." ${SANITIZER_DEFAULT})
if(MGARD_ENABLE_UBSAN)
  target_compile_options(mgard-library PUBLIC "-fsanitize=undefined")
  target_link_libraries(mgard-library PUBLIC "-fsanitize=undefined")
endif()
if(MGARD_ENABLE_ASAN)
  target_compile_options(mgard-library PUBLIC "-fsanitize=address")
  target_link_libraries(mgard-library PUBLIC "-fsanitize=address")
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  execute_process(COMMAND "arch" OUTPUT_VARIABLE ARCH)
  string(REGEX REPLACE "\n$" "" ARCH "${ARCH}")
  if("${ARCH}" STREQUAL "x86_64")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -ffast-math -fno-finite-math-only")
  elseif("${ARCH}" STREQUAL "ppc64le")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mcpu=native -mtune=native -ffast-math -fno-finite-math-only")
  endif()
endif()

target_include_directories(mgard-library
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mgard-library PUBLIC ZLIB::ZLIB)
if(MOAB_FOUND)
	add_compile_definitions(MGARD_MOAB)
	target_sources(mgard-library PRIVATE ${MGARD_MOAB_CPP})
	target_link_libraries(mgard-library PUBLIC MOAB)
endif()

if(zstd_FOUND)
  target_compile_definitions(mgard-library PUBLIC MGARD_ZSTD)
  target_link_libraries(mgard-library PUBLIC zstd::libzstd_shared)
  # target_link_libraries(mgard-library PUBLIC PkgConfig::ZSTD)
endif()

if (MGARD_ENABLE_SERIAL OR 
    MGARD_ENABLE_CUDA   OR
    MGARD_ENABLE_HIP    OR
    MGARD_ENABLE_KOKKOS)
  target_link_libraries(mgard-library PUBLIC zstd::libzstd_shared)
endif()

if(MGARD_ENABLE_CUDA)
  target_link_libraries(mgard-library PUBLIC nvcomp CUDA::cudart)
  # target_include_directories(mgard-library PUBLIC ${NVCOMP_INCLUDE_DIR})
endif()

if(MGARD_ENABLE_OPENMP)
	find_package(OpenMP)
	if(OpenMP_FOUND)
		target_link_libraries(mgard-library PUBLIC OpenMP::OpenMP_CXX)
	endif()
endif()

if(MGARD_ENABLE_KOKKOS)
  target_link_libraries(mgard-library PUBLIC Kokkos::kokkoscore)
endif()

option(MGARD_ENABLE_TIMING "Enable MGARD timing." OFF)
if(MGARD_ENABLE_TIMING)
  add_definitions(-DMGARD_TIMING)
endif()

target_compile_features(mgard-library PUBLIC cxx_std_17)

set_target_properties(mgard-library PROPERTIES
  EXPORT_NAME mgard
  VERSION ${MGARD_VERSION}
  SOVERSION ${MGARD_VERSION_MAJOR}
  ${MGARD_X_SEPARATE_COMPILE_COMPILATION_OPTION}
)

install(TARGETS mgard-library EXPORT mgard-targets)

install(FILES "${PROJECT_BINARY_DIR}/include/MGARDConfig.hpp" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard")
install(DIRECTORY "include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard")
install(FILES "${MGARD_FILE_FORMAT_HPP}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard/${MGARD_FILE_FORMAT_INCLUDESUBDIR_COMPONENT}")

set(
  MGARD_TESTS_HPP
  tests/include/testing_random.hpp
  tests/include/testing_utilities.hpp
)
set(
  MGARD_TESTS_TPP
  tests/include/testing_random.tpp
  tests/include/testing_utilities.tpp
)
set(
	MGARD_TESTS_CPP
	tests/src/testing_utilities.cpp
	tests/src/test_utilities.cpp
	tests/src/test_blas.cpp
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
	tests/src/test_compress.cpp
	tests/src/test_decompose.cpp
	tests/src/test_format.cpp
	tests/src/test_quantize.cpp
	tests/src/test_compressors.cpp
	tests/src/test_CompressedDataset.cpp
)
if(MOAB_FOUND)
  list(
    APPEND
    MGARD_TESTS_CPP
    tests/src/test_estimators.cpp
    tests/src/test_indicators.cpp
    tests/src/test_norms.cpp
    tests/src/test_data.cpp
    tests/src/test_pcg.cpp
    tests/src/test_measure.cpp
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
  )
endif()

set(MGARD_CLI_SRCDIR "${CMAKE_CURRENT_BINARY_DIR}/src")
set(MGARD_CLI_INTERNAL_CPP "${MGARD_CLI_SRCDIR}/cli_internal_generated.cpp")

set(
	MGARD_EXECUTABLE_HPP
	include/arguments.hpp
	include/cmdline.hpp
	include/output.hpp
	include/cli_internal.hpp
)
set(
	MGARD_EXECUTABLE_TPP
	include/arguments.tpp
	include/cli_internal.tpp
)
set(
	MGARD_EXECUTABLE_CPP
	src/arguments.cpp
	src/cmdline.cpp
	src/executable.cpp
	src/output.cpp
	src/cli_internal.cpp
	"${MGARD_CLI_INTERNAL_CPP}"
)

set(
  MGARD_BENCHMARK_CPP
  src/benchmarks.cpp
)

set(
	MGARD_DOC_PAGES
	README.md
	doc/MGARD-GPU.md
	examples/README.md
	examples/linking/README.md
	examples/compression/README.md
	examples/decompression/README.md
	examples/mgard-x/SimpleExample/README.md
)

if(MGARD_ENABLE_DOCS)
	find_program(DOXYGEN NAMES "doxygen")
	find_program(SED NAMES "sed")

	if(SED STREQUAL "SED-NOTFOUND")
		message(STATUS "Will not build documentation (sed not found).")
	elseif(PYTHON3 STREQUAL "PYTHON3-NOTFOUND")
		message(STATUS "Will not build documentation (Python 3 not found).")
	elseif(DOXYGEN STREQUAL "DOXYGEN-NOTFOUND")
		message(STATUS "Will not build documentation (Doxygen not found).")
	else()
		configure_file("scripts/modify_documentation_links.py.in" "scripts/modify_documentation_links.py")
		foreach(MGARD_DOC_PAGE ${MGARD_DOC_PAGES})
			set(MGARD_DOC_PAGE_FIXED "${CMAKE_CURRENT_BINARY_DIR}/${MGARD_DOC_PAGE}")
			add_custom_command(
				OUTPUT "${MGARD_DOC_PAGE_FIXED}"
				#Fix the relative links.
				COMMAND "${PYTHON3}"
				ARGS
				"${CMAKE_CURRENT_BINARY_DIR}/scripts/modify_documentation_links.py"
				"${MGARD_DOC_PAGE}"
				"${MGARD_DOC_PAGE_FIXED}"
				COMMAND "${SED}"
				ARGS
				#Get rid of the workflow status badges.
				"--expression" "'s/\\s*\\[!\\[[^]]\\+\\]\\[[^]]\\+\\]\\]\\[[^]]\\+\\]//g'"
				#Stop Doxygen from interpreting something like 'Huffman@GPU' – but not something like ' @ref' (added by the Python script above) – as a command.
				"--expression" "'s/\\([[:alnum:]]\\)@\\([[:alnum:]]\\)/\\1\\\\@\\2/g'"
				"--in-place"
				"${MGARD_DOC_PAGE_FIXED}"
				WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
				DEPENDS "${MGARD_DOC_PAGE}"
			)
		endforeach()
		list(TRANSFORM MGARD_DOC_PAGES PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")
		list(GET MGARD_DOC_PAGES 0 MGARD_DOC_MAINPAGE)
		list(JOIN MGARD_DOC_PAGES " " MGARD_DOC_PAGES_JOINED)
		list(JOIN MGARD_EXECUTABLE_HPP " " MGARD_EXECUTABLE_HPP_JOINED)
		configure_file(".doxygen.in" ".doxygen")
		add_custom_target(
			mgard-documentation
			ALL
			COMMAND "${DOXYGEN}" "${CMAKE_CURRENT_BINARY_DIR}/.doxygen"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			DEPENDS "${MGARD_DOC_PAGES}"
		)
		install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doxygen/html/" TYPE DOC)
	endif()
endif()

if(BUILD_TESTING)
	find_package(Catch2 3.0.0)

	if(Catch2_FOUND)
		configure_file("tests/include/testing_paths.hpp.in" "include/testing_paths.hpp")
		add_executable(mgard-tests ${MGARD_TESTS_CPP})
		target_include_directories(mgard-tests PRIVATE "tests/include" "${CMAKE_CURRENT_BINARY_DIR}/include")
		target_link_libraries(mgard-tests PRIVATE mgard-library Catch2::Catch2WithMain)
		set_target_properties(mgard-tests PROPERTIES OUTPUT_NAME tests)
	else()
		message(STATUS "Will not build tests (Catch2 not found).")
	endif()
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
  "${PROJECT_SOURCE_DIR}/include/MGARDConfig.hpp.in"
  "${PROJECT_BINARY_DIR}/include/MGARDConfig.hpp"
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

if(MGARD_ENABLE_BENCHMARKS)
	find_package(benchmark)

	if(benchmark_FOUND)
		add_executable(mgard-benchmarks ${MGARD_BENCHMARK_CPP})
		target_link_libraries(mgard-benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main mgard-library)
		set_target_properties(mgard-benchmarks PROPERTIES OUTPUT_NAME benchmarks)
	else()
		message(STATUS "Will not build benchmarks (Google Benchmark not found).")
	endif()
endif()


if(MGARD_ENABLE_CLI)
	pkg_search_module(TCLAP IMPORTED_TARGET GLOBAL tclap>=1.4)
	if(NOT TCLAP_FOUND)
		find_package(TCLAP 1.4)
		if(TCLAP_FOUND)
			#Mimic target defined by `pkg_search_module`.
			add_library(PkgConfig::TCLAP ALIAS TCLAP::TCLAP)
		endif()
	endif()

	if(NOT TCLAP_FOUND)
		message(STATUS "Will not build executable (TCLAP not found).")
	else()
		file(MAKE_DIRECTORY "${MGARD_CLI_SRCDIR}")
		add_custom_command(
			OUTPUT
			"${MGARD_CLI_INTERNAL_CPP}"
			COMMAND "${PYTHON3}"
			ARGS
			"${MGARD_GENERATE_DIMENSION_SWITCH}"
			"--compress"
			"${MAXIMUM_DIMENSION}"
			"${MGARD_CLI_INTERNAL_CPP}"
		)

		add_executable(mgard-executable ${MGARD_EXECUTABLE_CPP})
		target_include_directories(mgard-executable PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")
		target_link_libraries(mgard-executable PRIVATE mgard-library PkgConfig::TCLAP)
		set_target_properties(mgard-executable PROPERTIES OUTPUT_NAME mgard)

		install(TARGETS mgard-executable RUNTIME)
		install(FILES man/man1/mgard.1 DESTINATION "${CMAKE_INSTALL_MANDIR}/man1")
	endif()
endif()

message(STATUS "")
message(STATUS "Configuration summary:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

message(STATUS "Installation directories:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(STATUS "prefix:  ${CMAKE_INSTALL_PREFIX}")
message(STATUS "bin:     ${CMAKE_INSTALL_FULL_BINDIR}")
message(STATUS "lib:     ${CMAKE_INSTALL_FULL_LIBDIR}")
message(STATUS "include: ${CMAKE_INSTALL_FULL_INCLUDEDIR}")
message(STATUS "cmake:   ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_CMAKEDIR}")
list(POP_BACK CMAKE_MESSAGE_INDENT)

message(STATUS "")
message(STATUS "Options:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(STATUS "build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "shared libs:   ${BUILD_SHARED_LIBS}")
message(STATUS "MGARD-X portable backends:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(STATUS "SERIAL:      ${MGARD_ENABLE_SERIAL}")
message(STATUS "CUDA:        ${MGARD_ENABLE_CUDA}")
if (MGARD_ENABLE_CUDA)
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  message(STATUS "CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()
message(STATUS "HIP:         ${MGARD_ENABLE_HIP}")
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "timing:        ${MGARD_ENABLE_TIMING}")
message(STATUS "ASan:          ${MGARD_ENABLE_ASAN}")
message(STATUS "UBSan:         ${MGARD_ENABLE_UBSAN}")
message(STATUS "tests:         ${BUILD_TESTING}")
message(STATUS "benchmarks:    ${MGARD_ENABLE_BENCHMARKS}")
message(STATUS "documentation: ${MGARD_ENABLE_DOCS}")
message(STATUS "CLI:           ${MGARD_ENABLE_CLI}")
list(POP_BACK CMAKE_MESSAGE_INDENT)

message(STATUS "")
message(STATUS "Libraries:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(STATUS "MOAB:      ${MOAB_FOUND}")
message(STATUS "protobuf:  ${protobuf_FOUND}")
if(MGARD_ENABLE_OPENMP)
	message(STATUS "OpenMP:    ${OpenMP_FOUND}")
endif()
if(BUILD_TESTING)
	message(STATUS "Catch2:    ${Catch2_FOUND}")
endif()
if(MGARD_ENABLE_BENCHMARKS)
	message(STATUS "benchmark: ${benchmark_FOUND}")
endif()
if(MGARD_ENABLE_CLI)
	message(STATUS "TCLAP:     ${TCLAP_FOUND}")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)

list(POP_BACK CMAKE_MESSAGE_INDENT)
