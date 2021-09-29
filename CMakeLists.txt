cmake_minimum_required(VERSION 3.19)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported.  Please create a build directory "
    "separate from the source directory")
endif()

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(MGARD_VERSION_MAJOR "0")
set(MGARD_VERSION_MINOR "1")
set(MGARD_VERSION_PATCH "0")

set(MGARD_FILE_VERSION_MAJOR "0")
set(MGARD_FILE_VERSION_MINOR "0")
set(MGARD_FILE_VERSION_PATCH "0")

project(
	MGARD
	HOMEPAGE_URL "https://github.com/CODARcode/MGARD"
	VERSION "${MGARD_VERSION_MAJOR}.${MGARD_VERSION_MINOR}.${MGARD_VERSION_PATCH}"
	#CUDA will be enabled below if `MGARD_ENABLE_CUDA` is `ON`.
	LANGUAGES CXX
)

#Configurable Options
option(MGARD_ENABLE_CUDA "Enable CUDA support" OFF)

#For performance optimization
option(MGARD_ENABLE_CUDA_FMA "Enable CUDA support with fused multiply–add instruction" OFF)
option(MGARD_ENABLE_CUDA_OPTIMIZE_VOLTA "Optimized for Volta GPUs" OFF)
option(MGARD_ENABLE_CUDA_OPTIMIZE_TURING "Optimized for Turing GPUs" OFF)

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

if(MGARD_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(NVCOMP)
  find_package(CUDAToolkit)
  set (CMAKE_CXX_STANDARD 14)
  set (CMAKE_CXX_STANDARD_REQUIRED ON)
  add_subdirectory (src/cuda)
  add_subdirectory (include/cuda)

  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA")
  SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA")
  set (MGARD_CUDA_SEPARATE_COMPILE_COMPILATION_OPTION 
       CUDA_SEPARABLE_COMPILATION ON)

  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")

  if(MGARD_ENABLE_CUDA_FMA)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_FMA")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_FMA")
  endif()

  if(MGARD_ENABLE_CUDA_OPTIMIZE_TURING)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_OPTIMIZE_TURING")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_OPTIMIZE_TURING")
    set(CUDA_ARCH_STRING "${CUDA_ARCH_STRING};75")
  endif()
  if(MGARD_ENABLE_CUDA_OPTIMIZE_VOLTA)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMGARD_CUDA_OPTIMIZE_VOLTA")
    SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -DMGARD_CUDA_OPTIMIZE_VOLTA")
    set(CUDA_ARCH_STRING "${CUDA_ARCH_STRING};70;72")
  endif()
endif()

find_package(ZLIB REQUIRED)
find_package(ZSTD)
find_package(MOAB)

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
	src/compressors.cpp
)

add_library(
	mgard-library
	STATIC
	${MGARD_LIBRARY_CPP}
	${MGARD_CUDA_SRC}
)
set_target_properties(mgard-library PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(mgard-library PROPERTIES OUTPUT_NAME mgard)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")
else()
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
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mgard-library PUBLIC ZLIB::ZLIB)
if(MOAB_FOUND)
	target_sources(mgard-library PRIVATE ${MGARD_MOAB_CPP})
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

install(FILES "${PROJECT_BINARY_DIR}/include/MGARDConfig.hpp" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard")
install(DIRECTORY "include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard")

#It would be better if these took `MOAB_FOUND` into account.
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
	tests/src/test_compress.cpp
	tests/src/test_decompose.cpp
)

set(
	MGARD_EXECUTABLE_HPP
	include/arguments.hpp
	include/cmdline.hpp
	include/metadata.hpp
	include/output.hpp
)
set(
	MGARD_EXECUTABLE_TPP
	include/arguments.tpp
)
set(
	MGARD_EXECUTABLE_CPP
	src/arguments.cpp
	src/cmdline.cpp
	src/executable.cpp
	src/metadata.cpp
	src/output.cpp
)

set(
	MGARD_BENCHMARK_CPP
	src/benchmarks.cpp
)

find_program(DOXYGEN NAMES "doxygen")
find_program(SED NAMES "sed")
find_program(PYTHON3 NAMES "python3" "python")

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
		set(PYTHON3 "PYTHON3-NOTFOUND")
	endif()
endif()

set(
	MGARD_DOC_PAGES
	README.md
	doc/MGARD-GPU.md
	examples/README.md
	examples/linking/README.md
	examples/compression/README.md
)

if(NOT DOXYGEN STREQUAL "DOXYGEN-NOTFOUND" AND NOT SED STREQUAL "SED-NOTFOUND" AND NOT PYTHON3 STREQUAL "PYTHON3-NOTFOUND")
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
			#Get rid of the workflow status badges.
			COMMAND "${SED}"
			ARGS
				"--expression" "s/\\\\s*\\\\[!\\\\[[^]]\\\\+\\\\]\\\\[[^]]\\\\+\\\\]\\\\]\\\\[[^]]\\\\+\\\\]//g"
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
else()
	message("-- Will not build documentation (dependencies not met).")
endif()

find_package(Catch2)

if(Catch2_FOUND)
	configure_file("tests/include/testing_paths.hpp.in" "include/testing_paths.hpp")
	add_executable(mgard-tests ${MGARD_TESTS_CPP})
	target_include_directories(mgard-tests PRIVATE tests/include "${CMAKE_CURRENT_BINARY_DIR}/include")
	target_link_libraries(mgard-tests PRIVATE mgard-library Catch2::Catch2WithMain)
	set_target_properties(mgard-tests PROPERTIES OUTPUT_NAME tests)
else()
	message("-- Will not build tests (dependencies not met).")
endif()

if(MGARD_ENABLE_CUDA)
		add_executable(MgardCudaExec ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MgardCudaExec.cpp)
    target_link_libraries(MgardCudaExec mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(MgardCudaExec PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

    add_executable(MgardSerialExec ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MgardSerialExec.cpp)
    target_link_libraries(MgardSerialExec mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(MgardSerialExec PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

    add_executable(BatchTests ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/Testing/BatchTests.cpp)
    target_link_libraries(BatchTests mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(BatchTests PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
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
	add_executable(mgard-benchmarks ${MGARD_BENCHMARK_CPP})
	target_link_libraries(mgard-benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main mgard-library)
	set_target_properties(mgard-benchmarks PROPERTIES OUTPUT_NAME benchmarks)
else()
	message("-- Will not build benchmarks (dependencies not met).")
endif()

find_package(yaml-cpp)
find_package(archive)
find_package(TCLAP)

if(yaml-cpp_FOUND AND archive_FOUND AND TCLAP_FOUND)
	add_executable(mgard-executable ${MGARD_EXECUTABLE_CPP})
	target_include_directories(mgard-executable PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")
	target_link_libraries(mgard-executable PRIVATE mgard-library yaml-cpp::yaml-cpp archive::archive TCLAP::TCLAP)
	set_target_properties(mgard-executable PROPERTIES OUTPUT_NAME mgard)

	install(TARGETS mgard-executable RUNTIME)
	install(FILES man/man1/mgard.1 DESTINATION "${CMAKE_INSTALL_MANDIR}/man1")
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
message("    CUDA-FMA: ${MGARD_ENABLE_CUDA_FMA}")
message("  CUDA-VOLTA: ${MGARD_ENABLE_CUDA_OPTIMIZE_VOLTA}")
message(" CUDA-TURING: ${MGARD_ENABLE_CUDA_OPTIMIZE_TURING}")
message("        MOAB: ${MOAB_FOUND}")
message("      Catch2: ${Catch2_FOUND}")
message("    yaml-cpp: ${yaml-cpp_FOUND}")
message("     archive: ${archive_FOUND}")
message("       TCLAP: ${TCLAP_FOUND}")
message("   benchmark: ${benchmark_FOUND}")
