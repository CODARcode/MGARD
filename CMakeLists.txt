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
set(MGARD_VERSION_MINOR "5")
set(MGARD_VERSION_PATCH "2")

set(MGARD_FILE_VERSION_MAJOR "1")
set(MGARD_FILE_VERSION_MINOR "0")
set(MGARD_FILE_VERSION_PATCH "0")

project(
  MGARD
  HOMEPAGE_URL "https://github.com/CODARcode/MGARD"
  VERSION "${MGARD_VERSION_MAJOR}.${MGARD_VERSION_MINOR}.${MGARD_VERSION_PATCH}"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Configurable Options
option(MGARD_ENABLE_BENCHMARKS "Build benchmarks." OFF)

option(MGARD_ENABLE_DOCS "Build documentation." OFF)

option(MGARD_ENABLE_OPENMP "Enable OpenMP support." OFF)
option(MGARD_ENABLE_CUDA "Enable CUDA support" OFF)
option(MGARD_ENABLE_SERIAL "Enable SERIAL support" ON)
option(MGARD_ENABLE_HIP "Enable HIP support" OFF)
option(MGARD_ENABLE_SYCL "Enable SYCL support" OFF)
option(MGARD_ENABLE_LEGACY_CUDA "Enable legacy CUDA support" OFF)
option(MGARD_ENABLE_COORDINATE_NORMALIZATION "Enable coordinate normalication for uniform grid" ON)
option(MGARD_ENABLE_AUTO_TUNING "Enable auto tuning" OFF)
option(MGARD_ENABLE_MDR "Enable MDR" OFF)
option(MGARD_ENABLE_EXTERNAL_COMPRESSOR "Enable external compressor" OFF)

option(MGARD_ENABLE_CLI "Build executable." OFF)

option(MGARD_ENABLE_UNSTRUCTURED "Enable experimental unstructured mesh support." OFF)

option(MGARD_ENABLE_TIMING "Enable timing." OFF)

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
find_package(OpenMP)

# Default to a release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

if (MGARD_ENABLE_SERIAL OR 
    MGARD_ENABLE_OPENMP OR 
    MGARD_ENABLE_CUDA   OR
    MGARD_ENABLE_HIP    OR
    MGARD_ENABLE_SYCL)

  if(MGARD_ENABLE_SYCL)
    set(MGARD_ENABLE_SERIAL OFF)
    set(MGARD_ENABLE_OPENMP OFF)
  endif()

  set (CMAKE_CXX_STANDARD 17)
  set (CMAKE_CXX_STANDARD_REQUIRED ON)
  set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")

  include(MgardXGenerateSource)
  add_subdirectory (src/mgard-x)
  add_subdirectory (include/mgard-x)
  add_subdirectory (include/mgard-x/MDR)
  add_subdirectory (include/mgard-x/MDR-X)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(zstd IMPORTED_TARGET GLOBAL libzstd)

  add_executable(mgard-x ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard-x/Executables/mgard-x.cpp)
  target_link_libraries(mgard-x mgard-library  ${CMAKE_DL_LIBS})
  target_include_directories(mgard-x PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
  install(TARGETS mgard-x)

  add_executable(mgard-x-autotuner ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard-x/Executables/mgard-x-autotuner.cpp)
  target_link_libraries(mgard-x-autotuner mgard-library  ${CMAKE_DL_LIBS})
  target_include_directories(mgard-x-autotuner PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
  install(TARGETS mgard-x-autotuner)

  add_executable(mgard-cpu ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard-x/Executables/MgardSerialExec.cpp)
  target_link_libraries(mgard-cpu mgard-library  ${CMAKE_DL_LIBS})
  target_include_directories(mgard-cpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
  install(TARGETS mgard-cpu)

  if(MGARD_ENABLE_MDR)
    add_executable(mdr-x ${CMAKE_CURRENT_SOURCE_DIR}/src/mgard-x/Executables/mdr-x.cpp)
    target_link_libraries(mdr-x mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(mdr-x PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                              $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
    install(TARGETS mdr-x)
  endif()
endif()

#Adding library here so we can set compile definitions for it.
add_library(mgard-library)

if(MGARD_ENABLE_SERIAL)
  target_compile_definitions(mgard-library PUBLIC MGARD_ENABLE_SERIAL)
  set_source_files_properties(${MGARD_X_SERIAL_SRC} PROPERTIES LANGUAGE CXX)
endif()

if(MGARD_ENABLE_OPENMP)
  target_compile_definitions(mgard-library PUBLIC MGARD_ENABLE_OPENMP)
  set_source_files_properties(${MGARD_X_OPENMP_SRC} PROPERTIES LANGUAGE CXX)
endif()

if(MGARD_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()

  find_package(nvcomp REQUIRED)
  find_package(CUDAToolkit REQUIRED)

  target_compile_definitions(mgard-library PUBLIC MGARD_ENABLE_CUDA)
  set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w --generate-line-info")
  
  set (MGARD_X_SEPARATE_COMPILE_COMPILATION_OPTION 
       CUDA_SEPARABLE_COMPILATION ON)

  set_source_files_properties(${MGARD_X_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
endif()

if (MGARD_ENABLE_LEGACY_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()

  find_package(nvcomp REQUIRED)
  find_package(CUDAToolkit REQUIRED)

  add_subdirectory (src/cuda)
  add_subdirectory (include/cuda)
  add_executable(mgard-gpu ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MgardCudaExec.cpp)
    target_link_libraries(mgard-gpu mgard-library  ${CMAKE_DL_LIBS})
    target_include_directories(mgard-gpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

  target_compile_definitions(mgard-library PUBLIC MGARD_ENABLE_LEGACY_CUDA)
  set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")
  
  set (MGARD_X_SEPARATE_COMPILE_COMPILATION_OPTION 
       CUDA_SEPARABLE_COMPILATION ON)

  install(TARGETS mgard-gpu)
endif()

if (MGARD_ENABLE_HIP) 
  find_package(HIP REQUIRED)
  enable_language(HIP)
  target_compile_definitions(mgard-library PUBLIC MGARD_ENABLE_HIP)
  set (CMAKE_HIP_FLAGS  "${CMAKE_HIP_FLAGS} -w")
  set_source_files_properties(${MGARD_X_HIP_SRC} PROPERTIES LANGUAGE HIP)
endif()

if (MGARD_ENABLE_SYCL)
  # No need to link with sycl libraries for now
  # find_package(IntelDPCPP REQUIRED)
  set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")
endif()

find_package(ZLIB REQUIRED)

find_package(PkgConfig REQUIRED)

set(
  MGARD_LIBRARY_CPP
        src/compress.cpp
        src/compress_internal.cpp
  src/compressors.cpp
  src/format.cpp
)

set(MAXIMUM_DIMENSION 4 CACHE STRING "Maximum supported dimension for self-describing decompression.")

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
list(APPEND MGARD_LIBRARY_CPP "${MGARD_FILE_FORMAT_CPP}")

set(MGARD_GENERATE_DIMENSION_SWITCH "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_dimension_switch.py")
set(MGARD_COMPRESS_INTERNAL_CPP "${MGARD_FILE_FORMAT_SRCDIR}/compress_internal_generated.cpp")
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
list(APPEND MGARD_LIBRARY_CPP "${MGARD_COMPRESS_INTERNAL_CPP}")

#See <https://github.com/protocolbuffers/protobuf/issues/2032>.
set_source_files_properties("${MGARD_FILE_FORMAT_CPP}" PROPERTIES COMPILE_FLAGS "-Wno-unused-parameter")

target_sources(
  mgard-library
  PRIVATE
  ${MGARD_LIBRARY_CPP}
  ${MGARD_CUDA_SRC}
  ${MGARD_X_SRC}
  ${MGARD_X_SERIAL_SRC}
  ${MGARD_X_OPENMP_SRC}
  ${MGARD_X_CUDA_SRC}
  ${MGARD_X_HIP_SRC}
  ${MGARD_X_SYCL_SRC}
)

set_target_properties(mgard-library PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(mgard-library PROPERTIES OUTPUT_NAME mgard)

target_link_libraries(mgard-library PUBLIC PkgConfig::protobuf)

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

target_include_directories(mgard-library
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mgard-library PUBLIC ZLIB::ZLIB)

if(zstd_FOUND)
  target_compile_definitions(mgard-library PUBLIC MGARD_ZSTD)
  target_link_libraries(mgard-library PUBLIC PkgConfig::zstd)
endif()

if(MGARD_ENABLE_CUDA)
  target_link_libraries(mgard-library PUBLIC nvcomp::nvcomp CUDA::cudart)
endif()

if(MGARD_ENABLE_LEGACY_CUDA)
  target_link_libraries(mgard-library PUBLIC nvcomp::nvcomp CUDA::cudart)
endif()

if(MGARD_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(mgard-library PUBLIC OpenMP::OpenMP_CXX)
  # A bug in CMake causing the '-fopenmp' flag missing when compiling device units during separate compilation
    # Reference: github.com/pkestene/cuda-proj-tmpl/blob/master/src/CMakeLists.txt
    target_compile_options(mgard-library PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
      $<$<COMPILE_LANGUAGE:HIP>:${OpenMP_CXX_FLAGS}>
    )
endif()

if(MGARD_ENABLE_TIMING)
	target_compile_definitions(mgard-library PRIVATE MGARD_TIMING)
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
install(FILES "${PROJECT_BINARY_DIR}/include/MGARDXConfig.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard/mgard-x/RuntimeX")
install(DIRECTORY "include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard")
install(FILES "${MGARD_FILE_FORMAT_HPP}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgard/${MGARD_FILE_FORMAT_INCLUDESUBDIR_COMPONENT}")

if(MGARD_ENABLE_UNSTRUCTURED)
	add_subdirectory("src/unstructured")
endif()

if(MGARD_ENABLE_BENCHMARKS)
	add_subdirectory("src/benchmarks")
endif()

if(MGARD_ENABLE_DOCS)
	add_subdirectory("doc")
endif()

if(BUILD_TESTING)
	add_subdirectory("tests")
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

configure_file(
    "${PROJECT_SOURCE_DIR}/include/MGARDXConfig.h.in"
    "${PROJECT_BINARY_DIR}/include/MGARDXConfig.h"
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

if(MGARD_ENABLE_CLI)
	add_subdirectory("src/cli")
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
message(STATUS "portable backends:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(STATUS "SERIAL:      ${MGARD_ENABLE_SERIAL}")
message(STATUS "OPENMP:      ${MGARD_ENABLE_OPENMP}")
message(STATUS "CUDA:        ${MGARD_ENABLE_CUDA}")
if (MGARD_ENABLE_CUDA)
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  message(STATUS "CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()
message(STATUS "HIP:         ${MGARD_ENABLE_HIP}")
if (MGARD_ENABLE_HIP)
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  message(STATUS "HIP Arch: ${CMAKE_HIP_ARCHITECTURES}")
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()
message(STATUS "SYCL:        ${MGARD_ENABLE_SYCL}")
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "LEGACY CUDA:   ${MGARD_ENABLE_LEGACY_CUDA}")
if (MGARD_ENABLE_LEGACY_CUDA)
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  message(STATUS "CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()
message(STATUS "auto tuning:   ${MGARD_ENABLE_AUTO_TUNING}")
message(STATUS "MDR:           ${MGARD_ENABLE_MDR}")
message(STATUS "timing:        ${MGARD_ENABLE_TIMING}")
message(STATUS "ASan:          ${MGARD_ENABLE_ASAN}")
message(STATUS "UBSan:         ${MGARD_ENABLE_UBSAN}")
message(STATUS "tests:         ${BUILD_TESTING}")
message(STATUS "benchmarks:    ${MGARD_ENABLE_BENCHMARKS}")
message(STATUS "documentation: ${MGARD_ENABLE_DOCS}")
message(STATUS "CLI:           ${MGARD_ENABLE_CLI}")
message(STATUS "unstructured:  ${MGARD_ENABLE_UNSTRUCTURED}")
list(POP_BACK CMAKE_MESSAGE_INDENT)

message(STATUS "")
message(STATUS "Libraries:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(STATUS "zstd:      ${zstd_FOUND}")
message(STATUS "protobuf:  ${protobuf_FOUND}")
if(MGARD_ENABLE_UNSTRUCTURED)
	message(STATUS "MOAB:      ${MOAB_FOUND}")
endif()
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

message(STATUS "")
message(STATUS "Programs:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(STATUS "python3: ${PYTHON3}")
message(STATUS "protoc:  ${PROTOC}")
if(MGARD_ENABLE_DOCS)
	message(STATUS "doxygen: ${DOXYGEN}")
	message(STATUS "sed:     ${SED}")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)
list(POP_BACK CMAKE_MESSAGE_INDENT)
