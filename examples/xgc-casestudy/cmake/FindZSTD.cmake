#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#
#
# FindZSTD
# -----------
#
# Try to find the ZSTD library
#
# This module defines the following variables:
#
#   ZSTD_FOUND        - System has ZSTD
#   ZSTD_INCLUDE_DIRS - The ZSTD include directory
#   ZSTD_LIBRARIES    - Link these to use ZSTD
#
# and the following imported targets:
#   ZSTD::ZSTD - The ZSTD compression library target
#
# You can also set the following variable to help guide the search:
#   ZSTD_ROOT - The install prefix for ZSTD containing the
#              include and lib folders
#              Note: this can be set as a CMake variable or an
#                    environment variable.  If specified as a CMake
#                    variable, it will override any setting specified
#                    as an environment variable.

if(NOT ZSTD_FOUND)
  if((NOT ZSTD_ROOT) AND (NOT (ENV{ZSTD_ROOT} STREQUAL "")))
    set(ZSTD_ROOT "$ENV{ZSTD_ROOT}")
  endif()
  if(ZSTD_ROOT)
    set(ZSTD_INCLUDE_OPTS HINTS ${ZSTD_ROOT}/include NO_DEFAULT_PATHS)
    set(ZSTD_LIBRARY_OPTS
      HINTS ${ZSTD_ROOT}/lib ${ZSTD_ROOT}/lib64
      NO_DEFAULT_PATHS
    )
  endif()

  find_path(ZSTD_INCLUDE_DIR zstd.h ${ZSTD_INCLUDE_OPTS})
  find_library(ZSTD_LIBRARY NAMES zstd ${ZSTD_LIBRARY_OPTS})

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(ZSTD
    FOUND_VAR ZSTD_FOUND
    REQUIRED_VARS ZSTD_LIBRARY ZSTD_INCLUDE_DIR
  )
  if(ZSTD_FOUND)
    set(ZSTD_INCLUDE_DIRS ${ZSTD_INCLUDE_DIR})
    set(ZSTD_LIBRARIES ${ZSTD_LIBRARY})
    add_definitions(-DMGARD_ZSTD)
    include_directories(${ZSTD_INCLUDE_DIR})
    if(ZSTD_FOUND AND NOT TARGET ZSTD::ZSTD)
      add_library(ZSTD::ZSTD UNKNOWN IMPORTED)
      set_target_properties(ZSTD::ZSTD PROPERTIES
        IMPORTED_LOCATION             "${ZSTD_LIBRARY}"
        INTERFACE_LINK_LIBRARIES      "${ZSTD_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()
