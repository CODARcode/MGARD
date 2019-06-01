#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#
#
# FindBLOSC
# -----------
#
# Try to find the BLOSC library
#
# This module defines the following variables:
#
#   BLOSC_FOUND        - System has BLOSC
#   BLOSC_INCLUDE_DIRS - The BLOSC include directory
#   BLOSC_LIBRARIES    - Link these to use BLOSC
#
# and the following imported targets:
#   BLOSC::BLOSC - The BLOSC compression library target
#
# You can also set the following variable to help guide the search:
#   BLOSC_ROOT - The install prefix for BLOSC containing the
#              include and lib folders
#              Note: this can be set as a CMake variable or an
#                    environment variable.  If specified as a CMake
#                    variable, it will override any setting specified
#                    as an environment variable.

if(NOT BLOSC_FOUND)
  if((NOT BLOSC_ROOT) AND (NOT (ENV{BLOSC_ROOT} STREQUAL "")))
    set(BLOSC_ROOT "$ENV{BLOSC_ROOT}")
  endif()
  if(BLOSC_ROOT)
    set(BLOSC_INCLUDE_OPTS HINTS ${BLOSC_ROOT}/include NO_DEFAULT_PATHS)
    set(BLOSC_LIBRARY_OPTS
      HINTS ${BLOSC_ROOT}/lib ${BLOSC_ROOT}/lib64
      NO_DEFAULT_PATHS
    )
  endif()

  find_path(BLOSC_INCLUDE_DIR blosc.h ${BLOSC_INCLUDE_OPTS})
  find_library(BLOSC_LIBRARY NAMES blosc ${BLOSC_LIBRARY_OPTS})

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(BLOSC
    FOUND_VAR BLOSC_FOUND
    REQUIRED_VARS BLOSC_LIBRARY BLOSC_INCLUDE_DIR
  )
  if(BLOSC_FOUND)
    set(BLOSC_INCLUDE_DIRS ${BLOSC_INCLUDE_DIR})
    set(BLOSC_LIBRARIES ${BLOSC_LIBRARY})
    if(BLOSC_FOUND AND NOT TARGET BLOSC::BLOSC)
      add_library(BLOSC::BLOSC UNKNOWN IMPORTED)
      set_target_properties(BLOSC::BLOSC PROPERTIES
        IMPORTED_LOCATION             "${BLOSC_LIBRARY}"
        INTERFACE_LINK_LIBRARIES      "${BLOSC_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()
