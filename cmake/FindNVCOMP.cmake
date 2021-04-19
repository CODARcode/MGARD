#
# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
#
#
# FindNVCOMP
# -----------
#
# Try to find the NVCOMP library
#
# This module defines the following variables:
#
#   NVCOMP_FOUND        - System has NVCOMP
#   NVCOMP_INCLUDE_DIRS - The NVCOMP include directory
#   NVCOMP_LIBRARIES    - Link these to use NVCOMP
#
# and the following imported targets:
#   NVCOMP - The NVCOMP compression library target
#
# You can also set the following variable to help guide the search:
#   NVCOMP_ROOT - The install prefix for NVCOMP containing the
#              include and lib folders
#              Note: this can be set as a CMake variable or an
#                    environment variable.  If specified as a CMake
#                    variable, it will override any setting specified
#                    as an environment variable.

if(NOT NVCOMP_FOUND)
  if((NOT NVCOMP_ROOT) AND (NOT (ENV{NVCOMP_ROOT} STREQUAL "")))
    set(NVCOMP_ROOT "$ENV{NVCOMP_ROOT}")
  endif()
  if(NVCOMP_ROOT)
    set(NVCOMP_INCLUDE_OPTS HINTS ${NVCOMP_ROOT}/include NO_DEFAULT_PATHS)
    set(NVCOMP_LIBRARY_OPTS
      HINTS ${NVCOMP_ROOT}/lib
      NO_DEFAULT_PATHS
    )
  endif()

  find_path(NVCOMP_INCLUDE_DIR nvcomp.h ${NVCOMP_INCLUDE_OPTS})
  find_library(NVCOMP_LIBRARY NAMES nvcomp ${NVCOMP_LIBRARY_OPTS})

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(NVCOMP
    FOUND_VAR NVCOMP_FOUND
    REQUIRED_VARS NVCOMP_LIBRARY NVCOMP_INCLUDE_DIR
  )
  if(NVCOMP_FOUND)
    set(NVCOMP_INCLUDE_DIRS ${NVCOMP_INCLUDE_DIR})
    set(NVCOMP_LIBRARIES ${NVCOMP_LIBRARY})
    include_directories(${NVCOMP_INCLUDE_DIR})
    if(NVCOMP_FOUND AND NOT TARGET NVCOMP)
      add_library(NVCOMP::NVCOMP UNKNOWN IMPORTED)
      set_target_properties(NVCOMP::NVCOMP PROPERTIES
        IMPORTED_LOCATION             "${NVCOMP_LIBRARY}"
        INTERFACE_LINK_LIBRARIES      "${NVCOMP_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVCOMP_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()
