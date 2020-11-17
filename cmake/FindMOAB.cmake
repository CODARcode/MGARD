# FindMOAB
# --------
#
# Find the MOAB headers and library. See <https://press3.mcs.anl.gov/sigma/moab-library/>.
#
# This module defines the following variables:
#
#     MOAB_FOUND        - Whether MOAB was found.
#     MOAB_INCLUDE_DIRS - Location of MOAB headers.
#     MOAB_LIBRARIES    - MOAB library and other required libraries.

set(MOAB_LIBRARIES_PATHS ~/.local/lib /usr/local/lib /usr/lib)
set(MOAB_INCLUDE_DIRS_PATHS ~/.local/include /usr/local/include /usr/include)

set(MOAB_LIBRARIES_ERROR_MESSAGE "Could not find MOAB library.")
set(MOAB_INCLUDE_DIRS_ERROR_MESSAGE "MOAB headers not found.")
set(MOAB_FOUND_ERROR_MESSAGE "MOAB not found.")

set(MOAB_FOUND TRUE)

if(MOAB_FIND_QUIETLY OR NOT MOAB_FIND_REQUIRED)
	find_path(MOAB_INCLUDE_DIRS moab PATHS ${MOAB_INCLUDE_DIRS_PATHS})
else()
	find_path(MOAB_INCLUDE_DIRS moab PATHS ${MOAB_INCLUDE_DIRS_PATHS} REQUIRED)
endif()

if (MOAB_INCLUDE_DIRS STREQUAL "MOAB_INCLUDE_DIRS-NOTFOUND")
	set(MOAB_FOUND FALSE)
	if(NOT MOAB_FIND_QUIETLY)
		message(STATUS ${MOAB_INCLUDE_DIRS_ERROR_MESSAGE})
	endif()
endif()

if (MOAB_FIND_QUIETLY OR NOT MOAB_FIND_REQUIRED)
	find_library(MOAB_LIBRARIES MOAB PATHS ${MOAB_LIBRARIES_PATHS})
else()
	find_library(MOAB_LIBRARIES MOAB PATHS ${MOAB_LIBRARIES_PATHS} REQUIRED)
endif()

if(MOAB_LIBRARIES STREQUAL "MOAB_LIBRARIES-NOTFOUND")
	set(MOAB_FOUND FALSE)
	if(NOT MOAB_FIND_QUIETLY)
		message(STATUS ${MOAB_LIBRARIES_ERROR_MESSAGE})
	endif()
endif()

if(MOAB_FIND_REQUIRED)
	find_package(LAPACK REQUIRED)
else()
	find_package(LAPACK)
endif()

if(NOT LAPACK_FOUND)
	set(MOAB_FOUND FALSE)
endif()

set(MOAB_LIBRARIES ${MOAB_LIBRARIES} ${LAPACK_LIBRARIES})

if (NOT MOAB_FOUND)
	if(MOAB_FIND_REQUIRED)
		message(FATAL_ERROR ${MOAB_FOUND_ERROR_MESSAGE})
	else()
		message(STATUS ${MOAB_FOUND_ERROR_MESSAGE})
	endif()
endif()
