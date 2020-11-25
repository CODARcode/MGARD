# FindTCLAP
# ---------
#
# Find the TCLAP headers. See <http://tclap.sourceforge.net/>.
#
# This module defines the following variables:
#
#     TCLAP_FOUND        - Whether TCLAP was found.
#
# and the following imported targets:
#
#     TCLAP::TCLAP     - TCLAP library.

set(TCLAP_INCLUDE_DIRS_PATHS ~/.local/include /usr/local/include /usr/include)

set(TCLAP_INCLUDE_DIRS_ERROR_MESSAGE "TCLAP headers not found.")
set(TCLAP_NOT_FOUND_MESSAGE "TCLAP not found.")

set(TCLAP_FOUND TRUE)

if(TCLAP_FIND_QUIETLY OR NOT TCLAP_FIND_REQUIRED)
	find_path(TCLAP_INCLUDE_DIRS tclap PATHS ${TCLAP_INCLUDE_DIRS_PATHS})
else()
	find_path(TCLAP_INCLUDE_DIRS tclap PATHS ${TCLAP_INCLUDE_DIRS_PATHS} REQUIRED)
endif()

if (TCLAP_INCLUDE_DIRS STREQUAL "TCLAP_INCLUDE_DIRS-NOTFOUND")
	set(TCLAP_FOUND FALSE)
	if(NOT TCLAP_FIND_QUIETLY)
		message(STATUS ${TCLAP_INCLUDE_DIRS_ERROR_MESSAGE})
	endif()
endif()

if (TCLAP_FOUND)
	add_library(TCLAP::TCLAP INTERFACE IMPORTED)
	set_target_properties(TCLAP::TCLAP PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${TCLAP_INCLUDE_DIRS})
else()
	if(TCLAP_FIND_REQUIRED)
		message(FATAL_ERROR ${TCLAP_NOT_FOUND_MESSAGE})
	else()
		message(STATUS ${TCLAP_NOT_FOUND_MESSAGE})
	endif()
endif()
