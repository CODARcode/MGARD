# Findarchive
# -----------
#
# Find the archive headers and library. See <https://www.libarchive.org/>.
#
# This module defines the following variables:
#
#     ARCHIVE_FOUND        - Whether archive was found.
#     ARCHIVE_INCLUDE_DIRS - Location of archive headers.
#     ARCHIVE_LIBRARIES    - archive library and other required libraries.

set(ARCHIVE_LIBRARIES_PATHS ~/.local/lib /usr/local/lib /usr/lib)
set(ARCHIVE_INCLUDE_DIRS_PATHS ~/.local/include /usr/local/include /usr/include)

set(ARCHIVE_LIBRARIES_ERROR_MESSAGE "Could not find archive library.")
set(ARCHIVE_INCLUDE_DIRS_ERROR_MESSAGE "archive headers not found.")
set(ARCHIVE_FOUND_ERROR_MESSAGE "archive not found.")

set(ARCHIVE_FOUND TRUE)

if(ARCHIVE_FIND_QUIETLY OR NOT ARCHIVE_FIND_REQUIRED)
	find_path(ARCHIVE_INCLUDE_DIRS NAMES archive.h archive_entry.h PATHS ${ARCHIVE_INCLUDE_DIRS_PATHS})
else()
	find_path(ARCHIVE_INCLUDE_DIRS NAMES archive.h archive_entry.h PATHS ${ARCHIVE_INCLUDE_DIRS_PATHS} REQUIRED)
endif()

if (ARCHIVE_INCLUDE_DIRS STREQUAL "ARCHIVE_INCLUDE_DIRS-NOTFOUND")
	set(ARCHIVE_FOUND FALSE)
	if(NOT ARCHIVE_FIND_QUIETLY)
		message(STATUS ${ARCHIVE_INCLUDE_DIRS_ERROR_MESSAGE})
	endif()
endif()

if (ARCHIVE_FIND_QUIETLY OR NOT ARCHIVE_FIND_REQUIRED)
	find_library(ARCHIVE_LIBRARIES archive PATHS ${ARCHIVE_LIBRARIES_PATHS})
else()
	find_library(ARCHIVE_LIBRARIES archive PATHS ${ARCHIVE_LIBRARIES_PATHS} REQUIRED)
endif()

if(ARCHIVE_LIBRARIES STREQUAL "ARCHIVE_LIBRARIES-NOTFOUND")
	set(ARCHIVE_FOUND FALSE)
	if(NOT ARCHIVE_FIND_QUIETLY)
		message(STATUS ${ARCHIVE_LIBRARIES_ERROR_MESSAGE})
	endif()
endif()

if (NOT ARCHIVE_FOUND)
	if(ARCHIVE_FIND_REQUIRED)
		message(FATAL_ERROR ${ARCHIVE_FOUND_ERROR_MESSAGE})
	else()
		message(STATUS ${ARCHIVE_FOUND_ERROR_MESSAGE})
	endif()
endif()
