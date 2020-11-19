# Findarchive
# -----------
#
# Find the archive headers and library. See <https://www.libarchive.org/>.
#
# This module defines the following variables:
#
#     archive_FOUND        - Whether archive was found.
#     archive_INCLUDE_DIRS - Location of archive headers.
#     archive_LIBRARIES    - archive library and other required libraries.

set(archive_LIBRARIES_PATHS ~/.local/lib /usr/local/lib /usr/lib)
set(archive_INCLUDE_DIRS_PATHS ~/.local/include /usr/local/include /usr/include)

set(archive_LIBRARIES_ERROR_MESSAGE "Could not find archive library.")
set(archive_INCLUDE_DIRS_ERROR_MESSAGE "archive headers not found.")
set(archive_FOUND_ERROR_MESSAGE "archive not found.")

set(archive_FOUND TRUE)

if(archive_FIND_QUIETLY OR NOT archive_FIND_REQUIRED)
	find_path(archive_INCLUDE_DIRS NAMES archive.h archive_entry.h PATHS ${archive_INCLUDE_DIRS_PATHS})
else()
	find_path(archive_INCLUDE_DIRS NAMES archive.h archive_entry.h PATHS ${archive_INCLUDE_DIRS_PATHS} REQUIRED)
endif()

if (archive_INCLUDE_DIRS STREQUAL "archive_INCLUDE_DIRS-NOTFOUND")
	set(archive_FOUND FALSE)
	if(NOT archive_FIND_QUIETLY)
		message(STATUS ${archive_INCLUDE_DIRS_ERROR_MESSAGE})
	endif()
endif()

if (archive_FIND_QUIETLY OR NOT archive_FIND_REQUIRED)
	find_library(archive_LIBRARIES archive PATHS ${archive_LIBRARIES_PATHS})
else()
	find_library(archive_LIBRARIES archive PATHS ${archive_LIBRARIES_PATHS} REQUIRED)
endif()

if(archive_LIBRARIES STREQUAL "archive_LIBRARIES-NOTFOUND")
	set(archive_FOUND FALSE)
	if(NOT archive_FIND_QUIETLY)
		message(STATUS ${archive_LIBRARIES_ERROR_MESSAGE})
	endif()
endif()

if (NOT archive_FOUND)
	if(archive_FIND_REQUIRED)
		message(FATAL_ERROR ${archive_FOUND_ERROR_MESSAGE})
	else()
		message(STATUS ${archive_FOUND_ERROR_MESSAGE})
	endif()
endif()
