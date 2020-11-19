# Findyaml-cpp
# ------------
#
# Find the yaml-cpp headers and library. See <https://github.com/jbeder/yaml-cpp>.
#
# This module defines the following variables:
#
#     yaml-cpp_FOUND        - Whether yaml-cpp was found.
#     yaml-cpp_INCLUDE_DIRS - Location of yaml-cpp headers.
#     yaml-cpp_LIBRARIES    - yaml-cpp library and other required libraries.

set(yaml-cpp_LIBRARIES_PATHS ~/.local/lib /usr/local/lib /usr/lib)
set(yaml-cpp_INCLUDE_DIRS_PATHS ~/.local/include /usr/local/include /usr/include)

set(yaml-cpp_LIBRARIES_ERROR_MESSAGE "Could not find yaml-cpp library.")
set(yaml-cpp_INCLUDE_DIRS_ERROR_MESSAGE "yaml-cpp headers not found.")
set(yaml-cpp_FOUND_ERROR_MESSAGE "yaml-cpp not found.")

set(yaml-cpp_FOUND TRUE)

if(yaml-cpp_FIND_QUIETLY OR NOT yaml-cpp_FIND_REQUIRED)
	find_path(yaml-cpp_INCLUDE_DIRS yaml-cpp PATHS ${yaml-cpp_INCLUDE_DIRS_PATHS})
else()
	find_path(yaml-cpp_INCLUDE_DIRS yaml-cpp PATHS ${yaml-cpp_INCLUDE_DIRS_PATHS} REQUIRED)
endif()

if (yaml-cpp_INCLUDE_DIRS STREQUAL "yaml-cpp_INCLUDE_DIRS-NOTFOUND")
	set(yaml-cpp_FOUND FALSE)
	if(NOT yaml-cpp_FIND_QUIETLY)
		message(STATUS ${yaml-cpp_INCLUDE_DIRS_ERROR_MESSAGE})
	endif()
endif()

if (yaml-cpp_FIND_QUIETLY OR NOT yaml-cpp_FIND_REQUIRED)
	find_library(yaml-cpp_LIBRARIES yaml-cpp PATHS ${yaml-cpp_LIBRARIES_PATHS})
else()
	find_library(yaml-cpp_LIBRARIES yaml-cpp PATHS ${yaml-cpp_LIBRARIES_PATHS} REQUIRED)
endif()

if(yaml-cpp_LIBRARIES STREQUAL "yaml-cpp_LIBRARIES-NOTFOUND")
	set(yaml-cpp_FOUND FALSE)
	if(NOT yaml-cpp_FIND_QUIETLY)
		message(STATUS ${yaml-cpp_LIBRARIES_ERROR_MESSAGE})
	endif()
endif()

if (NOT yaml-cpp_FOUND)
	if(yaml-cpp_FIND_REQUIRED)
		message(FATAL_ERROR ${yaml-cpp_FOUND_ERROR_MESSAGE})
	else()
		message(STATUS ${yaml-cpp_FOUND_ERROR_MESSAGE})
	endif()
endif()
