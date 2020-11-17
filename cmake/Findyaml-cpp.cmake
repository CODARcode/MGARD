# Findyaml-cpp
# ------------
#
# Find the yaml-cpp headers and library. See <https://github.com/jbeder/yaml-cpp>.
#
# This module defines the following variables:
#
#     YAML-CPP_FOUND        - Whether yaml-cpp was found.
#     YAML-CPP_INCLUDE_DIRS - Location of yaml-cpp headers.
#     YAML-CPP_LIBRARIES    - yaml-cpp library and other required libraries.

set(YAML-CPP_LIBRARIES_PATHS ~/.local/lib /usr/local/lib /usr/lib)
set(YAML-CPP_INCLUDE_DIRS_PATHS ~/.local/include /usr/local/include /usr/include)

set(YAML-CPP_LIBRARIES_ERROR_MESSAGE "Could not find yaml-cpp library.")
set(YAML-CPP_INCLUDE_DIRS_ERROR_MESSAGE "yaml-cpp headers not found.")
set(YAML-CPP_FOUND_ERROR_MESSAGE "yaml-cpp not found.")

set(YAML-CPP_FOUND TRUE)

if(YAML-CPP_FIND_QUIETLY OR NOT YAML-CPP_FIND_REQUIRED)
	find_path(YAML-CPP_INCLUDE_DIRS yaml-cpp PATHS ${YAML-CPP_INCLUDE_DIRS_PATHS})
else()
	find_path(YAML-CPP_INCLUDE_DIRS yaml-cpp PATHS ${YAML-CPP_INCLUDE_DIRS_PATHS} REQUIRED)
endif()

if (YAML-CPP_INCLUDE_DIRS STREQUAL "YAML-CPP_INCLUDE_DIRS-NOTFOUND")
	set(YAML-CPP_FOUND FALSE)
	if(NOT YAML-CPP_FIND_QUIETLY)
		message(STATUS ${YAML-CPP_INCLUDE_DIRS_ERROR_MESSAGE})
	endif()
endif()

if (YAML-CPP_FIND_QUIETLY OR NOT YAML-CPP_FIND_REQUIRED)
	find_library(YAML-CPP_LIBRARIES yaml-cpp PATHS ${YAML-CPP_LIBRARIES_PATHS})
else()
	find_library(YAML-CPP_LIBRARIES yaml-cpp PATHS ${YAML-CPP_LIBRARIES_PATHS} REQUIRED)
endif()

if(YAML-CPP_LIBRARIES STREQUAL "YAML-CPP_LIBRARIES-NOTFOUND")
	set(YAML-CPP_FOUND FALSE)
	if(NOT YAML-CPP_FIND_QUIETLY)
		message(STATUS ${YAML-CPP_LIBRARIES_ERROR_MESSAGE})
	endif()
endif()

if (NOT YAML-CPP_FOUND)
	if(YAML-CPP_FIND_REQUIRED)
		message(FATAL_ERROR ${YAML-CPP_FOUND_ERROR_MESSAGE})
	else()
		message(STATUS ${YAML-CPP_FOUND_ERROR_MESSAGE})
	endif()
endif()
