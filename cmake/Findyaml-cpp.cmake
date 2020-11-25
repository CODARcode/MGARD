# Findyaml-cpp
# ------------
#
# Find the yaml-cpp headers and library. See <https://github.com/jbeder/yaml-cpp>.
#
# This module defines the following variables:
#
#     yaml-cpp_FOUND        - Whether yaml-cpp was found.
#
# and the following imported targets:
#
#     yaml-cpp:yaml-cpp     - yaml-cpp library.

set(yaml-cpp_LIBRARIES_PATHS ~/.local/lib /usr/local/lib /usr/lib)
set(yaml-cpp_INCLUDE_DIRS_PATHS ~/.local/include /usr/local/include /usr/include)

set(yaml-cpp_LIBRARIES_ERROR_MESSAGE "Could not find yaml-cpp library.")
set(yaml-cpp_INCLUDE_DIRS_ERROR_MESSAGE "yaml-cpp headers not found.")
set(yaml-cpp_NOT_FOUND_MESSAGE "yaml-cpp not found.")

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

if (yaml-cpp_FOUND)
	add_library(yaml-cpp::yaml-cpp STATIC IMPORTED)
	set_target_properties(yaml-cpp::yaml-cpp PROPERTIES IMPORTED_LOCATION ${yaml-cpp_LIBRARIES} INTERFACE_INCLUDE_DIRECTORIES ${yaml-cpp_INCLUDE_DIRS})
else()
	if(yaml-cpp_FIND_REQUIRED)
		message(FATAL_ERROR ${yaml-cpp_NOT_FOUND_MESSAGE})
	else()
		message(STATUS ${yaml-cpp_NOT_FOUND_MESSAGE})
	endif()
endif()
