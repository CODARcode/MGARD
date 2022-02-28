# Findzstd
# ---------
#
# Find zstd. See <https://www.zstd.net/>.
#
# This module defines the following variables:
#
#     zstd_FOUND        - Whether zstd was found.
#
# and the following imported targets:
#
#     zstd::zstd        - zstd library.

set(zstd_NOT_FOUND_MESSAGE "\"zstd\" not found.")

#We'd have to pass these to the `find_package` call.
if(zstd_FIND_VERSION_COMPLETE)
	message(FATAL_ERROR "Upgrade `Findzstd.cmake` to support package version.")
endif()

if(zstd_FIND_COMPONENTS)
	message(FATAL_ERROR "Upgrade `Findzstd.cmake` to support package components.")
endif()

#In case `find_package(zstd)` was run twice.
if(TARGET zstd::zstd)
	return()
endif()

if(zstd_FIND_QUIETLY)
	find_package(zstd QUIET CONFIG)
else()
	message(VERBOSE "Attempting to find \"zstd\" with `find_package`.")
	find_package(zstd CONFIG)
endif()

if(zstd_FOUND)
	if(TARGET zstd::libzstd_shared)
		if(NOT zstd_FIND_QUIETLY)
			message(VERBOSE "Adding `zstd::zstd` as an alias to `zstd::libzstd_shared`.")
		endif()
		add_library(zstd::zstd ALIAS zstd::libzstd_shared)
	elseif(TARGET zstd::libzstd_static)
		if(NOT zstd_FIND_QUIETLY)
			message(VERBOSE "Adding `zstd::zstd` as an alias to `zstd::libzstd_static`.")
		endif()
		add_library(zstd::zstd ALIAS zstd::libzstd_static)
	else()
		if(NOT zstd_FIND_QUIETLY)
			message(STATUS "\"zstd\" found but neither `zstd::libzstd_shared` nor `zstd::libzstd_static` is an existing target.")
		endif()
		set(zstd_FOUND FALSE)
	endif()
endif()

if(zstd_FOUND)
	return()
endif()

find_package(PkgConfig)
if(NOT PKG_CONFIG_FOUND)
	set(zstd_NOT_FOUND_MESSAGE "\"zstd\" not found with `find_package` and \"PkgConfig\" not found.")
else()
	if(zstd_FIND_QUIETLY)
		pkg_search_module(libzstd QUIET IMPORTED_TARGET GLOBAL libzstd)
	else()
		message(VERBOSE "Attempting to find \"zstd\" with `pkg_search_module`.")
		pkg_search_module(libzstd IMPORTED_TARGET GLOBAL libzstd)
	endif()
	if(NOT libzstd_FOUND)
		set(zstd_NOT_FOUND_MESSAGE "\"zstd\" found with neither `find_package` nor `pkg_search_module`.")
	else()
		set(zstd_FOUND TRUE)
		if(NOT zstd_FIND_QUIETLY)
			message(VERBOSE "Adding `zstd::zstd` as an alias to `PkgConfig::libzstd`.")
		endif()
		add_library(zstd::zstd ALIAS PkgConfig::libzstd)
	endif()
endif()

if(NOT zstd_FOUND)
	if(zstd_FIND_REQUIRED)
		message(FATAL_ERROR "${zstd_NOT_FOUND_MESSAGE}")
	else()
		message(STATUS "${zstd_NOT_FOUND_MESSAGE}")
	endif()
endif()
