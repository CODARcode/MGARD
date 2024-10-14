# - Find zstd
# Find the zstd compression library and includes
#
# ZSTD_INCLUDE_DIRS - where to find zstd.h, etc.
# ZSTD_LIBRARIES - List of libraries when using zstd.
# ZSTD_FOUND - True if zstd found.

find_package(zstd QUIET)
if (zstd_FOUND)
    message(STATUS "Found Zstd (CMAKE): ${zstd_DIR}")
else()
    pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)
    message(STATUS "Found Zstd (PkgConfig): ${ZSTD_LINK_LIBRARIES}")
endif()