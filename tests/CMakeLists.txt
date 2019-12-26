#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

if(ADIOS2_FOUND)
  add_subdirectory(gray-scott)
endif()

add_subdirectory(cosine)
add_subdirectory(simple1d)
add_subdirectory(dim2kplus1)
add_subdirectory(constant3d)
