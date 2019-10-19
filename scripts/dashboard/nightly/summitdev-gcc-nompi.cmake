# Client maintainer: chuck.atkins@kitware.com
set(CTEST_SITE "summitdev.ccs.ornl.gov")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_FLAGS "-k -j10")
set(CTEST_TEST_ARGS PARALLEL_LEVEL 10)

set(CTEST_BUILD_NAME "Linux-EL7-PPC64LE_GCC-7.1.0_NoMPI")
set(dashboard_model Nightly)
set(CTEST_DASHBOARD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/${CTEST_BUILD_NAME})


include(${CMAKE_CURRENT_LIST_DIR}/../EnvironmentModules.cmake)
module(purge)
module(load git)
module(load gcc/7.1.0)
module(load hdf5)

set(ENV{CC}  gcc)
set(ENV{CXX} g++)
set(ENV{FC}  gfortran)

set(dashboard_cache "
#ADIOS2_USE_BZip2:STRING=ON
#ADIOS2_USE_DataMan:STRING=ON
ADIOS2_USE_Fortran:STRING=ON
#ADIOS2_USE_HDF5:STRING=ON
#ADIOS2_USE_MPI:STRING=ON
#ADIOS2_USE_Python:STRING=ON
#ADIOS2_USE_ZFP:STRING=ON
#ADIOS2_USE_ZeroMQ:STRING=ON
")

include(${CMAKE_CURRENT_LIST_DIR}/../adios_common.cmake)
