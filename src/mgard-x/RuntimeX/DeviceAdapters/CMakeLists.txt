if(MGARD_ENABLE_SERIAL)
  list(APPEND MGARD_X_SERIAL_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterSerial.cpp)
  set(MGARD_X_SERIAL_SRC ${MGARD_X_SERIAL_SRC} PARENT_SCOPE)
endif()
if(MGARD_ENABLE_OPENMP)
  list(APPEND MGARD_X_OPENMP_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterOpenmp.cpp)
  set(MGARD_X_OPENMP_SRC ${MGARD_X_OPENMP_SRC} PARENT_SCOPE)
endif()
if(MGARD_ENABLE_CUDA)
  list(APPEND MGARD_X_CUDA_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterCuda.cu)
  set(MGARD_X_CUDA_SRC ${MGARD_X_CUDA_SRC} PARENT_SCOPE)
endif()
if(MGARD_ENABLE_HIP)
  list(APPEND MGARD_X_HIP_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterHip.cpp)
  set(MGARD_X_HIP_SRC ${MGARD_X_HIP_SRC} PARENT_SCOPE)
endif()
if(MGARD_ENABLE_SYCL)
  list(APPEND MGARD_X_SYCL_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterSycl.cpp)
  set(MGARD_X_SYCL_SRC ${MGARD_X_SYCL_SRC} PARENT_SCOPE)
endif()
if(MGARD_ENABLE_KOKKOS)
  list(APPEND MGARD_X_KOKKOS_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterKokkos.cpp)
  set(MGARD_X_KOKKOS_SRC ${MGARD_X_KOKKOS_SRC} PARENT_SCOPE)
endif()