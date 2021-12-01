if(MGARD_ENABLE_SERIAL)
  list(APPEND MGARD_X_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterSerial.cpp)
endif()
if(MGARD_ENABLE_CUDA)
  list(APPEND MGARD_X_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterCuda.cu)
endif()
if(MGARD_ENABLE_HIP)
  list(APPEND MGARD_X_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterHip.cpp)
endif()
if(MGARD_ENABLE_KOKKOS)
  list(APPEND MGARD_X_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapterKokkos.cpp)
endif()


set(MGARD_X_SRC ${MGARD_X_SRC} PARENT_SCOPE)