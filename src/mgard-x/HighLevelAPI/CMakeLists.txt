if(MGARD_ENABLE_SERIAL)
  add_subdirectory (Serial)
endif()
if(MGARD_ENABLE_CUDA)
  add_subdirectory (CUDA)
endif()

list(APPEND MGARD_X_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/DynamicAPI.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/Metadata.cpp)

set(MGARD_X_SRC ${MGARD_X_SRC} PARENT_SCOPE)
