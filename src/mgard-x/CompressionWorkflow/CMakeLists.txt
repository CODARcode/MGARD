if(MGARD_ENABLE_CUDA)
  add_subdirectory (CUDA)
endif()
set(MGARD_X_SRC ${MGARD_X_SRC} PARENT_SCOPE)