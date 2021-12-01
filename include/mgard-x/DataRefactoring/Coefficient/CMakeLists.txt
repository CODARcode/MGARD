list(APPEND MGARD_X_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/GPKFunctor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernel.hpp   
    ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernel3D.hpp   
    ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernel3D.h
    )
set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)