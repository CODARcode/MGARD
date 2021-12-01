list(APPEND MGARD_X_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/IPKFunctor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernel.hpp   
    ${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernel3D.hpp   
    ${CMAKE_CURRENT_SOURCE_DIR}/LPKFunctor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearProcessingKernel.hpp   
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearProcessingKernel3D.hpp   
    ${CMAKE_CURRENT_SOURCE_DIR}/LevelwiseProcessingKernel.hpp   
    )
set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)