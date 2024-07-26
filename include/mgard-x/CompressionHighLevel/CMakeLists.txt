list(APPEND MGARD_X_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/CompressionHighLevel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CompressionHighLevel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DomainDecomposer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ErrorToleranceCalculator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GPUPipeline.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CPUPipeline.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShapeAdjustment.hpp
    )
set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)