add_subdirectory(DataRefactoring)
add_subdirectory(Lossless)
add_subdirectory(Quantization)
add_subdirectory(RuntimeX)
list(APPEND MGARD_X_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/CompressionWorkflow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CompressionWorkflow.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Handle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Handle.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HighLevelAPI.h
    ${CMAKE_CURRENT_SOURCE_DIR}/HighLevelAPI.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Metadata.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Types.h
    )
set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)

