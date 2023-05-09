add_subdirectory(MultiDimension)
add_subdirectory(SingleDimension)
list(APPEND MGARD_X_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactorInterface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HybridHierarchyDataRefactor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HybridHierarchyDataRefactorInterface.hpp
    )
set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)