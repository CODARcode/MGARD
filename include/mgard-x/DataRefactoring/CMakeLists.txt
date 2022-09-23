add_subdirectory(MultiDimension)
add_subdirectory(SingleDimension)
list(APPEND MGARD_X_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoringWorkspace.hpp
    )
set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)