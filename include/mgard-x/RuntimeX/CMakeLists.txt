add_subdirectory(AutoTuners)
add_subdirectory(DataStructures)
add_subdirectory(Functors)
add_subdirectory(Kernel)
add_subdirectory(Tasks)
add_subdirectory(Utilities)
list(APPEND MGARD_X_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/DataTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeX.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeXPublic.h
    )
set(MGARD_X_HEADER ${MGARD_X_HEADER} PARENT_SCOPE)