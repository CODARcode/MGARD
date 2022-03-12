list(APPEND MGARD_X_HIP_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_1D_Float.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_1D_Double.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_1D_Float.cpp    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_1D_Double.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_2D_Float.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_2D_Double.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_2D_Float.cpp    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_2D_Double.cpp 
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_3D_Float.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_3D_Double.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_3D_Float.cpp    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_3D_Double.cpp   
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_4D_Float.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_4D_Double.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_4D_Float.cpp    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_4D_Double.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_5D_Float.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_5D_Double.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_5D_Float.cpp    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_5D_Double.cpp
      )

set(MGARD_X_HIP_SRC ${MGARD_X_HIP_SRC} PARENT_SCOPE)