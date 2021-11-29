list(APPEND MGARD_X_SRC  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_1D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_1D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_1D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_1D_Double.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_2D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_2D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_2D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_2D_Double.cu 
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_3D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_3D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_3D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_3D_Double.cu   
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_4D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_4D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_4D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_4D_Double.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_5D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/Decomposition_5D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_5D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/Recomposition_5D_Double.cu
      )

set(MGARD_X_SRC ${MGARD_X_SRC} PARENT_SCOPE)