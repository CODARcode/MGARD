set(MGARD_X_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/API.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/Array.cu
      #${CMAKE_CURRENT_SOURCE_DIR}/SubArray.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/Message.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/CompressionWorkflow.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/Handle.cu
      #${CMAKE_CURRENT_SOURCE_DIR}/MemoryManagement.cu   
      ${CMAKE_CURRENT_SOURCE_DIR}/Metadata.cpp               
      ${CMAKE_CURRENT_SOURCE_DIR}/CommonInternal.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/PrecomputeKernels.cu  
      #${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring.cu   
      # ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/DataRefactoring1D.cu  
      # ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/DataRefactoring2D.cu  
      # ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/DataRefactoring3D.cu  
      # ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/DataRefactoring4D.cu  
      # ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/DataRefactoring5D.cu   
      ${CMAKE_CURRENT_SOURCE_DIR}/DeviceAdapters/DeviceAdapterCuda.cu   
      #${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring_AMR.cu   
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_1D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_1D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_1D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_1D_Double.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_2D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_2D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_2D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_2D_Double.cu 
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_3D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_3D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_3D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_3D_Double.cu   
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_4D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_4D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_4D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_4D_Double.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_5D_Float.cu  
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Decomposition_5D_Double.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_5D_Float.cu    
      ${CMAKE_CURRENT_SOURCE_DIR}/DataRefactoring/CUDA/Recomposition_5D_Double.cu  

      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/InterpCalcCoeff.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/InterpCalcCoeff_3D.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/Interpolation.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/CalcCoefficients.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/Reorder.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/InterpCoeffRestore.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/InterpCoeffRestore_3D.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/InterpolationRev.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/CoefficientRestore.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/GridProcessingKernels/ReorderRestore.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LinearProcessingKernels/MassTrans1.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LinearProcessingKernels/MassTrans2.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LinearProcessingKernels/MassTrans3.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LinearProcessingKernels/MassTrans1_3D.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LinearProcessingKernels/MassTrans2_3D.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LinearProcessingKernels/MassTrans3_3D.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernels/SolveTridiag1.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernels/SolveTridiag2.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernels/SolveTridiag3.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernels/SolveTridiag1_3D.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernels/SolveTridiag2_3D.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernels/SolveTridiag3_3D.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LevelwiseProcessingKernels/CopyLevel.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LevelwiseProcessingKernels/AddLevel.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LevelwiseProcessingKernels/SubtractLevel.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LinearQuantization/LevelwiseLinearQuantization.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/LinearQuantization/LevelwiseLinearDequantization.cu

      #${CMAKE_CURRENT_SOURCE_DIR}/IterativeProcessingKernels/SolveTridiag1_3D_AMR.cu
      
      
      #${CMAKE_CURRENT_SOURCE_DIR}/LosslessCompression.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ErrorCalculator.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/canonical.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/constants.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/cuda_mem.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/format.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/histogram.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/Histogram.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/huffman_codec.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/huffman_workflow.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/huffman.cu
      # ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/par_huffman.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/par_huffman_sortbyfreq.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/par_merge.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ParallelHuffman/types.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/MDR/BitplaneEncoder/PerBitBPEncoderGPU.cu
      #${CMAKE_CURRENT_SOURCE_DIR}/Testing/ReorderToolsGPU.cu
      #${CMAKE_CURRENT_SOURCE_DIR}/Testing/ReorderToolsCPU.cpp
      PARENT_SCOPE)

add_subdirectory (DataRefactoring)