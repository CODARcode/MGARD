set(_MGARD_X_TEST_DIR "${CMAKE_CURRENT_LIST_DIR}")

if(NOT Catch2_FOUND)
  message(STATUS "Will not build mgard-x tests (Catch2 not found).")
  return()
endif()

if(NOT DEVICE_TYPE_LIST)
  message(STATUS "Will not build mgard-x tests (no device backend enabled).")
  return()
endif()

# Generate one test source file from a .cpp.in template for the given device type,
# set its compile language, and append it to MGARD_X_TEST_SOURCES.
macro(mgard_x_add_test SRC_NAME DEVICE_TYPE)
  set(_src "${_MGARD_X_TEST_DIR}/src/${SRC_NAME}.cpp.in")
  set(_gen "${CMAKE_CURRENT_BINARY_DIR}/src/${SRC_NAME}_${DEVICE_TYPE}.cpp")
  configure_file(${_src} ${_gen})
  if(${DEVICE_TYPE} STREQUAL "CUDA")
    set_source_files_properties(${_gen} PROPERTIES LANGUAGE CUDA)
  elseif(${DEVICE_TYPE} STREQUAL "HIP")
    set_source_files_properties(${_gen} PROPERTIES LANGUAGE HIP)
  else()
    set_source_files_properties(${_gen} PROPERTIES LANGUAGE CXX)
  endif()
  list(APPEND MGARD_X_TEST_SOURCES ${_gen})
endmacro()

# BPEncoderRegisterBlock's progressive_decode() unconditionally macro-unrolls
# 64 kernel variants per (T_data, T_bitplane, NegaBinary, ControlL2)
# configuration it is instantiated with. Templating across all of them in one
# translation unit (as test_Compressor etc. do for their own axes) overwhelms
# the CUDA/HIP device compiler, so each configuration is generated into its
# own small file instead -- the same reason the library itself generates
# e.g. Compressor_1D_double_CUDA.cpp per type/dim combination rather than
# templating across them.
set(BP_ENCODER_LIST BPEncoderLocalityBlock BPEncoderRegisterBlock)
set(BP_DATA_TYPE_LIST float double)
set(BP_BITPLANE_TYPE_LIST uint32_t uint64_t)
set(BP_BOOL_LIST true false)

macro(mgard_x_add_bitplane_encoder_tests DEVICE_TYPE)
  foreach(ENCODER_NAME IN LISTS BP_ENCODER_LIST)
    # BPEncoderLocalityBlock has no batch-alignment requirement on n (its
    # encode/decode loops guard every element access with `< n`), so it is
    # deliberately tested with a size that is not a multiple of either
    # bitplane word's batch size (32 or 64). BPEncoderRegisterBlock requires n
    # to be a multiple of its batch size; 4096 is a multiple of both.
    if(ENCODER_NAME STREQUAL "BPEncoderLocalityBlock")
      set(BP_TEST_N 4001)
    else()
      set(BP_TEST_N 4096)
    endif()
    foreach(T_DATA IN LISTS BP_DATA_TYPE_LIST)
      foreach(T_BITPLANE IN LISTS BP_BITPLANE_TYPE_LIST)
        foreach(NEGA_BINARY IN LISTS BP_BOOL_LIST)
          foreach(COLLECT_ERROR IN LISTS BP_BOOL_LIST)
            set(_src "${_MGARD_X_TEST_DIR}/src/test_BitplaneEncoderConfig.cpp.in")
            set(_tag "${ENCODER_NAME}_${T_DATA}_${T_BITPLANE}_nega${NEGA_BINARY}_l2${COLLECT_ERROR}")
            set(_gen "${CMAKE_CURRENT_BINARY_DIR}/src/test_BitplaneEncoder_${_tag}_${DEVICE_TYPE}.cpp")
            configure_file(${_src} ${_gen})
            if(${DEVICE_TYPE} STREQUAL "CUDA")
              set_source_files_properties(${_gen} PROPERTIES LANGUAGE CUDA)
            elseif(${DEVICE_TYPE} STREQUAL "HIP")
              set_source_files_properties(${_gen} PROPERTIES LANGUAGE HIP)
            else()
              set_source_files_properties(${_gen} PROPERTIES LANGUAGE CXX)
            endif()
            list(APPEND MGARD_X_TEST_SOURCES ${_gen})
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
  mgard_x_add_test(test_BitplaneEncoderRegisterBlockThrows ${DEVICE_TYPE})
endmacro()

# Same reasoning as mgard_x_add_bitplane_encoder_tests: ComposedRefactor/
# Reconstructor pull in BPEncoderRegisterBlock's 64-way macro-unrolled decode
# kernels too, so each (D, T_data, Basis, ControlL2, NegaBinary) configuration
# gets its own file. Hierarchical + ControlL2 is skipped: L2 error tracking
# needs an orthogonal basis to be meaningful.
set(CR_D_LIST 1 2 3)
set(CR_T_LIST float double)
set(CR_BASIS_LIST Hierarchical Orthogonal)
set(CR_BOOL_LIST true false)

macro(mgard_x_add_composed_refactor_tests DEVICE_TYPE)
  foreach(CR_D IN LISTS CR_D_LIST)
    if(CR_D STREQUAL "1")
      set(CR_SHAPE "{4097}")
    elseif(CR_D STREQUAL "2")
      set(CR_SHAPE "{129, 129}")
    else()
      set(CR_SHAPE "{33, 33, 33}")
    endif()
    foreach(CR_T IN LISTS CR_T_LIST)
      foreach(CR_BASIS IN LISTS CR_BASIS_LIST)
        foreach(CR_CONTROL_L2 IN LISTS CR_BOOL_LIST)
          if(NOT (CR_BASIS STREQUAL "Hierarchical" AND CR_CONTROL_L2 STREQUAL "true"))
            foreach(CR_NEGA_BINARY IN LISTS CR_BOOL_LIST)
              set(_src "${_MGARD_X_TEST_DIR}/src/test_ComposedRefactorReconstructor.cpp.in")
              set(_tag "${CR_D}D_${CR_T}_${CR_BASIS}_l2${CR_CONTROL_L2}_nega${CR_NEGA_BINARY}")
              set(_gen "${CMAKE_CURRENT_BINARY_DIR}/src/test_ComposedRefactorReconstructor_${_tag}_${DEVICE_TYPE}.cpp")
              configure_file(${_src} ${_gen})
              if(${DEVICE_TYPE} STREQUAL "CUDA")
                set_source_files_properties(${_gen} PROPERTIES LANGUAGE CUDA)
              elseif(${DEVICE_TYPE} STREQUAL "HIP")
                set_source_files_properties(${_gen} PROPERTIES LANGUAGE HIP)
              else()
                set_source_files_properties(${_gen} PROPERTIES LANGUAGE CXX)
              endif()
              list(APPEND MGARD_X_TEST_SOURCES ${_gen})
            endforeach()
          endif()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endmacro()

foreach(DEVICE_TYPE IN LISTS DEVICE_TYPE_LIST)
  set(MGARD_X_TEST_SOURCES "")

  mgard_x_add_test(test_Huffman         ${DEVICE_TYPE})
  mgard_x_add_test(test_BlockDelta      ${DEVICE_TYPE})
  mgard_x_add_test(test_DataRefactoring ${DEVICE_TYPE})
  mgard_x_add_test(test_HybridHierarchy ${DEVICE_TYPE})
  mgard_x_add_test(test_HybridMetadata  ${DEVICE_TYPE})
  mgard_x_add_test(test_LevelCapMetadata ${DEVICE_TYPE})
  mgard_x_add_test(test_LevelCapRoundTrip ${DEVICE_TYPE})
  mgard_x_add_test(test_KernelFusionSwitch ${DEVICE_TYPE})
  mgard_x_add_test(test_Compressor      ${DEVICE_TYPE})
  mgard_x_add_test(test_Zstd            ${DEVICE_TYPE})
  mgard_x_add_test(test_RLE             ${DEVICE_TYPE})
  mgard_x_add_test(test_ZeroRLE         ${DEVICE_TYPE})
  mgard_x_add_bitplane_encoder_tests(${DEVICE_TYPE})
  mgard_x_add_test(test_DirectInterleaver ${DEVICE_TYPE})
  mgard_x_add_test(test_MGARDDecomposer ${DEVICE_TYPE})
  mgard_x_add_test(test_HybridLevelCompressor ${DEVICE_TYPE})
  mgard_x_add_composed_refactor_tests(${DEVICE_TYPE})
  mgard_x_add_test(test_Rans            ${DEVICE_TYPE})
  mgard_x_add_test(test_ComposedHuffmanZstd ${DEVICE_TYPE})

  # Portable (nvcomp-free) LZ4 backend: available on every device type.
  mgard_x_add_test(test_LZ4               ${DEVICE_TYPE})
  mgard_x_add_test(test_ComposedHuffmanLZ4 ${DEVICE_TYPE})

  set(TEST_TARGET "mgard-x-tests-${DEVICE_TYPE}")
  add_executable(${TEST_TARGET} ${MGARD_X_TEST_SOURCES})
  target_link_libraries(${TEST_TARGET} PRIVATE mgard-library Catch2::Catch2WithMain)

  string(TOLOWER "${DEVICE_TYPE}" DEVICE_TYPE_LOWER)
  catch_discover_tests(${TEST_TARGET} TEST_PREFIX "mgard-x/${DEVICE_TYPE_LOWER}/")
endforeach()
