#ifndef _MDR_GROUPED_BP_ENCODER_GPU_HPP
#define _MDR_GROUPED_BP_ENCODER_GPU_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

#define BINARY_TYPE BINARY
// #define BINARY_TYPE NEGABINARY

// #define DATA_ENCODING_ALGORITHM Bit_Transpose_Serial_All
#define DATA_ENCODING_ALGORITHM Bit_Transpose_Parallel_B_Serial_b
// #define DATA_ENCODING_ALGORITHM Bit_Transpose_Parallel_B_Atomic_b
// #define DATA_ENCODING_ALGORITHM Bit_Transpose_Parallel_B_Reduce_b
// #define DATA_ENCODING_ALGORITHM Bit_Transpose_Parallel_B_Ballot_b

// #define DATA_DECODING_ALGORITHM Bit_Transpose_Serial_All
#define DATA_DECODING_ALGORITHM Bit_Transpose_Parallel_B_Serial_b
// #define DATA_DECODING_ALGORITHM Bit_Transpose_Parallel_B_Atomic_b
// #define DATA_DECODING_ALGORITHM Bit_Transpose_Parallel_B_Reduce_b
// #define DATA_DECODING_ALGORITHM Bit_Transpose_Parallel_B_Ballot_b

// #define ERROR_COLLECTING_ALGORITHM Error_Collecting_Serial_All
// #define ERROR_COLLECTING_ALGORITHM
// Error_Collecting_Parallel_Bitplanes_Serial_Error #define
// ERROR_COLLECTING_ALGORITHM Error_Collecting_Parallel_Bitplanes_Atomic_Error
#define ERROR_COLLECTING_ALGORITHM                                             \
  Error_Collecting_Parallel_Bitplanes_Reduce_Error

namespace mgard_x {
namespace MDR {

template <typename T>
MGARDX_EXEC void print_bits(T v, int num_bits, bool reverse = false) {
  for (int j = 0; j < num_bits; j++) {
    if (!reverse)
      printf("%u", (v >> num_bits - 1 - j) & 1u);
    else
      printf("%u", (v >> j) & 1u);
  }
}

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, OPTION BinaryType, OPTION EncodingAlgorithm,
          OPTION ErrorColectingAlgorithm, typename DeviceType>
class GroupedEncoderFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  GroupedEncoderFunctor() {}
  MGARDX_CONT
  GroupedEncoderFunctor(SIZE n, SIZE num_batches_per_TB, SIZE num_bitplanes,
                        SIZE exp, SubArray<1, T, DeviceType> v,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes),
        num_batches_per_TB(num_batches_per_TB), exp(exp),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {
    Functor<DeviceType>();
    if (BinaryType == BINARY) {
      max_length_per_TB = num_batches_per_TB * 2;
    } else if (BinaryType == NEGABINARY) {
      max_length_per_TB = num_batches_per_TB;
    }
  }
  // exponent align
  // calculate error
  // store signs
  // find the most significant bit
  MGARDX_EXEC void Operation1() {

    debug = false;
    if (FunctorBase<DeviceType>::GetBlockIdZ() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdY() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdZ() == 0)
      debug = true;

    int8_t *sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_temp_errors = (T_error *)sm_p;
    sm_p += (num_bitplanes + 1) * num_elems_per_TB * sizeof(T_error);
    sm_errors = (T_error *)sm_p;
    sm_p += (num_bitplanes + 1) * sizeof(T_error);
    sm_fix_point = (T_fp *)sm_p;
    sm_p += num_elems_per_TB * sizeof(T_fp);
    if (BinaryType == BINARY) {
      sm_signs = (T_fp *)sm_p;
      sm_p += num_elems_per_TB * sizeof(T_fp);
    }
    sm_shifted = (T *)sm_p;
    sm_p += num_elems_per_TB * sizeof(T);
    sm_bitplanes = (T_bitplane *)sm_p;
    sm_p += (num_bitplanes + 1) * num_batches_per_TB * sizeof(T_bitplane);

    // sm_reduce =  (blockReduce_error.TempStorageType*) sm_p;
    // blockReduce_error.AllocateTempStorage();
    // thread orginal data mapping
    local_data_idx = FunctorBase<DeviceType>::GetThreadIdY() *
                         FunctorBase<DeviceType>::GetBlockDimX() +
                     FunctorBase<DeviceType>::GetThreadIdX();
    global_data_idx =
        FunctorBase<DeviceType>::GetBlockIdX() * num_elems_per_TB +
        local_data_idx;

    local_bitplane_idx = FunctorBase<DeviceType>::GetThreadIdY() *
                             FunctorBase<DeviceType>::GetBlockDimX() +
                         FunctorBase<DeviceType>::GetThreadIdX();
    // // thread bitplane mapping (transposed of data mapping for more efficient
    // ) bitplane_idx = FunctorBase<DeviceType>::GetThreadIdX() / B; block_idx =
    // FunctorBase<DeviceType>::GetThreadIdX() % B;

    // if (local_data_idx < num_elems_per_TB) {
    //   sm_fix_point[local_data_idx] = 0;
    //   sm_shifted[local_data_idx] = 0;
    //   sm_signs[local_data_idx] = 0;
    // }

    for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes + 1;
         bitplane_idx++) {
      if (local_data_idx < num_elems_per_TB) {
        sm_temp_errors[bitplane_idx * num_elems_per_TB + local_data_idx] = 0;
      }
    }

    if (local_bitplane_idx < num_bitplanes + 1) {
      sm_errors[local_bitplane_idx] = 0;
    }
  }

  MGARDX_EXEC void Operation2() {
    // __syncthreads();
    if (BinaryType == NEGABINARY)
      exp += 2;
    // convert to fixpoint data
    if (local_data_idx < num_elems_per_TB && global_data_idx < n) {
      T cur_data = *v(global_data_idx);
      T shifted_data = ldexp(cur_data, (int)num_bitplanes - (int)exp);
      T_fp fp_data;
      if (BinaryType == BINARY) {
        fp_data = (T_fp)fabs(shifted_data);
      } else if (BinaryType == NEGABINARY) {
        fp_data = Math<DeviceType>::binary2negabinary((T_sfp)shifted_data);
        // printf("2^%d %f->%u\n", (int)num_bitplanes - (int)exp, shifted_data,
        // fp_data);
      }
      // save fp_data to shared memory
      sm_fix_point[local_data_idx] = fp_data;
      sm_shifted[local_data_idx] = shifted_data;
      if (BinaryType == BINARY) {
        sm_signs[local_data_idx] = ((T_sfp)signbit(cur_data))
                                   << (sizeof(T_fp) * 8 - 1);
        // printf("data: %f, signbit(cur_data): %d, sm_signs: %llu\n", cur_data,
        // signbit(cur_data), sm_signs[local_data_idx]);
      }
      // printf("%llu, %f -> %f-> %u\n", global_data_idx, cur_data,
      // shifted_data, sm_fix_point[local_data_idx] );
      // printf("sm_fix_point[%llu]: %u\n", local_data_idx,
      // sm_fix_point[local_data_idx]);
    }
  }

  // convert fix point to bit-planes
  // level error reduction (intra block)
  MGARDX_EXEC void Operation3() {
    // data
    // BlockBitTranspose<T_fp, T_bitplane, 32, 32, 1, ALIGN_LEFT,
    //                   EncodingAlgorithm, DeviceType>
    //     blockBitTranspose;
    for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx++) {
      BlockBitTranspose<
          T_fp, T_bitplane, 32, 32, 1, ALIGN_LEFT, EncodingAlgorithm,
          DeviceType>::Transpose(sm_fix_point + batch_idx * num_elems_per_batch,
                                 sm_bitplanes + batch_idx * num_bitplanes,
                                 num_elems_per_batch, num_bitplanes,
                                 FunctorBase<DeviceType>::GetThreadIdX(),
                                 FunctorBase<DeviceType>::GetThreadIdY());
    }
    if (BinaryType == BINARY) {
      // sign
      for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx++) {
        BlockBitTranspose<
            T_fp, T_bitplane, 32, 32, 1, ALIGN_LEFT, EncodingAlgorithm,
            DeviceType>::Transpose(sm_signs + batch_idx * num_elems_per_batch,
                                   sm_bitplanes +
                                       num_batches_per_TB * num_bitplanes +
                                       batch_idx,
                                   num_elems_per_batch, 1,
                                   FunctorBase<DeviceType>::GetThreadIdX(),
                                   FunctorBase<DeviceType>::GetThreadIdY());
      }
    }
    // error
    BlockErrorCollect<
        T, T_fp, T_sfp, T_error, 32, 32, 1, ErrorColectingAlgorithm, BinaryType,
        DeviceType>::Collect(sm_shifted, sm_temp_errors, sm_errors,
                             num_elems_per_TB, num_bitplanes,
                             FunctorBase<DeviceType>::GetThreadIdX(),
                             FunctorBase<DeviceType>::GetThreadIdY());
  }

  // get max bit-plane length
  MGARDX_EXEC void Operation4() {
    // data
    block_offset = max_length_per_TB * FunctorBase<DeviceType>::GetBlockIdX();
    for (SIZE bitplane_idx = FunctorBase<DeviceType>::GetThreadIdY();
         bitplane_idx < num_bitplanes; bitplane_idx += 32) {
      for (SIZE batch_idx = FunctorBase<DeviceType>::GetThreadIdX();
           batch_idx < num_batches_per_TB; batch_idx += 32) {
        *encoded_bitplanes(bitplane_idx, block_offset + batch_idx) =
            sm_bitplanes[batch_idx * num_bitplanes + bitplane_idx];
      }
    }

    if (BinaryType == BINARY) {
      // sign
      if (local_data_idx < num_batches_per_TB) {
        *encoded_bitplanes(0,
                           block_offset + num_batches_per_TB + local_data_idx) =
            sm_bitplanes[num_batches_per_TB * num_bitplanes + local_data_idx];
      }
    }

    // error
    if (local_bitplane_idx < num_bitplanes + 1) {
      sm_errors[local_bitplane_idx] =
          ldexp(sm_errors[local_bitplane_idx], 2 * (-(int)num_bitplanes + exp));
    }

    if (local_bitplane_idx < num_bitplanes + 1) {
      *level_errors_workspace(local_bitplane_idx,
                              FunctorBase<DeviceType>::GetBlockIdX()) =
          sm_errors[local_bitplane_idx];
    }
  }

  MGARDX_EXEC void Operation5() {
    if (debug) {
      // clang-format off
      // for (int i = 0; i < num_elems_per_TB; i++) {
      //   printf("input[%u]\torg\t%f\t2^%d\tfp\t%llu:\t", i,
      //   *v(FunctorBase<DeviceType>::GetBlockIdX()*num_elems_per_TB+i),
      //   (int)num_bitplanes - (int)exp, sm_fix_point[i]);
      //   print_bits(sm_fix_point[i], num_bitplanes);
      //   printf("\n");
      // }

      // for (int i = 0; i < num_elems_per_TB; i++) {
      //   printf("sm_signs[%u]\t", i);
      //   print_bits(sm_signs[i], sizeof(T_fp)*8);
      //   printf("\n");
      // }

      // for (int i = 0; i < num_bitplanes; i++) {
      //   printf("sm_bitplane %d: ", i);
      //   for (int j = 0; j < num_batches_per_TB; j++) {
      //     printf("\t%u:\t", sm_bitplanes[j * num_bitplanes + i]);
      //     print_bits(sm_bitplanes[j * num_bitplanes + i], sizeof(T_bitplane)*8, false);

      //   }
      //   printf("\n");
      // }

      // for (int j = 0; j < num_batches_per_TB; j++) {
      //   printf("sm_bitplane_sign[%d]: ", j);
      //   printf("\t%u:\t", sm_bitplanes[num_batches_per_TB * num_bitplanes + j]); 
      //   print_bits(sm_bitplanes[num_batches_per_TB * num_bitplanes + j], sizeof(T_bitplane)*8, false); 
      //   printf("\n");
      // }

      // for (int i = 0; i < num_bitplanes; i++) {
      //   printf("bitplane %d: ", i);
      //   for (int j = 0; j < num_batches_per_TB; j++) {
      //     printf("\t%u:\t", *encoded_bitplanes(i, block_offset + j));
      //     print_bits(*encoded_bitplanes(i, block_offset + j), sizeof(T_bitplane)*8, false);
      //   }
      //   printf("\n");
      // }

      // for (int i = 0; i < num_batches_per_TB; i ++) {
      //   printf("sign %d: ", i);
      //   printf("\t%u:\t", *encoded_bitplanes(0, block_offset + num_batches_per_TB + i)); 
      //   print_bits(*encoded_bitplanes(0, block_offset + num_batches_per_TB + i), sizeof(T_bitplane)*8, false);
      //   printf("\n");
      // }

      // for (int i = 0; i < num_bitplanes + 1; i++) {
      //   printf("error %d/%d: ", i, num_bitplanes + 1);
      //   printf (" %f ", sm_errors[i]);
      //   printf("\n");
      // }
      // clang-format on
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += (num_bitplanes + 1) * num_elems_per_TB * sizeof(T_error);
    size += (num_bitplanes + 1) * sizeof(T_error);
    size += num_elems_per_TB * sizeof(T_fp);
    size += (num_bitplanes + 1) * num_batches_per_TB * sizeof(T_bitplane);
    size += num_elems_per_TB * sizeof(T);
    if (BinaryType == BINARY) {
      size += num_elems_per_TB * sizeof(T_fp);
    }
    // printf("shared_memory_size: %u\n", size);
    return size;
  }

private:
  // parameters
  SIZE n;
  SIZE num_batches_per_TB;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<1, T, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;

  // stateful thread local variables

  bool debug;
  IDX local_data_idx, global_data_idx, local_bitplane_idx;

  SIZE num_elems_per_batch = sizeof(T_bitplane) * 8;
  SIZE num_elems_per_TB = num_elems_per_batch * num_batches_per_TB;
  SIZE max_length_per_TB;
  SIZE block_offset;
  T_error *sm_temp_errors;
  T_error *sm_errors;
  T_fp *sm_fix_point;
  T *sm_shifted;
  T_bitplane *sm_bitplanes;
  T_fp *sm_signs;
};

template <typename T, typename T_bitplane, typename T_error, OPTION BinaryType,
          OPTION EncodingAlgorithm, OPTION ErrorColectingAlgorithm,
          typename DeviceType>
class GroupedEncoderKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp encoder";
  MGARDX_CONT
  GroupedEncoderKernel(SIZE n, SIZE num_batches_per_TB, SIZE num_bitplanes,
                       SIZE exp, SubArray<1, T, DeviceType> v,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes),
        num_batches_per_TB(num_batches_per_TB), exp(exp),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;
  using FunctorType =
      GroupedEncoderFunctor<T, T_fp, T_sfp, T_bitplane, T_error, BinaryType,
                            EncodingAlgorithm, ErrorColectingAlgorithm,
                            DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, num_batches_per_TB, num_bitplanes, exp, v,
                        encoded_bitplanes, level_errors_workspace);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
    tbz = 1;
    tby = 32;
    tbx = 32;
    gridz = 1;
    gridy = 1;
    gridx = (n - 1) / num_elems_per_TB + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  SIZE num_batches_per_TB;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<1, T, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
};

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
          OPTION BinaryType, OPTION DecodingAlgorithm, typename DeviceType>
class GroupedDecoderFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  GroupedDecoderFunctor() {}
  MGARDX_CONT
  GroupedDecoderFunctor(SIZE n, SIZE num_batches_per_TB, SIZE starting_bitplane,
                        SIZE num_bitplanes, SIZE exp,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<1, bool, DeviceType> signs,
                        SubArray<1, T, DeviceType> v)
      : n(n), num_batches_per_TB(num_batches_per_TB),
        starting_bitplane(starting_bitplane), num_bitplanes(num_bitplanes),
        exp(exp), encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
    if (BinaryType == BINARY) {
      max_length_per_TB = num_batches_per_TB * 2;
    } else if (BinaryType == NEGABINARY) {
      max_length_per_TB = num_batches_per_TB;
    }
  }

  // exponent align
  // store signs
  // find the most significant bit
  MGARDX_EXEC void Operation1() {
    debug = false;
    if (FunctorBase<DeviceType>::GetBlockIdZ() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdY() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdZ() == 0)
      debug = true;

    debug2 = false;
    if (FunctorBase<DeviceType>::GetBlockIdZ() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdX() == 0)
      debug2 = true;

    int8_t *sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_fix_point = (T_fp *)sm_p;
    sm_p += num_elems_per_TB * sizeof(T_fp);
    if (BinaryType == BINARY) {
      sm_signs = (T_fp *)sm_p;
      sm_p += num_elems_per_TB * sizeof(T_fp);
    }
    sm_bitplanes = (T_bitplane *)sm_p;
    sm_p += num_batches_per_TB * (num_bitplanes + 1) * sizeof(T_bitplane);

    local_data_idx = FunctorBase<DeviceType>::GetThreadIdY() *
                         FunctorBase<DeviceType>::GetBlockDimX() +
                     FunctorBase<DeviceType>::GetThreadIdX();
    global_data_idx =
        FunctorBase<DeviceType>::GetBlockIdX() * num_elems_per_TB +
        local_data_idx;

    ending_bitplane = starting_bitplane + num_bitplanes;

    if (BinaryType == NEGABINARY)
      exp += 2;
    // data
    block_offset = max_length_per_TB * FunctorBase<DeviceType>::GetBlockIdX();
    for (SIZE bitplane_idx = FunctorBase<DeviceType>::GetThreadIdY();
         bitplane_idx < num_bitplanes; bitplane_idx += 32) {
      for (SIZE batch_idx = FunctorBase<DeviceType>::GetThreadIdX();
           batch_idx < num_batches_per_TB; batch_idx += 32) {
        sm_bitplanes[batch_idx * num_bitplanes + bitplane_idx] =
            *encoded_bitplanes(bitplane_idx + starting_bitplane,
                               block_offset + batch_idx);
      }
    }

    if (BinaryType == BINARY) {
      // sign
      sign = 0; // 0: positive
                // 1: negative
      if (starting_bitplane == 0) {
        if (local_data_idx < num_batches_per_TB) {
          sm_bitplanes[num_batches_per_TB * num_bitplanes + local_data_idx] =
              *encoded_bitplanes(0, block_offset + num_batches_per_TB +
                                        local_data_idx);
        }
      } else {
        if (local_data_idx < num_elems_per_TB && global_data_idx < n) {
          sm_signs[local_data_idx] = *signs(global_data_idx);
        }
      }
    }
  }

  // convert fix point to bit-planes
  // level error reduction (intra block)
  MGARDX_EXEC void Operation2() {
    // data
    // BlockBitTranspose<T_bitplane, T_fp, 32, 32, 1, ALIGN_RIGHT,
    // DecodingAlgorithm, DeviceType> blockBitTranspose;
    for (SIZE i = 0; i < num_batches_per_TB; i++) {
      BlockBitTranspose<
          T_bitplane, T_fp, 32, 32, 1, ALIGN_RIGHT, DecodingAlgorithm,
          DeviceType>::Transpose(sm_bitplanes + i * num_bitplanes,
                                 sm_fix_point + i * num_elems_per_batch,
                                 num_bitplanes, num_elems_per_batch,
                                 FunctorBase<DeviceType>::GetThreadIdX(),
                                 FunctorBase<DeviceType>::GetThreadIdY());
    }

    if (BinaryType == BINARY) {
      // sign
      if (starting_bitplane == 0) {
        for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx++) {
          BlockBitTranspose<
              T_bitplane, T_fp, 32, 32, 1, ALIGN_RIGHT, DecodingAlgorithm,
              DeviceType>::Transpose(sm_bitplanes +
                                         num_batches_per_TB * num_bitplanes +
                                         batch_idx,
                                     sm_signs + batch_idx * num_elems_per_batch,
                                     1, num_elems_per_batch,
                                     FunctorBase<DeviceType>::GetThreadIdX(),
                                     FunctorBase<DeviceType>::GetThreadIdY());
        }
      }
    }

    // // decoding
    // T_fp bit;
    // bool sign;
    // SIZE encoding_block_idx = local_data_idx/(sizeof(T_bitplane)*8);
    // SIZE encoding_bit_idx = local_data_idx%(sizeof(T_bitplane)*8);
    // T_fp fp_data = 0;
    // if (local_data_idx < num_elems_per_TB) {
    //   for (SIZE i = 0; i < num_bitplanes; i++) {
    //     bit = (sm_bitplanes[encoding_block_idx*num_bitplanes + i] >>
    //     encoding_bit_idx) & 1u ; fp_data += bit << num_bitplanes - 1 - i;
    //   }

    //   if (starting_bitplane == 0) {
    //     // decoding signs (total B blocks)
    //       T_bitplane sign_bitplane = *encoded_bitplanes(0, block_offset + B +
    //       encoding_block_idx); sign = (sign_bitplane >> encoding_bit_idx) &
    //       1u; if (encoding_bit_idx == 0) {
    //         *signs(local_data_idx) = sign;
    //       }
    //   } else {
    //     sign = *signs(local_data_idx);
    //   }
    // }

    // T cur_data = ldexp((T)fp_data, - ending_bitplane + exp);

    // // if (debug) printf("fp[%llu]: %u -> 2^%u %f\n", local_data_idx,
    // fp_data, - ending_bitplane + exp, cur_data); *v(local_data_idx) = sign ?
    // -cur_data : cur_data;
  }

  // store bit-plane
  MGARDX_EXEC void Operation3() {
    if (local_data_idx < num_elems_per_TB) {
      T_fp fp_data = sm_fix_point[local_data_idx];
      if (BinaryType == BINARY) {
        T cur_data = ldexp((T)fp_data, -ending_bitplane + exp);
        if (global_data_idx < n) {
          *v(global_data_idx) = sm_signs[local_data_idx] ? -cur_data : cur_data;
          *signs(global_data_idx) = sm_signs[local_data_idx];
        }
      } else if (BinaryType == NEGABINARY) {
        T cur_data = ldexp((T)Math<DeviceType>::negabinary2binary(fp_data),
                           -ending_bitplane + exp);
        if (global_data_idx < n) {
          *v(global_data_idx) = ending_bitplane % 2 != 0 ? -cur_data : cur_data;
        }
      }
    }
  }

  MGARDX_EXEC void Operation4() {

    if (debug) {
      // for (int i = 0; i < num_bitplanes; i++) {
      //   printf("decode bitpane[%d]: ", i);
      //   for (int j = 0; j < num_batches_per_TB; j++) {
      //     printf(" %u  ", sm_bitplanes[j*num_bitplanes+i]);
      //     // for (int k = 0; k < B; k++) {
      //     //   printf("%u", (sm_bitplanes[j*B+i] >> B-1-k) & 1u);
      //     // }
      //   }
      //   printf("\n");
      // }
      // printf("\n");
    }

    // if (debug) {
    //   printf("sm_signs: ");
    //   for (int i = 0; i < num_elems_per_TB; i++) {
    //     printf("%u ,", sm_signs[i]);
    //   }
    //   printf("\n");
    // }

    // if (debug) {
    //   printf("decoded data:\t");
    //   for (int i = 0; i < num_elems_per_TB; i++) {
    //     printf("%f\t", *v(FunctorBase<DeviceType>::GetBlockIdX() *
    //     num_elems_per_TB + i));
    //   }
    //   printf("\n");
    // }
  }

  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += num_batches_per_TB * (num_bitplanes + 1) * sizeof(T_bitplane);
    size += num_elems_per_TB * sizeof(T_fp);
    if (BinaryType == BINARY) {
      size += num_elems_per_TB * sizeof(T_fp);
    }
    return size;
  }

private:
  // parameters
  SIZE n;
  SIZE num_batches_per_TB;
  SIZE starting_bitplane;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T, DeviceType> v;

  // stateful thread local variables
  bool debug, debug2;
  IDX local_data_idx, global_data_idx;

  SIZE num_elems_per_batch = sizeof(T_bitplane) * 8;
  SIZE num_elems_per_TB = num_elems_per_batch * num_batches_per_TB;
  SIZE max_length_per_TB;
  SIZE block_offset;
  SIZE ending_bitplane;
  SIZE bitplane_max_length;
  T_bitplane *sm_bitplanes;
  T_fp *sm_fix_point;
  bool sign;
  T_fp *sm_signs;
};

template <typename T, typename T_bitplane, OPTION BinaryType,
          OPTION DecodingAlgorithm, typename DeviceType>
class GroupedDecoderKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp decoder";
  MGARDX_CONT
  GroupedDecoderKernel(SIZE n, SIZE num_batches_per_TB, SIZE starting_bitplane,
                       SIZE num_bitplanes, SIZE exp,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<1, bool, DeviceType> signs,
                       SubArray<1, T, DeviceType> v)
      : n(n), num_batches_per_TB(num_batches_per_TB),
        starting_bitplane(starting_bitplane), num_bitplanes(num_bitplanes),
        exp(exp), encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;
  using FunctorType =
      GroupedDecoderFunctor<T, T_fp, T_sfp, T_bitplane, BinaryType,
                            DecodingAlgorithm, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {

    FunctorType functor(n, num_batches_per_TB, starting_bitplane, num_bitplanes,
                        exp, encoded_bitplanes, signs, v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
    tbz = 1;
    tby = 32;
    tbx = 32;
    gridz = 1;
    gridy = 1;
    gridx = (n - 1) / num_elems_per_TB + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  SIZE num_batches_per_TB;
  SIZE starting_bitplane;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T, DeviceType> v;
};

// general bitplane encoder that encodes data by block using T_stream type
// buffer
template <DIM D, typename T_data, typename T_bitplane, typename T_error,
          typename DeviceType>
class GroupedBPEncoder
    : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                DeviceType> {
public:
  GroupedBPEncoder(Hierarchy<D, T_data, DeviceType> hierarchy)
      : hierarchy(hierarchy) {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");

    // std::vector<SIZE> level_num_elems(hierarchy.l_target() + 1);
    // SIZE prev_num_elems = 0;
    // for (int level_idx = 0; level_idx < hierarchy.l_target() + 1;
    // level_idx++) {
    //   SIZE curr_num_elems = 1;
    //   for (DIM d = 0; d < D; d++) {
    //     curr_num_elems *= hierarchy.level_shape(level_idx, d);
    //   }
    //   level_num_elems[level_idx] = curr_num_elems - prev_num_elems;
    //   prev_num_elems = curr_num_elems;
    //   // printf("%u ", level_num_elems[level_idx]);
    // }
    SIZE max_level_num_elems = hierarchy.level_num_elems(hierarchy.l_target());

    SIZE max_bitplane = 64;
    level_errors_work_array = Array<2, T_error, DeviceType>(
        {max_bitplane + 1, num_blocks(max_level_num_elems)});
    DeviceCollective<DeviceType>::Sum(num_blocks(max_level_num_elems),
                                      SubArray<1, T_error, DeviceType>(),
                                      SubArray<1, T_error, DeviceType>(),
                                      level_error_sum_work_array, false, 0);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    Hierarchy<D, T_data, DeviceType> hierarchy(shape, Config());
    SIZE max_level_num_elems = hierarchy.level_num_elems(hierarchy.l_target());
    SIZE max_bitplane = 64;
    size_t size = 0;
    size += hierarchy.EstimateMemoryFootprint(shape);
    size +=
        (max_bitplane + 1) * num_blocks(max_level_num_elems) * sizeof(T_error);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += hierarchy.level_num_elems(level_idx) * sizeof(bool);
    }
    return size;
  }

  void encode(SIZE n, SIZE num_bitplanes, int32_t exp,
              SubArray<1, T_data, DeviceType> v,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
              SubArray<1, T_error, DeviceType> level_errors,
              std::vector<SIZE> &streams_sizes, int queue_idx) {

    SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

    DeviceLauncher<DeviceType>::Execute(
        GroupedEncoderKernel<T_data, T_bitplane, T_error, BINARY_TYPE,
                             DATA_ENCODING_ALGORITHM,
                             ERROR_COLLECTING_ALGORITHM, DeviceType>(
            n, num_batches_per_TB, num_bitplanes, exp, v, encoded_bitplanes,
            level_errors_work),
        queue_idx);

    SIZE reduce_size = num_blocks(n);
    for (int i = 0; i < num_bitplanes + 1; i++) {
      SubArray<1, T_error, DeviceType> curr_errors({reduce_size},
                                                   level_errors_work(i, 0));
      SubArray<1, T_error, DeviceType> sum_error({1}, level_errors(i));
      DeviceCollective<DeviceType>::Sum(reduce_size, curr_errors, sum_error,
                                        level_error_sum_work_array, true,
                                        queue_idx);
    }
    for (int i = 0; i < num_bitplanes; i++) {
      streams_sizes[i] = buffer_size(n) * sizeof(T_bitplane);
    }
  }

  void decode(SIZE n, SIZE num_bitplanes, int32_t exp,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes, int level,
              SubArray<1, T_data, DeviceType> v, int queue_idx) {}

  // decode the data and record necessary information for progressiveness
  void progressive_decode(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes,
                          int32_t exp,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> level_signs, int level,
                          SubArray<1, T_data, DeviceType> v, int queue_idx) {
    if (num_bitplanes > 0) {
      DeviceLauncher<DeviceType>::Execute(
          GroupedDecoderKernel<T_data, T_bitplane, BINARY_TYPE,
                               DATA_DECODING_ALGORITHM, DeviceType>(
              n, num_batches_per_TB, starting_bitplane, num_bitplanes, exp,
              encoded_bitplanes, level_signs, v),
          queue_idx);
    }
  }

  static SIZE buffer_size(SIZE n) {
    const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    SIZE num_blocks = (n - 1) / num_elems_per_TB + 1;
    SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_blocks;
    return bitplane_max_length_total;
  }

  static SIZE num_blocks(SIZE n) {
    const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    SIZE num_blocks = (n - 1) / num_elems_per_TB + 1;
    return num_blocks;
  }

  void print() const { std::cout << "Grouped bitplane encoder" << std::endl; }

private:
  Hierarchy<D, T_data, DeviceType> hierarchy;
  static constexpr SIZE num_batches_per_TB = 2;
  Array<2, T_error, DeviceType> level_errors_work_array;
  Array<1, Byte, DeviceType> level_error_sum_work_array;
  std::vector<std::vector<uint8_t>> level_recording_bitplanes;
};
} // namespace MDR
} // namespace mgard_x
#endif
