#ifndef _MDR_GROUPED_BP_ENCODER_GPU_HPP
#define _MDR_GROUPED_BP_ENCODER_GPU_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

#define BINARY_TYPE BINARY

#define DATA_ENCODING_ALGORITHM Bit_Transpose_Serial_All
// #define DATA_ENCODING_ALGORITHM Bit_Transpose_Parallel_B_Serial_b
// #define DATA_ENCODING_ALGORITHM Bit_Transpose_Parallel_B_Atomic_b
// #define DATA_ENCODING_ALGORITHM Bit_Transpose_Parallel_B_Reduce_b
// #define DATA_ENCODING_ALGORITHM Bit_Transpose_Parallel_B_Ballot_b

#define DATA_DECODING_ALGORITHM Bit_Transpose_Serial_All
// #define DATA_DECODING_ALGORITHM Bit_Transpose_Parallel_B_Serial_b
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

    __syncthreads();
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
        fp_data = binary2negabinary((T_sfp)shifted_data);
        // printf("2^%d %f->%u\n", (int)num_bitplanes - (int)exp, shifted_data,
        // fp_data);
      }
      // save fp_data to shared memory
      sm_fix_point[local_data_idx] = fp_data;
      sm_shifted[local_data_idx] = shifted_data;
      if (BinaryType == BINARY) {
        sm_signs[local_data_idx] = signbit(cur_data) << (sizeof(T_fp) * 8 - 1);
      }
      // printf("%llu, %f -> %f-> %u\n", global_data_idx, cur_data,
      // shifted_data, sm_fix_point[local_data_idx] );
      // printf("sm_fix_point[%llu]: %u\n", local_data_idx,
      // sm_fix_point[local_data_idx]);
    }
  }

  // convert fix point to bit-planes
  // level error reduction (intra block)
  MGARDX_EXEC void Operation2() {
    // data
    BlockBitTranspose<T_fp, T_bitplane, 32, 32, 1, ALIGN_LEFT,
                      EncodingAlgorithm, DeviceType>
        blockBitTranspose;
    for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx++) {
      blockBitTranspose.Transpose(sm_fix_point +
                                      batch_idx * num_elems_per_batch,
                                  sm_bitplanes + batch_idx * num_bitplanes,
                                  num_elems_per_batch, num_bitplanes);
    }
    if (BinaryType == BINARY) {
      // sign
      for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx++) {
        blockBitTranspose.Transpose(
            sm_signs + batch_idx * num_elems_per_batch,
            sm_bitplanes + num_batches_per_TB * num_bitplanes + batch_idx,
            num_elems_per_batch, 1);
      }
    }
    // error
    ErrorCollect<T, T_fp, T_sfp, T_error, 32, 32, 1, ErrorColectingAlgorithm,
                 BinaryType, DeviceType>()
        .Collect(sm_shifted, sm_temp_errors, sm_errors, num_elems_per_TB,
                 num_bitplanes);
  }

  // get max bit-plane length
  MGARDX_EXEC void Operation3() {
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

  MGARDX_EXEC void Operation4() {
    if (debug) {
      // for (int i = 0; i < num_elems_per_TB; i++) {
      //   printf("input[%u]\torg\t%f\t2^%d\tfp\t%llu:\t", i,
      //   *v(FunctorBase<DeviceType>::GetBlockIdX()*num_elems_per_TB+i),
      //   (int)num_bitplanes - (int)exp, sm_fix_point[i]);
      //   print_bits(sm_fix_point[i], num_bitplanes);
      //   printf("\n");
      // }

      // for (int i = 0; i < num_bitplanes; i++) {
      //   printf("sm_bitplane %d: ", i);
      //   for (int j = 0; j < num_batches_per_TB; j++) {
      //     printf("\t%u:\t", sm_bitplanes[j * num_bitplanes + i]);
      //     print_bits(sm_bitplanes[j * num_bitplanes + i],
      //     sizeof(T_bitplane)*8, false);

      //   }
      //   printf("\n");
      // }

      // for (int j = 0; j < num_batches_per_TB; j++) {
      //   printf("sm_bitplane_sign[%d]: ", j);
      //   printf("\t%u:\t", sm_bitplanes[num_batches_per_TB * num_bitplanes +
      //   j]); print_bits(sm_bitplanes[num_batches_per_TB * num_bitplanes + j],
      //   sizeof(T_bitplane)*8, false); printf("\n");
      // }

      // for (int i = 0; i < num_bitplanes; i++) {
      //   printf("bitplane %d: ", i);
      //   for (int j = 0; j < num_batches_per_TB; j++) {
      //     printf("\t%u:\t", *encoded_bitplanes(i, block_offset + j));
      //     print_bits(*encoded_bitplanes(i, block_offset + j),
      //     sizeof(T_bitplane)*8, false);

      //   }
      //   printf("\n");
      // }

      // for (int i = 0; i < num_batches_per_TB; i ++) {
      //   printf("sign %d: ", i);
      //   printf("\t%u:\t", *encoded_bitplanes(0, block_offset +
      //   num_batches_per_TB + i)); print_bits(*encoded_bitplanes(0,
      //   block_offset + num_batches_per_TB + i), sizeof(T_bitplane)*8, false);
      //   printf("\n");
      // }

      // for (int i = 0; i < num_bitplanes + 1; i++) {
      //   printf("error %d/%d: ", i, num_bitplanes + 1);
      //     printf (" %f ", sm_errors[i]);
      //   printf("\n");
      // }
    }
  }

  MGARDX_EXEC void Operation5() {}

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
class GroupedEncoder : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GroupedEncoder() : AutoTuner<DeviceType>() {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;
  using FunctorType =
      GroupedEncoderFunctor<T, T_fp, T_sfp, T_bitplane, T_error, BinaryType,
                            EncodingAlgorithm, ErrorColectingAlgorithm,
                            DeviceType>;
  using TaskType = Task<FunctorType>;

  template <typename T_fp, typename T_sfp>
  MGARDX_CONT TaskType GenTask(
      SIZE n, SIZE num_batches_per_TB, SIZE num_bitplanes, SIZE exp,
      SubArray<1, T, DeviceType> v,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<2, T_error, DeviceType> level_errors_workspace, int queue_idx) {
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
    // printf("GroupedEncoder config(%u %u %u) (%u %u %u), sm_size: %llu\n",
    // tbx, tby, tbz, gridx, gridy, gridz, sm_size);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  MGARDX_CONT
  void Execute(SIZE n, SIZE num_batches_per_TB, SIZE num_bitplanes, SIZE exp,
               SubArray<1, T, DeviceType> v,
               SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
               SubArray<1, T_error, DeviceType> level_errors,
               SubArray<2, T_error, DeviceType> level_errors_workspace,
               int queue_idx) {

    // PrintSubarray("v", v);
    TaskType task = GenTask<T_fp, T_sfp>(n, num_batches_per_TB, num_bitplanes,
                                         exp, v, encoded_bitplanes,
                                         level_errors_workspace, queue_idx);
    DeviceAdapter<TaskType, DeviceType> adapter;
    adapter.Execute(task);
    DeviceRuntime<DeviceType>().SyncQueue(queue_idx);
    // this->handle.sync_all();
    // PrintSubarray("level_errors_workspace", level_errors_workspace);
    // get level error
    const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
    SIZE reduce_size = (n - 1) / num_elems_per_TB + 1;
    DeviceCollective<DeviceType> deviceReduce;
    for (int i = 0; i < num_bitplanes + 1; i++) {
      SubArray<1, T_error, DeviceType> curr_errors(
          {reduce_size}, level_errors_workspace(i, 0));
      SubArray<1, T_error, DeviceType> sum_error({1}, level_errors(i));
      deviceReduce.Sum(reduce_size, curr_errors, sum_error, queue_idx);
    }
    DeviceRuntime<DeviceType>().SyncQueue(queue_idx);
    // this->handle.sync_all();

    // PrintSubarray("v", v);

    // PrintSubarray("level_errors", level_errors);
    // PrintSubarray("encoded_bitplanes", encoded_bitplanes);
  }

  MGARDX_CONT
  SIZE MaxBitplaneLength(LENGTH n) {
    mgard_x::SIZE num_batches_per_TB = 2;
    const mgard_x::SIZE num_elems_per_TB =
        sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const mgard_x::SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    mgard_x::SIZE num_blocks = (n - 1) / num_elems_per_TB + 1;
    mgard_x::SIZE bitplane_max_length_total =
        bitplane_max_length_per_TB * num_blocks;
    return bitplane_max_length_total;
  }
};

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
          OPTION BinaryType, OPTION DecodingAlgorithm, typename DeviceType>
class GroupedDecoderFunctor : public Functor<DeviceType> {
public:
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
    for (SIZE bitplane_idx = FunctorBase<DeviceType>::GetThreadIdY(); bitplane_idx < num_bitplanes; bitplane_idx += 32) {
      for (SIZE batch_idx = FunctorBase<DeviceType>::GetThreadIdX(); batch_idx < num_batches_per_TB; batch_idx += 32) {
        sm_bitplanes[batch_idx * num_bitplanes + bitplane_idx] = *encoded_bitplanes(bitplane_idx, block_offset + batch_idx);
      }
    }

    if (BinaryType == BINARY) {
      // sign
      sign = 0; // 0: positive
                // 1: negative
      if (starting_bitplane == 0) {
        if (local_data_idx < num_batches_per_TB) {
          sm_bitplanes[num_batches_per_TB * num_bitplanes + local_data_idx] =
              *encoded_bitplanes(0, block_offset + num_batches_per_TB + local_data_idx);
        }
      } else {
        if (local_data_idx < num_elems_per_TB) {
          sm_signs[local_data_idx] = *signs(global_data_idx);
        }
      }
    }
  }

  // convert fix point to bit-planes
  // level error reduction (intra block)
  MGARDX_EXEC void Operation2() {
    // data
    BlockBitTranspose<T_bitplane, T_fp, 32, 32, 1, ALIGN_RIGHT, DecodingAlgorithm, DeviceType> blockBitTranspose;
    for (SIZE i = 0; i < num_batches_per_TB; i++) {
      blockBitTranspose.Transpose(sm_bitplanes + i * num_bitplanes,
                                  sm_fix_point + i * num_elems_per_batch,
                                  num_bitplanes, num_elems_per_batch);
    }

    if (BinaryType == BINARY) {
      // sign
      if (starting_bitplane == 0) {
        for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx++) {
          blockBitTranspose.Transpose(
              sm_bitplanes + num_batches_per_TB * num_bitplanes + batch_idx,
              sm_signs + batch_idx * num_elems_per_batch, 1,
              num_elems_per_batch);
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
        *v(global_data_idx) = sm_signs[local_data_idx] ? -cur_data : cur_data;
        *signs(global_data_idx) = sm_signs[local_data_idx];
      } else if (BinaryType == NEGABINARY) {
        T cur_data =
            ldexp((T)negabinary2binary(fp_data), -ending_bitplane + exp);
        *v(global_data_idx) = ending_bitplane % 2 != 0 ? -cur_data : cur_data;
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
class GroupedDecoder : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GroupedDecoder() : AutoTuner<DeviceType>() {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;
  using FunctorType =
      GroupedDecoderFunctor<T, T_fp, T_sfp, T_bitplane, BinaryType,
                            DecodingAlgorithm, DeviceType>;
  using TaskType = Task<FunctorType>;

  template <typename T_fp, typename T_sfp>
  MGARDX_CONT TaskType
  GenTask(SIZE n, SIZE num_batches_per_TB, SIZE starting_bitplane,
          SIZE num_bitplanes, SIZE exp,
          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
          SubArray<1, bool, DeviceType> signs, SubArray<1, T, DeviceType> v,
          int queue_idx) {
    // using FunctorType = GroupedDecoderFunctor<T, T_fp, T_sfp, T_bitplane,
    // BinaryType, DeviceType>;
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
    printf("GroupedDecoder config(%u %u %u) (%u %u %u)\n", tbx, tby, tbz, gridx,
           gridy, gridz);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  MGARDX_CONT
  void Execute(SIZE n, SIZE num_batches_per_TB, SIZE starting_bitplane,
               SIZE num_bitplanes, SIZE exp,
               SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
               SubArray<1, bool, DeviceType> signs,
               SubArray<1, T, DeviceType> v, int queue_idx) {
    TaskType task = GenTask<T_fp, T_sfp>(
        n, num_batches_per_TB, starting_bitplane, num_bitplanes, exp,
        encoded_bitplanes, signs, v, queue_idx);
    DeviceAdapter<TaskType, DeviceType> adapter;
    adapter.Execute(task);
    // this->handle.sync_all();
    // PrintSubarray("v", v);
  }
};

// general bitplane encoder that encodes data by block using T_stream type
// buffer
template <typename T_data,
          typename T_bitplane, typename T_error>
class GroupedBPEncoder
    : public concepts::BitplaneEncoderInterface<T_data,
                                                T_bitplane, T_error> {
public:
  GroupedBPEncoder(){
    std::cout << "GroupedBPEncoder\n";
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
  }

  Array<2, T_bitplane, CUDA>
  encode(SIZE n, SIZE num_bitplanes, int32_t exp,
         SubArray<1, T_data, CUDA> v,
         SubArray<1, T_error, CUDA> level_errors,
         std::vector<SIZE> &streams_sizes, int queue_idx) const {

    SIZE num_batches_per_TB = 2;
    const SIZE num_elems_per_TB =
        sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    SIZE num_blocks = (n - 1) / num_elems_per_TB + 1;
    SIZE bitplane_max_length_total =
        bitplane_max_length_per_TB * num_blocks;

    Array<2, T_error, CUDA> level_errors_work_array(
        {(SIZE)num_bitplanes + 1, num_blocks});
    SubArray<2, T_error, CUDA> level_errors_work(
        level_errors_work_array);

    Array<2, T_bitplane, CUDA> encoded_bitplanes_array(
        {(SIZE)num_bitplanes,
         (SIZE)bitplane_max_length_total});
    SubArray<2, T_bitplane, CUDA> encoded_bitplanes_subarray(
        encoded_bitplanes_array);

    MDR::GroupedEncoder<T_data, T_bitplane, T_error, BINARY_TYPE,
                                 DATA_ENCODING_ALGORITHM,
                                 ERROR_COLLECTING_ALGORITHM, CUDA>()
        .Execute(n, num_batches_per_TB, num_bitplanes, exp, v,
                 encoded_bitplanes_subarray, level_errors, level_errors_work,
                 queue_idx);

    for (int i = 0; i < num_bitplanes; i++) {
      streams_sizes[i] = bitplane_max_length_total * sizeof(T_bitplane);
    }

    return encoded_bitplanes_array;
  }

  Array<1, T_data, CUDA>
  decode(SIZE n, SIZE num_bitplanes, int32_t exp,
         SubArray<2, T_bitplane, CUDA> encoded_bitplanes,
         int level, int queue_idx) {}

  // decode the data and record necessary information for progressiveness
  Array<1, T_data, CUDA> progressive_decode(
      SIZE n, SIZE starting_bitplane,
      SIZE num_bitplanes, int32_t exp,
      SubArray<2, T_bitplane, CUDA> encoded_bitplanes,
      int level, int queue_idx) {

    SIZE num_batches_per_TB = 2;
    const SIZE num_elems_per_TB =
        sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    SIZE num_blocks = (n - 1) / num_elems_per_TB + 1;
    SIZE bitplane_max_length_total =
        bitplane_max_length_per_TB * num_blocks;

    if (level_signs.size() == level) {
      level_signs.push_back(Array<1, bool, CUDA>({n}));
    }

    // uint8_t * encoded_bitplanes = new uint8_t[bitplane_max_length_total *
    // num_bitplanes]; for (int i = 0; i < num_bitplanes; i++) {
    // memcpy(encoded_bitplanes + i * bitplane_max_length_total, streams[i],
    // bitplane_max_length_total * sizeof(uint8_t));
    // }
    // Array<2, uint8_t> encoded_bitplanes_array({(SIZE)num_bitplanes,
    // (SIZE)bitplane_max_length_total});
    // encoded_bitplanes_array.load(encoded_bitplanes);
    // SubArray<2, uint8_t> encoded_bitplanes_subarray(encoded_bitplanes_array);

    // Array<1, bool> signs_array({(SIZE)n});
    // signs_array.load(new_signs);
    SubArray<1, bool, CUDA> signs_subarray(
        level_signs[level]);

    Array<1, T_data, CUDA> v_array({(SIZE)n});
    SubArray<1, T_data, CUDA> v(v_array);

    // delete [] new_signs;

    // Array<1, T_data> v_array({(SIZE)n});
    // SubArray<1, T_data> v(v_array);

    if (num_bitplanes > 0) {
      MDR::GroupedDecoder<T_data, T_bitplane, BINARY_TYPE,
                                   DATA_DECODING_ALGORITHM, CUDA>()
          .Execute(n, num_batches_per_TB, starting_bitplane, num_bitplanes, exp,
                   encoded_bitplanes, signs_subarray, v, queue_idx);
    }

    // new_signs = signs_array.hostCopy();

    // for (int i = 0; i < n; i++) {
    //   signs[i] = new_signs[i];
    // }

    // T_data * temp_data = v_array.hostCopy();
    // memcpy(data, temp_data, n * sizeof(T_data));
    // return data;
    return v_array;
  }

  SIZE buffer_size(SIZE n) const {
    SIZE num_batches_per_TB = 2;
    const SIZE num_elems_per_TB =
        sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    SIZE num_blocks = (n - 1) / num_elems_per_TB + 1;
    SIZE bitplane_max_length_total =
        bitplane_max_length_per_TB * num_blocks;
    return bitplane_max_length_total;
  }

  void print() const { std::cout << "Grouped bitplane encoder" << std::endl; }

private:
  std::vector<Array<1, bool, CUDA>> level_signs;
  std::vector<std::vector<uint8_t>> level_recording_bitplanes;
};
} // namespace MDR
} // namespace mgard_x
#endif
