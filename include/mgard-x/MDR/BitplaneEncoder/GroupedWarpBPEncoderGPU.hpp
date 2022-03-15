#ifndef _MDR_GROUPED_WARP_BP_ENCODER_GPU_HPP
#define _MDR_GROUPED_WARP_BP_ENCODER_GPU_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

#define NUM_GROUPS_PER_WARP_PER_ITER 4
#define NUM_WARP_PER_TB 16

#define BINARY_TYPE BINARY

// #define DATA_ENCODING_ALGORITHM Warp_Bit_Transpose_Serial_All
#define DATA_ENCODING_ALGORITHM Warp_Bit_Transpose_Parallel_B_Serial_b
// #define DATA_ENCODING_ALGORITHM Warp_Bit_Transpose_Serial_B_Atomic_b
// #define DATA_ENCODING_ALGORITHM Warp_Bit_Transpose_Parallel_B_Reduce_b
// #define DATA_ENCODING_ALGORITHM Warp_Bit_Transpose_Parallel_B_Ballot_b

// #define DATA_DECODING_ALGORITHM Warp_Bit_Transpose_Serial_All
#define DATA_DECODING_ALGORITHM Warp_Bit_Transpose_Parallel_B_Serial_b
// #define DATA_DECODING_ALGORITHM Warp_Bit_Transpose_Serial_B_Atomic_b
// #define DATA_DECODING_ALGORITHM Warp_Bit_Transpose_Parallel_B_Reduce_b
// #define DATA_DECODING_ALGORITHM Warp_Bit_Transpose_Parallel_B_Ballot_b

// #define ERROR_COLLECTING_ALGORITHM Warp_Error_Collecting_Serial_All
#define ERROR_COLLECTING_ALGORITHM                                             \
  Warp_Error_Collecting_Parallel_Bitplanes_Serial_Error
// #define ERROR_COLLECTING_ALGORITHM
// Warp_Error_Collecting_Serial_Bitplanes_Atomic_Error #define
// ERROR_COLLECTING_ALGORITHM
// Warp_Error_Collecting_Serial_Bitplanes_Reduce_Error

namespace mgard_x {} // namespace mgard_x

namespace mgard_x {
namespace MDR {

template <typename T>
MGARDX_EXEC void print_bits2(T v, int num_bits, bool reverse = false) {
  for (int j = 0; j < num_bits; j++) {
    if (!reverse)
      printf("%u", (v >> num_bits - 1 - j) & 1u);
    else
      printf("%u", (v >> j) & 1u);
  }
}

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, SIZE NumEncodingBitplanes,
          SIZE NumGroupsPerWarpPerIter, SIZE NumWarpsPerTB, OPTION BinaryType,
          OPTION EncodingAlgorithm, OPTION ErrorColectingAlgorithm,
          typename DeviceType>
class GroupedWarpEncoderFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT GroupedWarpEncoderFunctor(
      LENGTH n, SIZE exp, SubArray<1, T, DeviceType> v,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), exp(exp), encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {
    Functor<DeviceType>();
    // MaxLengthPerWarpPerIter = NumGroupsPerWarpPerIter;
    // MaxLengthPerTBPerIter = NumGroupsPerTBPerIter;
    // if (BinaryType == BINARY) {
    //   MaxLengthPerWarpPerIter *= 2;
    //   MaxLengthPerTBPerIter *= 2;
    // }
  }
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
    sm_errors = (T_error *)sm_p;
    sm_p += NumWarpsPerTB * (NumEncodingBitplanes + 1) * sizeof(T_error);
    sm_errors_sum = (T_error *)sm_p;
    sm_p += (NumEncodingBitplanes + 1) * sizeof(T_error);
    sm_fix_point = (T_fp *)sm_p;
    sm_p += NumElemPerTBPerIter * sizeof(T_fp);

    if (BinaryType == BINARY) {
      sm_signs = (T_fp *)sm_p;
      sm_p += NumElemPerTBPerIter * sizeof(T_fp);
    }
    sm_shifted = (T *)sm_p;
    sm_p += NumElemPerTBPerIter * sizeof(T);
    sm_bitplanes = (T_bitplane *)sm_p;
    sm_p += NumEncodingBitplanes * NumGroupsPerTBPerIter * sizeof(T_bitplane);

    ld_sm_bitplanes = NumGroupsPerTBPerIter;

    sm_bitplanes_sign = (T_bitplane *)sm_p;
    sm_p += NumGroupsPerTBPerIter * sizeof(T_bitplane);
    ld_sm_bitplanes_sign = 1;

    lane_id = FunctorBase<DeviceType>::GetThreadIdX();
    warp_id = FunctorBase<DeviceType>::GetThreadIdY();

    // Data
    // For iter offsets
    NumElemPerIter =
        FunctorBase<DeviceType>::GetGridDimX() * NumElemPerTBPerIter;
    NumIters = (n - 1) / NumElemPerIter + 1;
    // TB and Warp offsets
    SIZE TB_data_offset =
        FunctorBase<DeviceType>::GetBlockIdX() * NumElemPerTBPerIter;
    SIZE warp_data_offset = warp_id * NumElemPerGroup * NumGroupsPerWarpPerIter;
    // Warp local shared memory
    T_fp *sm_warp_local_fix_point = sm_fix_point + warp_data_offset;
    T *sm_warp_local_shifted = sm_shifted + warp_data_offset;
    T_bitplane *sm_warp_local_signs;
    if (BinaryType == BINARY) {
      sm_warp_local_signs = sm_signs + warp_data_offset;
    }

    // Bitplane
    NumGroupsPerIter =
        FunctorBase<DeviceType>::GetGridDimX() * NumGroupsPerTBPerIter;
    // For iter offsets
    // MaxLengthPerIter = FunctorBase<DeviceType>::GetGridDimX() *
    // NumGroupsPerIter;//MaxLengthPerTBPerIter; TB and Warp offsets
    SIZE TB_bitplane_offset = FunctorBase<DeviceType>::GetBlockIdX() *
                              NumGroupsPerTBPerIter; // MaxLengthPerTBPerIter;
    SIZE warp_bitplane_offset =
        warp_id *
        NumGroupsPerWarpPerIter; // MaxLengthPerWarpPerIter;//NumGroupsPerWarpPerIter;
    T_bitplane *sm_warp_local_bitplanes = sm_bitplanes + warp_bitplane_offset;
    T_bitplane *sm_warp_local_bitplanes_sign =
        sm_bitplanes_sign + warp_bitplane_offset;

    // Error collect
    T_error *sm_warp_local_errors =
        sm_errors + (NumEncodingBitplanes + 1) * warp_id;

    // WarpBitTranspose<T_fp, T_bitplane, ALIGN_LEFT, EncodingAlgorithm,
    //                  DeviceType>
    //     warpBitTranspose;
    // WarpErrorCollect<T, T_fp, T_sfp, T_error, ErrorColectingAlgorithm,
    //                  BinaryType, DeviceType>
    //     warpErrorCollector;

    for (SIZE i = lane_id; i < NumEncodingBitplanes + 1;
         i += MGARDX_WARP_SIZE) {
      sm_warp_local_errors[i] = 0;
    }

    // convert to fixpoint data
    if (BinaryType == NEGABINARY)
      exp += 2;

    SIZE NumGroupsPerBatch = (MGARDX_WARP_SIZE - 1) / NumElemPerGroup + 1;
    SIZE NumBatches = NumGroupsPerWarpPerIter / NumGroupsPerBatch;

    for (SIZE Iter = 0; Iter < NumIters; Iter++) { // avoid TB context switch

      SIZE iter_data_offset = NumElemPerIter * Iter;
      SIZE iter_bitplane_offset = NumGroupsPerIter * Iter;

      SIZE global_bitplane_idx =
          iter_bitplane_offset + TB_bitplane_offset + warp_bitplane_offset;

      for (SIZE GroupIdx = 0; GroupIdx < NumGroupsPerWarpPerIter; GroupIdx++) {

        SIZE group_data_offset = GroupIdx * NumElemPerGroup;
        SIZE group_bitplane_offset = GroupIdx;

        SIZE global_data_idx = iter_data_offset + TB_data_offset +
                               warp_data_offset + group_data_offset;

        T cur_data = 0;
        if (global_data_idx + lane_id < n && lane_id < NumElemPerGroup) {
          cur_data = *v(global_data_idx + lane_id);
        }
        T shifted_data = ldexp(cur_data, (int)NumEncodingBitplanes - (int)exp);
        T_fp fp_data;
        if (BinaryType == BINARY) {
          fp_data = (T_fp)fabs(shifted_data);
        } else if (BinaryType == NEGABINARY) {
          fp_data = Math<DeviceType>::binary2negabinary((T_sfp)shifted_data);
        }
        // save fp_data to shared memory
        sm_warp_local_fix_point[group_data_offset + lane_id] = fp_data;
        sm_warp_local_shifted[group_data_offset + lane_id] = shifted_data;
        if (BinaryType == BINARY) {
          sm_warp_local_signs[group_data_offset + lane_id] =
              signbit(cur_data) << (sizeof(T_fp) * 8 - 1);
        }

        long long start;
        // if (debug && Iter == 0 && GroupIdx == 0) start = clock64();
        WarpBitTranspose<T_fp, T_bitplane, ALIGN_LEFT, EncodingAlgorithm,
                         sizeof(T_bitplane) * 8, NumEncodingBitplanes,
                         DeviceType>::Transpose(sm_warp_local_fix_point +
                                                    group_data_offset,
                                                1,
                                                sm_warp_local_bitplanes +
                                                    group_bitplane_offset,
                                                ld_sm_bitplanes, lane_id);
        //                            sizeof(T_bitplane)*8, num_bitplanes);
        // if (debug && Iter == 0 && GroupIdx == 0) { start = clock64() - start;
        // printf(" METHOD: %d, time: %llu\n", EncodingAlgorithm, start); }
        if (BinaryType == BINARY) {
          WarpBitTranspose<T_fp, T_bitplane, ALIGN_LEFT, EncodingAlgorithm,
                           sizeof(T_bitplane) * 8, 1,
                           DeviceType>::Transpose(sm_warp_local_signs +
                                                      group_data_offset,
                                                  1,
                                                  sm_warp_local_bitplanes_sign +
                                                      group_bitplane_offset,
                                                  ld_sm_bitplanes_sign,
                                                  lane_id);
          // NumElemPerGroup, 1);
          // printf("NumGroupsPerTBPerIter + group_bitplane_offset = %u\n",
          // NumGroupsPerTBPerIter + group_bitplane_offset); if (__mywarpid() <
          // 2 && lane_id == 0) printf("blockx: %llu, sign: %u\n",
          // FunctorBase<DeviceType>::GetBlockIdX(),
          // *(sm_warp_local_bitplanes_sign + group_bitplane_offset));
        }
        WarpErrorCollect<T, T_fp, T_sfp, T_error, ErrorColectingAlgorithm,
                         BinaryType, sizeof(T_bitplane) * 8,
                         NumEncodingBitplanes,
                         DeviceType>::Collect(sm_warp_local_shifted +
                                                  group_data_offset,
                                              sm_warp_local_errors, lane_id);
      }

      // store encoded bitplanes to gloabl memory
      for (SIZE bitplane_idx = 0; bitplane_idx < NumEncodingBitplanes;
           bitplane_idx++) {
        for (SIZE offset = lane_id; offset < NumGroupsPerWarpPerIter;
             offset += MGARDX_WARP_SIZE) {
          *encoded_bitplanes(bitplane_idx, global_bitplane_idx + offset) =
              sm_warp_local_bitplanes[bitplane_idx * ld_sm_bitplanes + offset];
        }
      }
      if (BinaryType == BINARY) {
        for (SIZE offset = lane_id; offset < NumGroupsPerWarpPerIter;
             offset += MGARDX_WARP_SIZE) {
          *encoded_bitplanes(0, global_bitplane_idx +
                                    NumGroupsPerIter * NumIters + offset) =
              sm_warp_local_bitplanes_sign[offset];
        }
      }
    }
  }

  MGARDX_EXEC void Operation2() {
    // Sum error from each warp
    SIZE liearized_idx = FunctorBase<DeviceType>::GetThreadIdY() *
                             FunctorBase<DeviceType>::GetBlockDimX() +
                         FunctorBase<DeviceType>::GetThreadIdX();
    // BlockReduce<T, NumWarpsPerTB, 1, 1, DeviceType> blockReducer;
    for (SIZE bitplane_idx = 0; bitplane_idx < NumEncodingBitplanes + 1;
         bitplane_idx++) {
      T_error error = 0;
      if (liearized_idx < NumWarpsPerTB) {
        error = sm_errors[liearized_idx * (NumEncodingBitplanes + 1) +
                          bitplane_idx];
      }

      // if (bitplane_idx == 0) printf("error: %f\n", error);
      T_error error_sum;
      BlockReduce<T_error, NumWarpsPerTB, 1, 1, DeviceType>::Sum(error,
                                                                 error_sum);
      if (liearized_idx == 0) {
        error_sum = ldexp(error_sum, 2 * (-(int)NumEncodingBitplanes + exp));
        sm_errors_sum[bitplane_idx] = error_sum;
      }
    }
    for (SIZE bitplane_idx = liearized_idx;
         bitplane_idx < NumEncodingBitplanes + 1;
         bitplane_idx += FunctorBase<DeviceType>::GetBlockDimX() *
                         FunctorBase<DeviceType>::GetBlockDimY()) {
      *level_errors_workspace(bitplane_idx,
                              FunctorBase<DeviceType>::GetBlockIdX()) =
          sm_errors_sum[bitplane_idx];
    }
  }

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {
    if (debug) {

      // printf("sm_fix_point:\t");
      // for (int i = 0; i < NumElemPerTBPerIter; i++) {
      //   printf("%u\t", sm_fix_point[i]);
      // }
      // printf("\n");

      // printf("encoded data:\t");
      // for (int i = 0; i < NumElemPerTBPerIter; i++) {
      //   printf("%f\t", *v(i));
      // }
      // printf("\n");

      // for (int i = 0; i < n; i++) {
      //   printf("input[%u]\torg\t%10.0f\t2^%d\tfp\t%llu:\t", i, sm_shifted[i],
      //   (int)NumEncodingBitplanes - (int)exp, sm_fix_point[i]);
      //   // printf("sign[%d]: %u\n", i, sm_signs[i]);
      //   print_bits2(sm_fix_point[i], NumEncodingBitplanes);
      //   printf("\n");
      // }

      // for (int i = 0; i < NumEncodingBitplanes; i++) {
      //   printf("sm_bitplane %d: ", i);
      //   for (int j = 0; j < NumGroupsPerTBPerIter; j++) {
      //     printf(" %10u: ", sm_bitplanes[i * ld_sm_bitplanes + j]);
      //     print_bits2(sm_bitplanes[i * ld_sm_bitplanes + j],
      //     sizeof(T_bitplane)*8, false);

      //   }
      //   printf("\n");
      // }

      // printf("sm_bitplane_sign: ");
      // for (int j = 0; j < 2; j++) {
      //   printf(" %10u: ", sm_bitplanes_sign[j]);
      //   print_bits2(sm_bitplanes[j], sizeof(T_bitplane)*8, false);

      // }
      // printf("\n");

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

      //   for (int i = 0; i < MGARDX_MAX_NUM_WARPS_PER_TB; i++) {
      //     printf("error-warp[%d]: ", i);
      //     for (int j = 0; j < num_bitplanes + 1; j++) {
      //       printf (" %f ", sm_errors[i * (num_bitplanes + 1) + j]);
      //     }
      //     printf("\n");
      //   }
      //   printf("\n");

      //   printf("error_sum: ");
      //   for (int i = 0; i < num_bitplanes + 1; i++) {
      //     printf (" %f ", sm_errors_sum[i]);
      //   }
      //   printf("\n");
    }
  }

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += NumWarpsPerTB * (NumEncodingBitplanes + 1) * sizeof(T_error);
    size += (NumEncodingBitplanes + 1) * sizeof(T_error);
    size += NumElemPerTBPerIter * sizeof(T_fp);
    size += NumEncodingBitplanes * NumGroupsPerTBPerIter * sizeof(T_bitplane);
    size += NumGroupsPerTBPerIter * sizeof(T_bitplane);
    size += NumElemPerTBPerIter * sizeof(T);
    if (BinaryType == BINARY) {
      size += NumElemPerTBPerIter * sizeof(T_fp);
    }
    // printf("shared_memory_size: %u\n", size);
    return size;
  }

private:
  // parameters
  LENGTH n;
  SIZE exp;
  SubArray<1, T, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;

  // stateful thread local variables

  bool debug;
  IDX local_data_idx, global_data_idx, local_bitplane_idx;
  IDX lane_id, warp_id;

  SIZE NumElemPerGroup = sizeof(T_bitplane) * 8;
  SIZE NumElemPerTBPerIter =
      NumElemPerGroup * NumGroupsPerWarpPerIter * NumWarpsPerTB;
  SIZE NumElemPerIter; // depends on num of TB
  SIZE NumIters;       // depends on num of TB

  SIZE NumGroupsPerTBPerIter = NumGroupsPerWarpPerIter * NumWarpsPerTB;
  SIZE NumGroupsPerIter; // depends on num of TB

  SIZE block_offset;
  T_error *sm_errors_sum;
  T_error *sm_errors;
  T_fp *sm_fix_point;
  T *sm_shifted;
  T_bitplane *sm_bitplanes;
  SIZE ld_sm_bitplanes;
  T_bitplane *sm_bitplanes_sign;
  SIZE ld_sm_bitplanes_sign;
  T_fp *sm_signs;
};

template <typename T, typename T_bitplane, typename T_error,
          SIZE NumGroupsPerWarpPerIter, SIZE NumWarpsPerTB, OPTION BinaryType,
          OPTION EncodingAlgorithm, OPTION ErrorColectingAlgorithm,
          typename DeviceType>
class GroupedWarpEncoder : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GroupedWarpEncoder() : AutoTuner<DeviceType>() {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;

  template <typename T_fp, typename T_sfp, SIZE NumEncodingBitplanes>
  MGARDX_CONT Task<GroupedWarpEncoderFunctor<
      T, T_fp, T_sfp, T_bitplane, T_error, NumEncodingBitplanes,
      NumGroupsPerWarpPerIter, NumWarpsPerTB, BinaryType, EncodingAlgorithm,
      ErrorColectingAlgorithm, DeviceType>>
  GenTask(LENGTH n, SIZE exp, SubArray<1, T, DeviceType> v,
          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
          SubArray<2, T_error, DeviceType> level_errors_workspace,
          int queue_idx) {
    using FunctorType =
        GroupedWarpEncoderFunctor<T, T_fp, T_sfp, T_bitplane, T_error,
                                  NumEncodingBitplanes, NumGroupsPerWarpPerIter,
                                  NumWarpsPerTB, BinaryType, EncodingAlgorithm,
                                  ErrorColectingAlgorithm, DeviceType>;
    FunctorType functor(n, exp, v, encoded_bitplanes, level_errors_workspace);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = NumWarpsPerTB;
    tbx = MGARDX_WARP_SIZE;
    gridz = 1;
    gridy = 1;
    gridx = MGARDX_NUM_SMs;
    // printf("GroupedWarpEncoder config(%u %u %u) (%u %u %u), sm_size: %llu\n",
    // tbx, tby, tbz, gridx, gridy, gridz, sm_size);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "GroupedWarpEncoder");
  }

  MGARDX_CONT
  void Execute(LENGTH n, SIZE num_bitplanes, SIZE exp,
               SubArray<1, T, DeviceType> v,
               SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
               SubArray<1, T_error, DeviceType> level_errors,
               SubArray<2, T_error, DeviceType> level_errors_workspace,
               int queue_idx) {

// PrintSubarray("v", v);
#define ENCODE(NumEncodingBitplanes)                                           \
  if (num_bitplanes == NumEncodingBitplanes) {                                 \
    using FunctorType = GroupedWarpEncoderFunctor<                             \
        T, T_fp, T_sfp, T_bitplane, T_error, NumEncodingBitplanes,             \
        NumGroupsPerWarpPerIter, NumWarpsPerTB, BinaryType, EncodingAlgorithm, \
        ErrorColectingAlgorithm, DeviceType>;                                  \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<T_fp, T_sfp, NumEncodingBitplanes>(                \
        n, exp, v, encoded_bitplanes, level_errors_workspace, queue_idx);      \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    adapter.Execute(task);                                                     \
  }

    ENCODE(1)
    ENCODE(2)
    ENCODE(3)
    ENCODE(4)
    ENCODE(5)
    ENCODE(6)
    ENCODE(7)
    ENCODE(8)
    ENCODE(9)
    ENCODE(10)
    ENCODE(11)
    ENCODE(12)
    ENCODE(13)
    ENCODE(14)
    ENCODE(15)
    ENCODE(16)
    ENCODE(17)
    ENCODE(18)
    ENCODE(19)
    ENCODE(20)
    ENCODE(21)
    ENCODE(22)
    ENCODE(23)
    ENCODE(24)
    ENCODE(25)
    ENCODE(26)
    ENCODE(27)
    ENCODE(28)
    ENCODE(29)
    ENCODE(30)
    ENCODE(31)
    ENCODE(32)
    ENCODE(33)
    ENCODE(34)
    ENCODE(35)
    ENCODE(36)
    ENCODE(37)
    ENCODE(38)
    ENCODE(39)
    ENCODE(40)
    ENCODE(41)
    ENCODE(42)
    ENCODE(43)
    ENCODE(44)
    ENCODE(45)
    ENCODE(46)
    ENCODE(47)
    ENCODE(48)
    ENCODE(49)
    ENCODE(50)
    ENCODE(51)
    ENCODE(52)
    ENCODE(53)
    ENCODE(54)
    ENCODE(55)
    ENCODE(56)
    ENCODE(57)
    ENCODE(58)
    ENCODE(59)
    ENCODE(60)
    ENCODE(61)
    ENCODE(62)
    ENCODE(63)
    ENCODE(64)

#undef ENCODE

    DeviceRuntime<DeviceType>().SyncQueue(queue_idx);
    // PrintSubarray("level_errors_workspace", level_errors_workspace);
    // get level error
    SIZE reduce_size = MGARDX_NUM_SMs;
    DeviceCollective<DeviceType> deviceReduce;
    for (int i = 0; i < num_bitplanes + 1; i++) {
      SubArray<1, T_error, DeviceType> curr_errors(
          {reduce_size}, level_errors_workspace(i, 0));
      SubArray<1, T_error, DeviceType> sum_error({1}, level_errors(i));
      deviceReduce.Sum(reduce_size, curr_errors, sum_error, queue_idx);
    }
    DeviceRuntime<DeviceType>().SyncQueue(queue_idx);

    // PrintSubarray("v", v);

    // PrintSubarray("level_errors", level_errors);
    // PrintSubarray("encoded_bitplanes", encoded_bitplanes);
  }

  MGARDX_CONT
  SIZE MaxBitplaneLength(LENGTH n) {
    SIZE NumElemPerGroup = sizeof(T_bitplane) * 8;
    SIZE NumElemPerTBPerIter =
        NumElemPerGroup * NumGroupsPerWarpPerIter * NumWarpsPerTB;
    SIZE MaxLengthPerTBPerIter = NumGroupsPerWarpPerIter * NumWarpsPerTB;
    if (BinaryType == BINARY) {
      MaxLengthPerTBPerIter *= 2;
    }
    LENGTH NumIters = (n - 1) / (MGARDX_NUM_SMs * NumElemPerTBPerIter) + 1;
    return MaxLengthPerTBPerIter * MGARDX_NUM_SMs * NumIters;
  }
};

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
          SIZE NumDecodingBitplanes, SIZE NumGroupsPerWarpPerIter,
          SIZE NumWarpsPerTB, OPTION BinaryType, OPTION DecodingAlgorithm,
          typename DeviceType>
class GroupedWarpDecoderFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT GroupedWarpDecoderFunctor(
      LENGTH n, SIZE starting_bitplane, SIZE exp,
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
      SubArray<1, bool, DeviceType> signs, SubArray<1, T, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), exp(exp),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
  }

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
      debug2 = false;

    int8_t *sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_fix_point = (T_fp *)sm_p;
    sm_p += NumElemPerTBPerIter * sizeof(T_fp);
    if (BinaryType == BINARY) {
      sm_signs = (T_fp *)sm_p;
      sm_p += NumElemPerTBPerIter * sizeof(T_fp);
    }
    sm_bitplanes = (T_bitplane *)sm_p;
    sm_p += NumDecodingBitplanes * NumGroupsPerTBPerIter * sizeof(T_bitplane);
    ld_sm_bitplanes = NumGroupsPerTBPerIter;

    sm_bitplanes_sign = (T_bitplane *)sm_p;
    sm_p += NumGroupsPerTBPerIter * sizeof(T_bitplane);
    ld_sm_bitplanes_sign = 1;

    int ending_bitplane = starting_bitplane + NumDecodingBitplanes;

    lane_id = FunctorBase<DeviceType>::GetThreadIdX();
    warp_id = FunctorBase<DeviceType>::GetThreadIdY();

    // Data
    // For iter offsets
    NumElemPerIter =
        FunctorBase<DeviceType>::GetGridDimX() * NumElemPerTBPerIter;
    NumIters = (n - 1) / NumElemPerIter + 1;
    // TB and Warp offsets
    SIZE TB_data_offset =
        FunctorBase<DeviceType>::GetBlockIdX() * NumElemPerTBPerIter;
    SIZE warp_data_offset = warp_id * NumElemPerGroup * NumGroupsPerWarpPerIter;
    // Warp local shared memory
    T_fp *sm_warp_local_fix_point = sm_fix_point + warp_data_offset;
    T_bitplane *sm_warp_local_signs;
    if (BinaryType == BINARY) {
      sm_warp_local_signs = sm_signs + warp_data_offset;
    }

    // Bitplane
    NumGroupsPerIter =
        FunctorBase<DeviceType>::GetGridDimX() * NumGroupsPerTBPerIter;
    // For iter offsets
    // MaxLengthPerIter = FunctorBase<DeviceType>::GetGridDimX() *
    // NumGroupsPerIter;//MaxLengthPerTBPerIter; TB and Warp offsets
    SIZE TB_bitplane_offset = FunctorBase<DeviceType>::GetBlockIdX() *
                              NumGroupsPerTBPerIter; // MaxLengthPerTBPerIter;
    SIZE warp_bitplane_offset =
        warp_id *
        NumGroupsPerWarpPerIter; // MaxLengthPerWarpPerIter;//NumGroupsPerWarpPerIter;
    T_bitplane *sm_warp_local_bitplanes = sm_bitplanes + warp_bitplane_offset;
    T_bitplane *sm_warp_local_bitplanes_sign =
        sm_bitplanes_sign + warp_bitplane_offset;

    // WarpBitTranspose<T_fp, T_bitplane, ALIGN_RIGHT, DecodingAlgorithm,
    //                  DeviceType>
    //     warpBitTranspose;
    if (BinaryType == NEGABINARY)
      exp += 2;

    for (SIZE Iter = 0; Iter < NumIters; Iter++) { // avoid TB context switch

      SIZE iter_data_offset = NumElemPerIter * Iter;
      SIZE iter_bitplane_offset = NumGroupsPerIter * Iter;

      SIZE global_bitplane_idx =
          iter_bitplane_offset + TB_bitplane_offset + warp_bitplane_offset;

      // load encoded bitplanes to shared memory
      for (SIZE bitplane_idx = 0; bitplane_idx < NumDecodingBitplanes;
           bitplane_idx++) {
        for (SIZE offset = lane_id; offset < NumGroupsPerWarpPerIter;
             offset += MGARDX_WARP_SIZE) {
          sm_warp_local_bitplanes[bitplane_idx * ld_sm_bitplanes + offset] =
              *encoded_bitplanes(bitplane_idx, global_bitplane_idx + offset);
        }
      }
      if (BinaryType == BINARY) {
        for (SIZE offset = lane_id; offset < NumGroupsPerWarpPerIter;
             offset += MGARDX_WARP_SIZE) {
          sm_warp_local_bitplanes_sign[offset] = *encoded_bitplanes(
              0, global_bitplane_idx + NumGroupsPerIter * NumIters + offset);
        }
      }

      for (SIZE GroupIdx = 0; GroupIdx < NumGroupsPerWarpPerIter; GroupIdx++) {
        SIZE group_data_offset = GroupIdx * NumElemPerGroup;
        SIZE group_bitplane_offset = GroupIdx;

        SIZE global_data_idx = iter_data_offset + TB_data_offset +
                               warp_data_offset + group_data_offset;

        WarpBitTranspose<T_fp, T_bitplane, ALIGN_RIGHT, DecodingAlgorithm,
                         NumDecodingBitplanes, sizeof(T_bitplane) * 8,
                         DeviceType>::Transpose(sm_warp_local_bitplanes +
                                                    group_bitplane_offset,
                                                ld_sm_bitplanes,
                                                sm_warp_local_fix_point +
                                                    group_data_offset,
                                                1, lane_id);
        // num_bitplanes, sizeof(T_bitplane)*8);

        if (BinaryType == BINARY) {
          if (starting_bitplane == 0) {
            WarpBitTranspose<T_fp, T_bitplane, ALIGN_RIGHT, DecodingAlgorithm,
                             1, sizeof(T_bitplane) * 8, DeviceType>::
                Transpose(sm_warp_local_bitplanes_sign + group_bitplane_offset,
                          ld_sm_bitplanes_sign,
                          sm_warp_local_signs + group_data_offset, 1, lane_id);
            // 1, NumElemPerGroup);
            // if (__mywarpid() < 2 && lane_id == 0) printf("blockx: %llu,
            // sign: %u\n", FunctorBase<DeviceType>::GetBlockIdX(),
            // *(sm_warp_local_bitplanes_sign + group_bitplane_offset));

          } else {
            if (global_data_idx + lane_id < n) {
              sm_warp_local_signs[group_data_offset + lane_id] =
                  *signs(global_data_idx + lane_id);
            } else {
              sm_warp_local_signs[group_data_offset + lane_id] = false;
            }
          }
        }

        if (global_data_idx + lane_id < n && lane_id < NumElemPerGroup) {
          T_fp fp_data = sm_warp_local_fix_point[group_data_offset + lane_id];
          if (BinaryType == BINARY) {
            T cur_data = ldexp((T)fp_data, -ending_bitplane + exp);
            *v(global_data_idx + lane_id) =
                sm_warp_local_signs[group_data_offset + lane_id] ? -cur_data
                                                                 : cur_data;
            if (starting_bitplane == 0) {
              *signs(global_data_idx + lane_id) =
                  sm_warp_local_signs[group_data_offset + lane_id];
            }
            // if (lane_id == 0) {
            //   printf("fp: %u, 2^%d, cur_data: %f\n", fp_data, -
            //   ending_bitplane + exp, cur_data);
            // }

          } else if (BinaryType == NEGABINARY) {
            T cur_data = ldexp((T)Math<DeviceType>::negabinary2binary(fp_data),
                               -ending_bitplane + exp);
            *v(global_data_idx + lane_id) =
                ending_bitplane % 2 != 0 ? -cur_data : cur_data;
          }
        }
      }
    }
  }

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {

    if (debug) {

      // printf("sm_fix_point:\t");
      // for (int i = 0; i < NumElemPerTBPerIter; i++) {
      //   printf("%u\t", sm_fix_point[i]);
      // }
      // printf("\n");

      // for (int i = 0; i < NumDecodingBitplanes; i++) {
      //   printf("sm_bitplane %d: ", i);
      //   for (int j = 0; j < NumGroupsPerTBPerIter; j++) {
      //     printf(" %10u: ", sm_bitplanes[i * ld_sm_bitplanes + j]);
      //     print_bits2(sm_bitplanes[i * ld_sm_bitplanes + j],
      //     sizeof(T_bitplane)*8, false);

      //   }
      //   printf("\n");
      // }

      //   printf("sm_bitplane_sign: ");
      //   for (int j = 0; j < 2; j++) {
      //     printf(" %10u: ", sm_bitplanes_sign[j]);
      //     print_bits2(sm_bitplanes_sign[j], sizeof(T_bitplane)*8, false);

      //   }
      //   printf("\n");

      //   printf("sm_signs: ");
      //   for (int i = 0; i < num_elems_per_TB; i++) {
      //     printf("%u ,", sm_signs[i]);
      //   }
      //   printf("\n");

      //   printf("decoded data:\t");
      //   for (int i = 0; i < NumElemPerTBPerIter; i++) {
      //     printf("%f\t", *v(i));
      //   }
      //   printf("\n");
    }
  }

  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += NumElemPerTBPerIter * sizeof(T_fp);
    if (BinaryType == BINARY) {
      size += NumElemPerTBPerIter * sizeof(T_fp);
    }
    size += NumDecodingBitplanes * NumGroupsPerTBPerIter * sizeof(T_bitplane);
    size += NumGroupsPerTBPerIter * sizeof(T_bitplane);
    return size;
  }

private:
  // parameters
  LENGTH n;
  SIZE starting_bitplane;
  SIZE exp;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T, DeviceType> v;

  // stateful thread local variables
  bool debug, debug2;
  IDX local_data_idx, global_data_idx;

  IDX lane_id, warp_id;

  SIZE NumElemPerGroup = sizeof(T_bitplane) * 8;
  SIZE NumElemPerTBPerIter =
      NumElemPerGroup * NumGroupsPerWarpPerIter * NumWarpsPerTB;
  SIZE NumElemPerIter; // depends on num of TB
  SIZE NumIters;       // depends on num of TB

  SIZE NumGroupsPerTBPerIter = NumGroupsPerWarpPerIter * NumWarpsPerTB;
  SIZE NumGroupsPerIter; // depends on num of TB

  T_bitplane *sm_bitplanes;
  SIZE ld_sm_bitplanes;
  T_bitplane *sm_bitplanes_sign;
  SIZE ld_sm_bitplanes_sign;
  T_fp *sm_fix_point;
  bool sign;
  T_fp *sm_signs;
};

template <typename T, typename T_bitplane, SIZE NumGroupsPerWarpPerIter,
          SIZE NumWarpsPerTB, OPTION BinaryType, OPTION DecodingAlgorithm,
          typename DeviceType>
class GroupedWarpDecoder : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GroupedWarpDecoder() : AutoTuner<DeviceType>() {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;

  template <typename T_fp, typename T_sfp, SIZE NumDecodingBitplanes>
  MGARDX_CONT Task<GroupedWarpDecoderFunctor<
      T, T_fp, T_sfp, T_bitplane, NumDecodingBitplanes, NumGroupsPerWarpPerIter,
      NumWarpsPerTB, BinaryType, DecodingAlgorithm, DeviceType>>
  GenTask(LENGTH n, SIZE starting_bitplane, SIZE exp,
          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
          SubArray<1, bool, DeviceType> signs, SubArray<1, T, DeviceType> v,
          int queue_idx) {
    using FunctorType =
        GroupedWarpDecoderFunctor<T, T_fp, T_sfp, T_bitplane,
                                  NumDecodingBitplanes, NumGroupsPerWarpPerIter,
                                  NumWarpsPerTB, BinaryType, DecodingAlgorithm,
                                  DeviceType>;
    FunctorType functor(n, starting_bitplane, exp, encoded_bitplanes, signs, v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = NumWarpsPerTB;
    tbx = MGARDX_WARP_SIZE;
    gridz = 1;
    gridy = 1;
    gridx = MGARDX_NUM_SMs;
    // printf("GroupedWarpDecoder config(%u %u %u) (%u %u %u), sm_size: %llu\n",
    // tbx, tby, tbz, gridx, gridy, gridz, sm_size);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "GroupedWarpDecoder");
  }

  MGARDX_CONT
  void Execute(LENGTH n, SIZE starting_bitplane, SIZE num_bitplanes, SIZE exp,
               SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
               SubArray<1, bool, DeviceType> signs,
               SubArray<1, T, DeviceType> v, int queue_idx) {
#define DECODE(NumDecodingBitplanes)                                           \
  if (num_bitplanes == NumDecodingBitplanes) {                                 \
    using FunctorType =                                                        \
        GroupedWarpDecoderFunctor<T, T_fp, T_sfp, T_bitplane,                  \
                                  NumDecodingBitplanes,                        \
                                  NumGroupsPerWarpPerIter, NumWarpsPerTB,      \
                                  BinaryType, DecodingAlgorithm, DeviceType>;  \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<T_fp, T_sfp, NumDecodingBitplanes>(                \
        n, starting_bitplane, exp, encoded_bitplanes, signs, v, queue_idx);    \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    adapter.Execute(task);                                                     \
  }

    DECODE(1)
    DECODE(2)
    DECODE(3)
    DECODE(4)
    DECODE(5)
    DECODE(6)
    DECODE(7)
    DECODE(8)
    DECODE(9)
    DECODE(10)
    DECODE(11)
    DECODE(12)
    DECODE(13)
    DECODE(14)
    DECODE(15)
    DECODE(16)
    DECODE(17)
    DECODE(18)
    DECODE(19)
    DECODE(20)
    DECODE(21)
    DECODE(22)
    DECODE(23)
    DECODE(24)
    DECODE(25)
    DECODE(26)
    DECODE(27)
    DECODE(28)
    DECODE(29)
    DECODE(30)
    DECODE(31)
    DECODE(32)
    DECODE(33)
    DECODE(34)
    DECODE(35)
    DECODE(36)
    DECODE(37)
    DECODE(38)
    DECODE(39)
    DECODE(40)
    DECODE(41)
    DECODE(42)
    DECODE(43)
    DECODE(44)
    DECODE(45)
    DECODE(46)
    DECODE(47)
    DECODE(48)
    DECODE(49)
    DECODE(50)
    DECODE(51)
    DECODE(52)
    DECODE(53)
    DECODE(54)
    DECODE(55)
    DECODE(56)
    DECODE(57)
    DECODE(58)
    DECODE(59)
    DECODE(60)
    DECODE(61)
    DECODE(62)
    DECODE(63)
    DECODE(64)

#undef DECODE

    DeviceRuntime<DeviceType>().SyncQueue(queue_idx);
    // PrintSubarray("v", v);
  }
};

// general bitplane encoder that encodes data by block using T_stream type
// buffer
template <typename T_data, typename T_bitplane, typename T_error,
          typename DeviceType>
class GroupedWarpBPEncoder
    : public concepts::BitplaneEncoderInterface<T_data, T_bitplane, T_error,
                                                DeviceType> {
public:
  GroupedWarpBPEncoder() {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
  }

  Array<2, T_bitplane, DeviceType>
  encode(SIZE n, SIZE num_bitplanes, int32_t exp,
         SubArray<1, T_data, DeviceType> v,
         SubArray<1, T_error, DeviceType> level_errors,
         std::vector<SIZE> &streams_sizes, int queue_idx) const {

    MDR::GroupedWarpEncoder<T_data, T_bitplane, T_error,
                            NUM_GROUPS_PER_WARP_PER_ITER, NUM_WARP_PER_TB,
                            BINARY_TYPE, DATA_ENCODING_ALGORITHM,
                            ERROR_COLLECTING_ALGORITHM, DeviceType>
        encoder;

    Array<2, T_error, DeviceType> level_errors_work_array(
        {num_bitplanes + 1, MGARDX_NUM_SMs});
    SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

    Array<2, T_bitplane, DeviceType> encoded_bitplanes_array(
        {num_bitplanes, encoder.MaxBitplaneLength(n)});
    SubArray<2, T_bitplane, DeviceType> encoded_bitplanes_subarray(
        encoded_bitplanes_array);

    encoder.Execute(n, num_bitplanes, exp, v, encoded_bitplanes_subarray,
                    level_errors, level_errors_work, queue_idx);

    for (int i = 0; i < num_bitplanes; i++) {
      streams_sizes[i] = encoder.MaxBitplaneLength(n) * sizeof(T_bitplane);
    }

    return encoded_bitplanes_array;
  }

  Array<1, T_data, DeviceType>
  decode(SIZE n, SIZE num_bitplanes, int32_t exp,
         SubArray<2, T_bitplane, DeviceType> encoded_bitplanes, int level,
         int queue_idx) {}

  // decode the data and record necessary information for progressiveness
  Array<1, T_data, DeviceType>
  progressive_decode(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes,
                     int32_t exp,
                     SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                     int level, int queue_idx) {

    // SIZE num_batches_per_TB = 2;
    // const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 *
    // num_batches_per_TB; const SIZE bitplane_max_length_per_TB =
    // num_batches_per_TB * 2; SIZE num_blocks =
    // (n-1)/num_elems_per_TB+1; SIZE bitplane_max_length_total =
    // bitplane_max_length_per_TB * num_blocks;

    // const SIZE NumGroupsPerWarpPerIter = 2;
    // const SIZE NumWarpsPerTB = 16;
    MDR::GroupedWarpDecoder<T_data, T_bitplane, NUM_GROUPS_PER_WARP_PER_ITER,
                            NUM_WARP_PER_TB, BINARY_TYPE,
                            DATA_DECODING_ALGORITHM, DeviceType>
        decoder;

    if (level_signs.size() == level) {
      level_signs.push_back(Array<1, bool, DeviceType>({(SIZE)n}));
    }

    SubArray<1, bool, DeviceType> signs_subarray(level_signs[level]);

    Array<1, T_data, DeviceType> v_array({(SIZE)n});
    SubArray<1, T_data, DeviceType> v(v_array);

    if (num_bitplanes > 0) {
      // if (num_bitplanes == 1) {
      //   PrintSubarray("signs_subarray", signs_subarray);
      // }
      decoder.Execute(n, starting_bitplane, num_bitplanes, exp,
                      encoded_bitplanes, signs_subarray, v, queue_idx);
    }
    return v_array;
  }

  SIZE buffer_size(SIZE n) const {
    MDR::GroupedWarpEncoder<T_data, T_bitplane, T_error,
                            NUM_GROUPS_PER_WARP_PER_ITER, NUM_WARP_PER_TB,
                            BINARY_TYPE, DATA_ENCODING_ALGORITHM,
                            ERROR_COLLECTING_ALGORITHM, DeviceType>
        encoder;
    return encoder.MaxBitplaneLength(n);
  }

  void print() const { std::cout << "Grouped bitplane encoder" << std::endl; }

private:
  std::vector<Array<1, bool, DeviceType>> level_signs;
  std::vector<std::vector<uint8_t>> level_recording_bitplanes;
};
} // namespace MDR
} // namespace mgard_x
#endif
