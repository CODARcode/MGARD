#ifndef _MDR_PERBIT_BP_ENCODER_GPU_HPP
#define _MDR_PERBIT_BP_ENCODER_GPU_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <bitset>
namespace mgard_x {
namespace MDR {
template <typename T, typename T_fp, typename T_bitplane, typename T_error,
          SIZE B, typename DeviceType>
class PerBitEncoderFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  PerBitEncoderFunctor(SIZE n, SIZE num_bitplanes, SIZE exp,
                       SubArray<1, T, CUDA> v,
                       SubArray<2, T_bitplane, CUDA> encoded_bitplanes,
                       SubArray<2, T_error, CUDA> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes), exp(exp),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {
    Functor<DeviceType>();
    bitplane_max_length = ((B * 2) - 1) / (sizeof(T_bitplane) * 8) + 1;
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
      debug = false;

    // assume 1D parallelization
    // B needs to be a multiply of MGARDX_WARP_SIZE
    // B >= num_bitplanes
    int8_t *sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_level_errors = (T_error *)sm_p;
    sm_p += ((num_bitplanes + 1) * B) * sizeof(T_error);
    sm_fix_point = (T_fp *)sm_p;
    sm_p += B * sizeof(T_fp);
    sm_shifted = (T *)sm_p;
    sm_p += B * sizeof(T);
    sm_bitplanes = (T_bitplane *)sm_p;
    sm_p += B * bitplane_max_length * sizeof(T_bitplane);
    sm_first_bit_pos = (SIZE *)sm_p;
    sm_p += B * sizeof(SIZE);
    sm_signs = (SIZE *)sm_p;
    sm_p += B * sizeof(SIZE);

    // task = threadx;
    // global_idx = blockx * nblockx + task;

    local_idx = FunctorBase<DeviceType>::GetThreadIdX();
    global_idx = FunctorBase<DeviceType>::GetBlockIdX() * B + local_idx;
    bitplane_idx = FunctorBase<DeviceType>::GetThreadIdY();

    // memset sm_level_errors to 0 to avoid thread divergence when reduce
    for (int k = 0; k < num_bitplanes + 1; k++) {
      sm_level_errors[k * B + local_idx] = 0;
    }

    // memset sm_bitplanes
    if (local_idx < num_bitplanes * bitplane_max_length) {
      sm_bitplanes[local_idx] = 0;
    }
    if (local_idx < B) {
      sm_fix_point[local_idx] = 0;
      sm_shifted[local_idx] = 0;
      sm_first_bit_pos[local_idx] = SIZE_MAX_VALUE; // no sign by default
    }

    // convert to fixpoint data
    if (global_idx < n) {
      T cur_data = *v(global_idx);
      T shifted_data = ldexp(cur_data, num_bitplanes - exp);

      T_fp fp_data = fabs((int64_t)shifted_data);
      // save fp_data to shared memory
      sm_fix_point[local_idx] = fp_data;
      sm_shifted[local_idx] = shifted_data;
      sm_signs[local_idx] =
          signbit(cur_data); // not using if to void thread divergence
      // printf("%llu, %f -> %u\n", global_idx, cur_data, fp_data );

      // detect first bit per elems
      // { // option 1: iteratively detect most significant bit
      //   bool first_bit_detected = false;
      //   for(int k=num_bitplanes-1; k>=0; k--){
      //     uint8_t index = num_bitplanes - 1 - k;
      //     uint8_t bit = (fp_data >> k) & 1u;
      //     // printf("%f %f %d\n", fp_data, fp_data>>k, (fp_data >> k) & 1u);
      //     if (bit && !first_bit_detected) {
      //       sm_first_bit_pos[task] = index;
      //       first_bit_detected = true;
      //       printf("first_bit_detected: %d\n", index);
      //     }
      //   }
      // }

      { // option 2: reverse fixpoint & use __ffsll detect least significant bit
        // printf("sizeof(T_fp) = %u\n", sizeof(T_fp));
        if (fp_data) {
          if (sizeof(T_fp) == sizeof(uint32_t)) {
            sm_first_bit_pos[local_idx] =
                __ffs(__brev(fp_data)) - (32 - num_bitplanes) - 1;
          } else if (sizeof(T_fp) == sizeof(uint64_t)) {
            sm_first_bit_pos[local_idx] =
                __ffsll(__brevll(fp_data)) - (64 - num_bitplanes) - 1;
          }
        }
      }
    }

    if (debug) {
      printf("encode data: ");
      for (int i = 0; i < B; i++) {
        if (FunctorBase<DeviceType>::GetBlockIdX() * B + i < n) {
          printf("%f ", *v(FunctorBase<DeviceType>::GetBlockIdX() * B + i));
        }
      }
      printf("\n");
    }
  }

  // convert fix point to bit-planes
  // level error reduction (intra block)
  MGARDX_EXEC void Operation2() {

    // collect errors
    T shifted_data = sm_shifted[local_idx];
    T_fp fp_data = sm_fix_point[local_idx];

    T abs_shifted_data = fabs(shifted_data);
    T_error mantissa = abs_shifted_data - fp_data;
    for (int k = 0; k < num_bitplanes; k++) {
      uint64_t mask = (1 << k) - 1;
      T_error diff = (T)(fp_data & mask) + mantissa;
      sm_level_errors[(num_bitplanes - k) * B + local_idx] = diff * diff;
    }
    sm_level_errors[local_idx] = abs_shifted_data * abs_shifted_data;

    { // option 1: one bit-plane per thread + put sign after most significant
      // bit
      bitplane_pos = 0;
      if (local_idx <
          num_bitplanes) { // each thread reponsibles for one bit-plane
        SIZE pos_in_buffer = 0;
        T_bitplane buffer = 0;
        uint8_t bitplane_index = local_idx;
        uint8_t bitplane_index_rev = min(num_bitplanes - 1 - bitplane_index, B);
        for (SIZE i = 0; i < B; i++) {
          uint8_t bit = (sm_fix_point[i] >> bitplane_index_rev) & 1u;

          if (debug) {
            printf("encode fp[%u] %u first bit %u\n", i, sm_fix_point[i],
                   sm_first_bit_pos[i]);
            for (int j = 0; j < B; j++) {
              printf("%u", (sm_fix_point[i] >> B - 1 - j) & 1u);
            }
            printf("\n");
          }

          buffer += bit << pos_in_buffer;
          pos_in_buffer++;

          if (pos_in_buffer == sizeof(T_bitplane) * 8) {
            sm_bitplanes[bitplane_pos * B + bitplane_index] = buffer;
            bitplane_pos++;
            buffer = 0;
            pos_in_buffer = 0;
            // printf("full bitplane_pos[%d]: %u\n", k, bitplane_pos);
          }
          if (bitplane_index == sm_first_bit_pos[i]) {
            buffer += sm_signs[i] << pos_in_buffer;
            pos_in_buffer++;
            if (pos_in_buffer == sizeof(T_bitplane) * 8) {
              sm_bitplanes[bitplane_pos * B + bitplane_index] = buffer;
              bitplane_pos++;
              buffer = 0;
              pos_in_buffer = 0;
              // printf("full bitplane_pos[%d]: %u\n", k, bitplane_pos);
            }
          }
        }
      }
    }

    { // option 2: one bit-plane per thread + store signs as a seperate
      // bit-plane
    }
  }

  // get max bit-plane length
  MGARDX_EXEC void Operation3() {

    { // level error reduction (intra block)
      // if (debug) {
      //   for (int i = 0; i < num_bitplanes+1; i++) {
      //     printf("[%d]: ", i);
      //     for (int j = 0; j < B; j++) {
      //       printf("%.2e ", sm_level_errors[i * B + j]);
      //     }
      //     printf("\n");
      //   }
      // }

      for (int i = 0; i < num_bitplanes + 1; i++) {
        T_error error = sm_level_errors[i * B + local_idx];
        T_error error_sum = 0;
        error_sum = blockReduce.Sum(error);
        if (FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
            FunctorBase<DeviceType>::GetThreadIdY() == 0 &&
            FunctorBase<DeviceType>::GetThreadIdZ() == 0) {
          *level_errors_workspace(i, FunctorBase<DeviceType>::GetBlockIdX()) =
              error_sum;
        }
        // if (debug) {
        //   printf("sum[%d]: %.2e\n", i, *level_errors_workspace(i, blockx));
        // }
      }
    }

    if (local_idx < B) {
      // printf("bitplane_pos[%llu]: %u\n", local_idx, bitplane_pos);
      SIZE max_bitplane = blockReduce.Max(bitplane_pos);
      // if (debug) {
      //   printf("max_bitplane: %u\n", max_bitplane);
      // }
    }
  }

  // store bit-plane
  MGARDX_EXEC void Operation4() {
    if (local_idx < bitplane_max_length) {
      for (SIZE bitplane_index = 0; bitplane_index < num_bitplanes;
           bitplane_index++) {
        SIZE block_offset =
            bitplane_max_length * FunctorBase<DeviceType>::GetBlockIdX();
        *encoded_bitplanes(bitplane_index, block_offset + local_idx) =
            sm_bitplanes[local_idx * B + bitplane_index];
      }
    }
    if (debug) {
      for (int i = 0; i < num_bitplanes; i++) {
        printf("encode bitpane[%d]: ", i);
        for (int j = 0; j < bitplane_max_length; j++) {
          printf(" %u  ", sm_bitplanes[j * B + i]);

          // for (int k = 0; k < B; k++) {
          //   printf("%u", (sm_bitplanes[j*B+i] >> B-1-k) & 1u);
          // }
        }
        printf("\n");
      }
      printf("\n");
    }
  }

  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += ((num_bitplanes + 1) * B) * sizeof(T_error);
    size += B * sizeof(T_fp);
    size += B * sizeof(T);
    size += B * bitplane_max_length * sizeof(T_bitplane);
    size += B * sizeof(SIZE);
    size += B * sizeof(SIZE);
    return size;
  }

private:
  // parameters
  SIZE n;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<1, T, CUDA> v;
  SubArray<2, T_bitplane, CUDA> encoded_bitplanes;
  SubArray<2, T_error, CUDA> level_errors_workspace;

  // stateful thread local variables
  bool debug;
  IDX local_idx, global_idx, bitplane_idx;
  SIZE bitplane_max_length;
  T_error *sm_level_errors;
  T_fp *sm_fix_point;
  T *sm_shifted;
  SIZE *sm_first_bit_pos;
  T_bitplane *sm_bitplanes;
  SIZE *sm_signs;
  SIZE bitplane_pos;
  BlockReduce<T_error, B, 1, 1, DeviceType> blockReduce;
};

template <typename T, typename DeviceType>
class PerBitEncoder : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  PerBitEncoder() : AutoTuner<DeviceType>() {}

  template <typename T_fp, typename T_bitplane, typename T_error, SIZE B>
  MGARDX_CONT
      Task<PerBitEncoderFunctor<T, T_fp, T_bitplane, T_error, B, DeviceType>>
      GenTask(SIZE n, SIZE num_bitplanes, SIZE exp, SubArray<1, T, CUDA> v,
              SubArray<2, T_bitplane, CUDA> encoded_bitplanes,
              SubArray<2, T_error, CUDA> level_errors_workspace,
              int queue_idx) {
    using FunctorType =
        PerBitEncoderFunctor<T, T_fp, T_bitplane, T_error, B, DeviceType>;
    FunctorType functor(n, num_bitplanes, exp, v, encoded_bitplanes,
                        level_errors_workspace);
    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = n;
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1; // num_bitplanes;
    tbx = B;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  MGARDX_CONT
  void Execute(SIZE n, SIZE num_bitplanes, SIZE exp, SubArray<1, T, CUDA> v,
               SubArray<2, uint8_t, CUDA> encoded_bitplanes,
               SubArray<1, double, CUDA> level_errors,
               SubArray<2, double, CUDA> level_errors_workspace,
               int queue_idx) {

    const int B = 32;
    if (std::is_same<T, double>::value) {
      using FunctorType =
          PerBitEncoderFunctor<T, uint64_t, uint8_t, double, B, DeviceType>;
      using TaskType = Task<FunctorType>;
      TaskType task = GenTask<uint64_t, uint8_t, double, B>(
          n, num_bitplanes, exp, v, encoded_bitplanes, level_errors_workspace,
          queue_idx);
      DeviceAdapter<TaskType, DeviceType> adapter;
      adapter.Execute(task);
    } else if (std::is_same<T, float>::value) {
      using FunctorType =
          PerBitEncoderFunctor<T, uint32_t, uint8_t, double, B, DeviceType>;
      using TaskType = Task<FunctorType>;
      TaskType task = GenTask<uint32_t, uint8_t, double, B>(
          n, num_bitplanes, exp, v, encoded_bitplanes, level_errors_workspace,
          queue_idx);
      DeviceAdapter<TaskType, DeviceType> adapter;
      adapter.Execute(task);
    }

    // PrintSubarray("level_errors_workspace", level_errors_workspace);
    // get level error
    using T_reduce = double;
    SIZE reduce_size = (n - 1) / B + 1;
    
    for (int i = 0; i < num_bitplanes + 1; i++) {
      SubArray<1, T_reduce, DeviceType> curr_errors(
          {reduce_size}, level_errors_workspace(i, 0));
      SubArray<1, T_reduce, DeviceType> sum_error({1}, level_errors(i));
      DeviceCollective<DeviceType>::Sum(reduce_size, curr_errors, sum_error, queue_idx);
    }

    // PrintSubarray("level_errors", level_errors);

    T_reduce *level_errors_temp = new T_reduce[num_bitplanes + 1];
    // level_errors.data(), (num_bitplanes+1)* sizeof(T_reduce), AUTO,
    // queue_idx);
    MemoryManager<DeviceType>().Copy1D(level_errors_temp, level_errors.data(),
                                       num_bitplanes + 1, queue_idx);

    for (int i = 0; i < num_bitplanes + 1; i++) {
      // printf("%f <- %f exp %d\n", ldexp(level_errors_temp[i], 2*(-
      // (int)num_bitplanes + exp)),
      //                             level_errors_temp[i], 2*(-
      //                             (int)num_bitplanes + exp));
      level_errors_temp[i] =
          ldexp(level_errors_temp[i], 2 * (-(int)num_bitplanes + exp));
    }
    // level_errors_temp, (num_bitplanes+1)* sizeof(T_reduce), AUTO, queue_idx);
    MemoryManager<DeviceType>().Copy1D(level_errors.data(), level_errors_temp,
                                       num_bitplanes + 1, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("level_errors", level_errors);

    delete[] level_errors_temp;
  }
};

template <typename T, typename T_fp, typename T_bitplane, SIZE B,
          typename DeviceType>
class PerBitDecoderFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  PerBitDecoderFunctor(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes,
                       SIZE exp,
                       SubArray<2, T_bitplane, CUDA> encoded_bitplanes,
                       SubArray<1, bool, CUDA> flags,
                       SubArray<1, bool, CUDA> signs, SubArray<1, T, CUDA> v)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), exp(exp),
        encoded_bitplanes(encoded_bitplanes), flags(flags), signs(signs), v(v) {
    Functor<DeviceType>();
    bitplane_max_length = ((B * 2) - 1) / (sizeof(T_bitplane) * 8) + 1;
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
      debug = false;

    // assume 1D parallelization
    // B needs to be a multiply of MGARDX_WARP_SIZE
    // B >= num_bitplanes
    int8_t *sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_bitplanes = (T_bitplane *)sm_p;
    sm_p += B * bitplane_max_length * sizeof(T_bitplane);
    sm_data = (T *)sm_p;
    sm_p += B * sizeof(T);
    // task = threadx;
    // global_idx = blockx * nblockx + task;

    local_idx = FunctorBase<DeviceType>::GetThreadIdX();
    global_idx = FunctorBase<DeviceType>::GetBlockIdX() * B + local_idx;
    bitplane_idx = FunctorBase<DeviceType>::GetThreadIdY();

    ending_bitplane = starting_bitplane + num_bitplanes;

    // memset sm_bitplanes
    if (local_idx < bitplane_max_length) {
      for (int i = 0; i < B; i++) {
        sm_bitplanes[local_idx * B + i] = 0;
      }
    }

    if (local_idx < bitplane_max_length) {
      for (SIZE i = 0; i < num_bitplanes; i++) {
        SIZE block_offset =
            bitplane_max_length * FunctorBase<DeviceType>::GetBlockIdX();
        sm_bitplanes[local_idx * B + i] =
            *encoded_bitplanes(i, block_offset + local_idx);
      }
    }
  }

  // convert fix point to bit-planes
  // level error reduction (intra block)
  MGARDX_EXEC void Operation2() {

    if (debug) {
      for (int i = 0; i < B; i++) {
        printf("decode bitpane[%d]: ", i);
        for (int j = 0; j < bitplane_max_length; j++) {
          printf(" %u  ", sm_bitplanes[j * B + i]);
          // for (int k = 0; k < B; k++) {
          //   printf("%u", (sm_bitplanes[j*B+i] >> B-1-k) & 1u);
          // }
        }
        printf("\n");
      }
      printf("\n");
    }

    { // option 1: one bit-plane per thread + put sign after most significant
      // bit

      SIZE bitplane_pos = 0;
      SIZE pos_in_buffer = 0;
      T_fp bit;
      bool sign;
      SIZE bitplane_index = local_idx;
      SIZE bitplane_index_rev = min(num_bitplanes - local_idx - 1, B);
      T_bitplane buffer = sm_bitplanes[bitplane_pos * B + bitplane_index];
      for (SIZE i = 0; i < B; i++) {
        LENGTH gloabl_data_idx = FunctorBase<DeviceType>::GetBlockIdX() * B + i;
        bit = buffer & 1u;
        buffer = buffer >> 1;
        pos_in_buffer++;
        if (pos_in_buffer == sizeof(T_bitplane) * 8) {
          bitplane_pos++;
          buffer = sm_bitplanes[bitplane_pos * B + bitplane_index];
          pos_in_buffer = 0;
        }

        T_fp fp_data = blockReduce.Sum(bit << bitplane_index_rev);
        fp_data = blockBroadcast.Broadcast(fp_data, 0, 0, 0);

        // if (i == 0) printf("decode fp[%u] %u\n", local_idx, fp_data);
        // if (i == 0) printf("pos_in_buffer %u bit[%u]: %u\n", pos_in_buffer,
        // bitplane_index, bit);

        // if (debug) {
        //   printf("fp[%u] (num_bitplanes: %u):", i, num_bitplanes);
        //   for (int j = 0; j < B; j++) {
        //     printf("%u", (fp_data >> B-1-j) & 1u);
        //     if (B-1-j == num_bitplanes) {
        //       printf("|");
        //     }
        //   }
        //   printf("\n");
        // }

        if (!(*flags(gloabl_data_idx))) {
          // get sign
          SIZE first_bit_pos = SIZE_MAX_VALUE;
          if (fp_data) {
            if (sizeof(T_fp) == sizeof(uint32_t)) {
              first_bit_pos = __ffs(__brev(fp_data)) - (32 - num_bitplanes) - 1;
            } else if (sizeof(T_fp) == sizeof(uint64_t)) {
              first_bit_pos =
                  __ffsll(__brevll(fp_data)) - (64 - num_bitplanes) - 1;
            }
          }

          // if (i == 0) printf("decode fp %u first bit %u\n",
          //                         fp_data, first_bit_pos);

          if (bitplane_index == first_bit_pos) {
            sign = buffer & 1u;
            buffer = buffer >> 1;
            pos_in_buffer++; // printf("sign [%u]++: %u\n", bitplane_index,
                             // pos_in_buffer);
            if (pos_in_buffer == sizeof(T_bitplane) * 8) {
              bitplane_pos++;
              buffer = sm_bitplanes[bitplane_pos * B + bitplane_index];
              pos_in_buffer = 0;
            }
            *signs(gloabl_data_idx) = sign;
            *flags(gloabl_data_idx) = true;
            T cur_data = ldexp((T)fp_data, -ending_bitplane + exp);
            sm_data[i] = sign ? -cur_data : cur_data;
          }
        } else {
          if (bitplane_index == 0) {
            sign = *signs(gloabl_data_idx);
            T cur_data = ldexp((T)fp_data, -ending_bitplane + exp);
            sm_data[i] = sign ? -cur_data : cur_data;
          }
        }
      }
    }

    { // option 2: one bit-plane per thread + store signs as a seperate
      // bit-plane
    }
  }

  // get max bit-plane length
  MGARDX_EXEC void Operation3() {
    *v(FunctorBase<DeviceType>::GetBlockIdX() * B + local_idx) =
        sm_data[local_idx];

    if (debug) {
      printf("dencode data: ");
      for (int i = 0; i < B; i++) {
        if (FunctorBase<DeviceType>::GetBlockIdX() * B + i < n) {
          printf("%f ", sm_data[i]);
        }
      }
      printf("\n");
    }
  }

  // store bit-plane
  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += B * bitplane_max_length * sizeof(T_bitplane);
    size += B * sizeof(T);
    return size;
  }

private:
  // parameters
  SIZE n;
  SIZE starting_bitplane;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> flags;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T, DeviceType> v;

  // stateful thread local variables
  bool debug;
  SIZE local_idx, global_idx, bitplane_idx;
  SIZE ending_bitplane;
  SIZE bitplane_max_length;
  T_bitplane *sm_bitplanes;
  T *sm_data;
  BlockReduce<T_fp, B, 1, 1, DeviceType> blockReduce;
  BlockBroadcast<T_fp, DeviceType> blockBroadcast;
};

template <typename T, typename DeviceType>
class PerBitDecoder : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  PerBitDecoder() : AutoTuner<DeviceType>() {}

  template <typename T_fp, typename T_bitplane, SIZE B>
  MGARDX_CONT Task<PerBitDecoderFunctor<T, T_fp, T_bitplane, B, DeviceType>>
  GenTask(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes, SIZE exp,
          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
          SubArray<1, bool, DeviceType> flags,
          SubArray<1, bool, DeviceType> signs, SubArray<1, T, DeviceType> v,
          int queue_idx) {
    using FunctorType =
        PerBitDecoderFunctor<T, T_fp, T_bitplane, B, DeviceType>;
    FunctorType functor(n, starting_bitplane, num_bitplanes, exp,
                        encoded_bitplanes, flags, signs, v);
    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = n;
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1; // num_bitplanes;
    tbx = B;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  MGARDX_CONT
  void Execute(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes, SIZE exp,
               SubArray<2, uint8_t, DeviceType> encoded_bitplanes,
               SubArray<1, bool, DeviceType> flags,
               SubArray<1, bool, DeviceType> signs,
               SubArray<1, T, DeviceType> v, int queue_idx) {

    const int B = 32;
    if (std::is_same<T, double>::value) {
      using FunctorType =
          PerBitDecoderFunctor<T, uint64_t, uint8_t, B, DeviceType>;
      using TaskType = Task<FunctorType>;
      TaskType task = GenTask<uint64_t, uint8_t, B>(
          n, starting_bitplane, num_bitplanes, exp, encoded_bitplanes, flags,
          signs, v, queue_idx);
      DeviceAdapter<TaskType, DeviceType> adapter;
      adapter.Execute(task);
    } else if (std::is_same<T, float>::value) {
      using FunctorType =
          PerBitDecoderFunctor<T, uint32_t, uint8_t, B, DeviceType>;
      using TaskType = Task<FunctorType>;
      TaskType task = GenTask<uint32_t, uint8_t, B>(
          n, starting_bitplane, num_bitplanes, exp, encoded_bitplanes, flags,
          signs, v, queue_idx);
      DeviceAdapter<TaskType, DeviceType> adapter;
      adapter.Execute(task);
    }

    // PrintSubarray("v", v);
  }
};

class BitEncoderGPU {
public:
  BitEncoderGPU(uint64_t *stream_begin_pos) {
    stream_begin = stream_begin_pos;
    stream_pos = stream_begin;
    buffer = 0;
    position = 0;
  }
  void encode(uint64_t b) {
    buffer += b << position;
    position++;
    if (position == 64) {
      // printf("encoder buffer full\n");
      *(stream_pos++) = buffer;
      buffer = 0;
      position = 0;
    }
  }
  void flush() {
    if (position) {
      *(stream_pos++) = buffer;
      buffer = 0;
      position = 0;
    }
  }
  uint32_t size() { return (stream_pos - stream_begin); }

private:
  uint64_t buffer = 0;
  uint8_t position = 0;
  uint64_t *stream_pos = NULL;
  uint64_t *stream_begin = NULL;
};

class BitDecoderGPU {
public:
  BitDecoderGPU(uint64_t const *stream_begin_pos) {
    stream_begin = stream_begin_pos;
    stream_pos = stream_begin;
    buffer = 0;
    position = 0;
  }
  uint8_t decode() {
    if (position == 0) {
      buffer = *(stream_pos++);
      position = 64;
    }
    uint8_t b = buffer & 1u;
    buffer >>= 1;
    position--;
    return b;
  }
  uint32_t size() { return (stream_pos - stream_begin); }

private:
  uint64_t buffer = 0;
  uint8_t position = 0;
  uint64_t const *stream_pos = NULL;
  uint64_t const *stream_begin = NULL;
};

#define PER_BIT_BLOCK_SIZE 1
// per bit bitplane encoder that encodes data by bit using T_stream type buffer
template <typename T_data, typename T_stream>
class PerBitBPEncoderGPU
    : public concepts::BitplaneEncoderInterface<T_data> {
public:
  PerBitBPEncoderGPU(){
    std::cout << "PerBitBPEncoder\n";
    static_assert(std::is_floating_point<T_data>::value,
                  "PerBitBPEncoderGPU: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "PerBitBPEncoderGPU: long double is not supported.");
    static_assert(std::is_unsigned<T_stream>::value,
                  "PerBitBPEncoderGPU: streams must be unsigned integers.");
    static_assert(std::is_integral<T_stream>::value,
                  "PerBitBPEncoderGPU: streams must be unsigned integers.");
  }

  std::vector<uint8_t *> encode(T_data const *data, SIZE n, int32_t exp,
                                uint8_t num_bitplanes,
                                std::vector<SIZE> &stream_sizes) const {

    assert(num_bitplanes > 0);
    // determine block size based on bitplane integer type
    const int32_t block_size = PER_BIT_BLOCK_SIZE;
    stream_sizes = std::vector<SIZE>(num_bitplanes, 0);
    // define fixed point type
    using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                           uint64_t, uint32_t>::type;
    std::vector<uint8_t *> streams;
    for (int i = 0; i < num_bitplanes; i++) {
      streams.push_back(
          (uint8_t *)malloc(2 * n / UINT8_BITS + sizeof(uint64_t)));
    }
    std::vector<BitEncoderGPU> encoders;
    for (int i = 0; i < streams.size(); i++) {
      encoders.push_back(
          BitEncoderGPU(reinterpret_cast<uint64_t *>(streams[i])));
    }
    T_data const *data_pos = data;

    for (int i = 0; i < n - block_size; i += block_size) {
      T_stream sign_bitplane = 0;
      for (int j = 0; j < block_size; j++) {
        T_data cur_data = *(data_pos++);
        T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
        bool sign = cur_data < 0;
        int64_t fix_point = (int64_t)shifted_data;
        T_fp fp_data = sign ? -fix_point : +fix_point;
        // compute level errors
        bool first_bit = true;
        for (int k = num_bitplanes - 1; k >= 0; k--) {
          uint8_t index = num_bitplanes - 1 - k;
          uint8_t bit = (fp_data >> k) & 1u;
          encoders[index].encode(bit);
          if (bit && first_bit) {
            encoders[index].encode(sign);
            first_bit = false;
          }
        }
      }
    }
    // leftover
    {
      int rest_size = n % block_size;
      if (rest_size == 0)
        rest_size = block_size;
      for (int j = 0; j < rest_size; j++) {
        T_data cur_data = *(data_pos++);
        T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
        bool sign = cur_data < 0;
        int64_t fix_point = (int64_t)shifted_data;
        T_fp fp_data = sign ? -fix_point : +fix_point;
        // compute level errors
        bool first_bit = true;
        for (int k = num_bitplanes - 1; k >= 0; k--) {
          uint8_t index = num_bitplanes - 1 - k;
          uint8_t bit = (fp_data >> k) & 1u;
          encoders[index].encode(bit);
          if (bit && first_bit) {
            encoders[index].encode(sign);
            first_bit = false;
          }
        }
      }
    }
    for (int i = 0; i < num_bitplanes; i++) {
      encoders[i].flush();
      stream_sizes[i] = encoders[i].size() * sizeof(uint64_t);
    }
    return streams;
  }

  // only differs in error collection
  std::vector<uint8_t *> encode(T_data const *data, SIZE n, int32_t exp,
                                uint8_t num_bitplanes,
                                std::vector<SIZE> &stream_sizes,
                                std::vector<double> &level_errors) const {

    // init level errors
    level_errors.clear();
    level_errors.resize(num_bitplanes + 1);
    for (int i = 0; i < level_errors.size(); i++) {
      level_errors[i] = 0;
    }
    stream_sizes = std::vector<SIZE>(num_bitplanes, 0);

    Array<1, T_data, CUDA> v_array({(SIZE)n});
    v_array.load(data);
    SubArray<1, T_data, CUDA> v(v_array);

    Array<1, double, CUDA> level_errors_array({(SIZE)num_bitplanes + 1});
    SubArray<1, double, CUDA> level_errors_subarray(level_errors_array);

    const int B = 32;
    using T_bitplane = uint8_t;
    SIZE bitplane_max_length_per_block =
        ((B * 2) - 1) / (sizeof(T_bitplane) * 8) + 1;
    SIZE num_blocks = (n - 1) / B + 1;
    SIZE bitplane_max_length_total = bitplane_max_length_per_block * num_blocks;

    Array<2, double, CUDA> level_errors_work_array(
        {(SIZE)num_bitplanes + 1, num_blocks});
    SubArray<2, double, CUDA> level_errors_work_subarray(
        level_errors_work_array);

    Array<2, uint8_t, CUDA> encoded_bitplanes_array(
        {(SIZE)num_bitplanes, (SIZE)bitplane_max_length_total});
    SubArray<2, uint8_t, CUDA> encoded_bitplanes_subarray(
        encoded_bitplanes_array);

    // printf("calling PerBitEncoder\n");
    PerBitEncoder<T_data, CUDA>().Execute(
        n, num_bitplanes, exp, v, encoded_bitplanes_subarray,
        level_errors_subarray, level_errors_work_subarray, 0);

    MemoryManager<CUDA>::Copy1D(level_errors.data(),
                                level_errors_subarray.data(), num_bitplanes + 1,
                                0);
    DeviceRuntime<CUDA>::SyncQueue(0);

    // PrintSubarray("encoded_bitplanes_subarray", encoded_bitplanes_subarray);

    uint8_t *encoded_bitplanes = encoded_bitplanes_array.hostCopy();

    std::vector<uint8_t *> streams2;
    for (int i = 0; i < num_bitplanes; i++) {
      stream_sizes[i] = bitplane_max_length_total * sizeof(uint8_t);
      streams2.push_back(
          (uint8_t *)malloc(bitplane_max_length_total * sizeof(uint8_t)));
      memcpy(streams2[i], encoded_bitplanes + i * bitplane_max_length_total,
             bitplane_max_length_total * sizeof(uint8_t));
    }
    return streams2;

    // for (int i = 0; i < num_bitplanes + 1; i++) {
    //   printf("error[i] = %f\n", level_errors_temp[i]);
    // }

    assert(num_bitplanes > 0);
    // determine block size based on bitplane integer type
    const int32_t block_size = PER_BIT_BLOCK_SIZE;

    // define fixed point type
    using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                           uint64_t, uint32_t>::type;
    std::vector<uint8_t *> streams;
    for (int i = 0; i < num_bitplanes; i++) {
      streams.push_back(
          (uint8_t *)malloc(2 * n / UINT8_BITS + sizeof(uint64_t)));
    }
    std::vector<BitEncoderGPU> encoders;
    for (int i = 0; i < streams.size(); i++) {
      encoders.push_back(
          BitEncoderGPU(reinterpret_cast<uint64_t *>(streams[i])));
    }

    T_data const *data_pos = data;
    // printf("n = %u\n",n);
    for (int i = 0; i < n - block_size; i += block_size) {
      T_stream sign_bitplane = 0;
      for (int j = 0; j < block_size; j++) {
        T_data cur_data = *(data_pos++);
        T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
        bool sign = cur_data < 0;
        int64_t fix_point = (int64_t)shifted_data;
        T_fp fp_data = sign ? -fix_point : +fix_point;
        // printf("cpu, %d %f -> %llu\n", data_pos-data-1, cur_data, fp_data);
        // compute level errors
        collect_level_errors(level_errors, fabs(shifted_data), num_bitplanes);
        bool first_bit = true;
        for (int k = num_bitplanes - 1; k >= 0; k--) {
          uint8_t index = num_bitplanes - 1 - k;
          uint8_t bit = (fp_data >> k) & 1u;
          encoders[index].encode(bit);
          // printf("encode bitplane[%u] <- %u from %u\n", index, bit,
          // data_pos-data);
          if (bit && first_bit) {
            // printf("first bit: %hu\n", index);
            encoders[index].encode(sign);
            first_bit = false;
          }
        }
      }
    }
    // leftover
    {
      int rest_size = n % block_size;
      if (rest_size == 0)
        rest_size = block_size;
      for (int j = 0; j < rest_size; j++) {
        T_data cur_data = *(data_pos++);
        T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
        bool sign = cur_data < 0;
        int64_t fix_point = (int64_t)shifted_data;
        T_fp fp_data = sign ? -fix_point : +fix_point;
        // printf("cpu fp_data[%d] %llu\n", data_pos-data-1, fp_data);
        // compute level errors
        collect_level_errors(level_errors, fabs(shifted_data), num_bitplanes);
        bool first_bit = true;
        for (int k = num_bitplanes - 1; k >= 0; k--) {
          uint8_t index = num_bitplanes - 1 - k;
          uint8_t bit = (fp_data >> k) & 1u;
          encoders[index].encode(bit);
          // printf("encode bitplane[%u] <- %u from %u\n", index, bit,
          // data_pos-data);
          if (bit && first_bit) {
            // printf("encode sign bitplane[%u] <- from %u\n", index,
            // data_pos-data);
            encoders[index].encode(sign);
            first_bit = false;
          }
        }
      }
    }
    for (int i = 0; i < num_bitplanes; i++) {
      encoders[i].flush();
      stream_sizes[i] = encoders[i].size() * sizeof(uint64_t);
      // printf("stream_sizes[%d]: %llu\n", i, stream_sizes[i]);
    }

    // for (int i = 0; i < num_bitplanes + 1; i++) {
    //   printf("level_errors[i] = %f\n", level_errors[i]);
    // }

    // translate level errors
    for (int i = 0; i < level_errors.size(); i++) {
      // printf("cpu %f <- %f exp %d\n", ldexp(level_errors[i], 2*(-
      // (int)num_bitplanes + exp)),
      //                       level_errors[i], 2*(- (int)num_bitplanes + exp));
      level_errors[i] = ldexp(level_errors[i], 2 * (-num_bitplanes + exp));
    }

    for (int i = 0; i < num_bitplanes + 1; i++) {
      printf("level_errors[i] = %f\n", level_errors[i]);
    }

    return streams;
  }

  T_data *decode(const std::vector<uint8_t const *> &streams, SIZE n, int exp,
                 uint8_t num_bitplanes) {
    const int32_t block_size = PER_BIT_BLOCK_SIZE;
    // define fixed point type
    using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                           uint64_t, uint32_t>::type;
    T_data *data = (T_data *)malloc(n * sizeof(T_data));
    if (num_bitplanes == 0) {
      memset(data, 0, n * sizeof(T_data));
      return data;
    }
    std::vector<BitDecoderGPU> decoders;
    for (int i = 0; i < streams.size(); i++) {
      decoders.push_back(
          BitDecoderGPU(reinterpret_cast<uint64_t const *>(streams[i])));
      decoders[i].size();
    }
    // decode
    T_data *data_pos = data;
    for (int i = 0; i < n - block_size; i += block_size) {
      for (int j = 0; j < block_size; j++) {
        T_fp fp_data = 0;
        // decode each bit of the data for each level component
        bool first_bit = true;
        bool sign = false;
        for (int k = num_bitplanes - 1; k >= 0; k--) {
          uint8_t index = num_bitplanes - 1 - k;
          uint8_t bit = decoders[index].decode();
          fp_data += bit << k;
          if (bit && first_bit) {
            // decode sign
            sign = decoders[index].decode();
            first_bit = false;
          }
        }
        T_data cur_data = ldexp((T_data)fp_data, -num_bitplanes + exp);
        *(data_pos++) = sign ? -cur_data : cur_data;
      }
    }
    // leftover
    {
      int rest_size = n % block_size;
      if (rest_size == 0)
        rest_size = block_size;
      for (int j = 0; j < rest_size; j++) {
        T_fp fp_data = 0;
        // decode each bit of the data for each level component
        bool first_bit = true;
        bool sign = false;
        for (int k = num_bitplanes - 1; k >= 0; k--) {
          uint8_t index = num_bitplanes - 1 - k;
          uint8_t bit = decoders[index].decode();
          fp_data += bit << k;
          if (bit && first_bit) {
            // decode sign
            sign = decoders[index].decode();
            first_bit = false;
          }
        }
        T_data cur_data = ldexp((T_data)fp_data, -num_bitplanes + exp);
        *(data_pos++) = sign ? -cur_data : cur_data;
      }
    }
    return data;
  }

  T_data *progressive_decode(const std::vector<uint8_t const *> &streams,
                             SIZE n, int exp, uint8_t starting_bitplane,
                             uint8_t num_bitplanes, int level) {

    if (level_signs.size() == level) {
      level_signs.push_back(std::vector<bool>(n, false));
      sign_flags.push_back(std::vector<bool>(n, false));
    }
    std::vector<bool> &signs = level_signs[level];
    std::vector<bool> &flags = sign_flags[level];

    T_data *data = (T_data *)malloc(n * sizeof(T_data));
    if (num_bitplanes == 0) {
      memset(data, 0, n * sizeof(T_data));
      return data;
    }

    Array<1, T_data, CUDA> v_array({(SIZE)n});
    SubArray<1, T_data, CUDA> v(v_array);

    const int B = 32;
    using T_bitplane = uint8_t;
    SIZE bitplane_max_length_per_block =
        ((B * 2) - 1) / (sizeof(T_bitplane) * 8) + 1;
    SIZE num_blocks = (n - 1) / B + 1;
    SIZE bitplane_max_length_total = bitplane_max_length_per_block * num_blocks;

    uint8_t *encoded_bitplanes =
        new uint8_t[bitplane_max_length_total * num_bitplanes];
    for (int i = 0; i < num_bitplanes; i++) {
      memcpy(encoded_bitplanes + i * bitplane_max_length_total, streams[i],
             bitplane_max_length_total * sizeof(uint8_t));
    }
    Array<2, uint8_t, CUDA> encoded_bitplanes_array(
        {(SIZE)num_bitplanes, (SIZE)bitplane_max_length_total});
    encoded_bitplanes_array.load(encoded_bitplanes);
    SubArray<2, uint8_t, CUDA> encoded_bitplanes_subarray(
        encoded_bitplanes_array);

    // PrintSubarray("decode encoded_bitplanes_subarray",
    // encoded_bitplanes_subarray);

    bool *new_flags = new bool[n];
    bool *new_signs = new bool[n];

    for (int i = 0; i < n; i++) {
      new_flags[i] = flags[i];
      new_signs[i] = signs[i];
    }

    Array<1, bool, CUDA> flags_array({(SIZE)n});
    flags_array.load(new_flags);
    SubArray<1, bool, CUDA> flags_subarray(flags_array);

    Array<1, bool, CUDA> signs_array({(SIZE)n});
    signs_array.load(new_signs);
    SubArray<1, bool, CUDA> signs_subarray(signs_array);

    delete[] new_flags;
    delete[] new_signs;

    PerBitDecoder<T_data, CUDA>().Execute(n, starting_bitplane, num_bitplanes,
                                          exp, encoded_bitplanes_subarray,
                                          flags_subarray, signs_subarray, v, 0);

    new_flags = flags_array.hostCopy();
    new_signs = signs_array.hostCopy();

    for (int i = 0; i < n; i++) {
      flags[i] = new_flags[i];
      signs[i] = new_signs[i];
    }

    T_data *temp_data = v_array.hostCopy();
    memcpy(data, temp_data, n * sizeof(T_data));
    return data;

    const int32_t block_size = PER_BIT_BLOCK_SIZE;
    // define fixed point type
    using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                           uint64_t, uint32_t>::type;

    std::vector<BitDecoderGPU> decoders;
    for (int i = 0; i < streams.size(); i++) {
      decoders.push_back(
          BitDecoderGPU(reinterpret_cast<uint64_t const *>(streams[i])));
      decoders[i].size();
    }

    const uint8_t ending_bitplane = starting_bitplane + num_bitplanes;
    // decode
    T_data *data_pos = data;
    for (int i = 0; i < n - block_size; i += block_size) {
      for (int j = 0; j < block_size; j++) {
        T_fp fp_data = 0;
        // decode each bit of the data for each level component
        bool sign = false;
        if (flags[i + j]) {
          // sign recorded
          sign = signs[i + j];
          for (int k = num_bitplanes - 1; k >= 0; k--) {
            uint8_t index = num_bitplanes - 1 - k;
            uint8_t bit = decoders[index].decode();
            fp_data += bit << k;
          }
        } else {
          // decode sign if possible
          bool first_bit = true;
          for (int k = num_bitplanes - 1; k >= 0; k--) {
            uint8_t index = num_bitplanes - 1 - k;
            uint8_t bit = decoders[index].decode();
            fp_data += bit << k;
            if (bit && first_bit) {
              // decode sign
              sign = decoders[index].decode();
              first_bit = false;
              flags[i + j] = true;
            }
          }
          signs[i + j] = sign;
        }
        T_data cur_data = ldexp((T_data)fp_data, -ending_bitplane + exp);
        *(data_pos++) = sign ? -cur_data : cur_data;
      }
    }
    // leftover
    {
      int rest_size = n % block_size;
      if (rest_size == 0)
        rest_size = block_size;
      for (int j = 0; j < rest_size; j++) {
        T_fp fp_data = 0;
        // decode each bit of the data for each level component
        bool sign = false;
        if (flags[n - rest_size + j]) {
          sign = signs[n - rest_size + j];
          for (int k = num_bitplanes - 1; k >= 0; k--) {
            uint8_t index = num_bitplanes - 1 - k;
            uint8_t bit = decoders[index].decode();
            fp_data += bit << k;
          }
        } else {
          bool first_bit = true;
          for (int k = num_bitplanes - 1; k >= 0; k--) {
            uint8_t index = num_bitplanes - 1 - k;
            uint8_t bit = decoders[index].decode();
            fp_data += bit << k;
            if (bit && first_bit) {
              // decode sign
              sign = decoders[index].decode();
              first_bit = false;
              flags[n - rest_size + j] = true;
            }
          }
          signs[n - rest_size + j] = sign;
        }
        T_data cur_data = ldexp((T_data)fp_data, -ending_bitplane + exp);
        *(data_pos++) = sign ? -cur_data : cur_data;
      }
    }
    return data;
  }
  void print() const { std::cout << "Per-bit bitplane encoder" << std::endl; }

private:
  inline void collect_level_errors(std::vector<double> &level_errors,
                                   double data, int num_bitplanes) const {
    uint64_t fp_data = (uint64_t)data;
    double mantissa = data - (uint64_t)data;
    level_errors[num_bitplanes] += mantissa * mantissa;
    for (int k = 1; k < num_bitplanes; k++) {
      uint64_t mask = (1 << k) - 1;
      double diff = (double)(fp_data & mask) + mantissa;
      level_errors[num_bitplanes - k] += diff * diff;
    }
    level_errors[0] += data * data;
  }
  std::vector<std::vector<bool>> level_signs;
  std::vector<std::vector<bool>> sign_flags;
};
} // namespace MDR
} // namespace mgard_x
#endif
