#ifndef _MDR_GROUPED_BP_ENCODER_GPU_HPP
#define _MDR_GROUPED_BP_ENCODER_GPU_HPP

#include "../../CommonInternal.h"
#include "../../Functor.h"
#include "../../AutoTuner.h"
#include "../../Task.h"
#include "../../DeviceAdapters/DeviceAdapterCuda.h"

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
// #define ERROR_COLLECTING_ALGORITHM Error_Collecting_Parallel_Bitplanes_Serial_Error
// #define ERROR_COLLECTING_ALGORITHM Error_Collecting_Parallel_Bitplanes_Atomic_Error
#define ERROR_COLLECTING_ALGORITHM Error_Collecting_Parallel_Bitplanes_Reduce_Error


namespace mgard_cuda {
namespace MDR {



  template <typename T>
  MGARDm_EXEC void
  print_bits(T v, int num_bits, bool reverse = false) {
    for (int j = 0; j < num_bits; j++) {
      if (!reverse) printf("%u", (v >> num_bits-1-j) & 1u);
      else printf("%u", (v >> j) & 1u);
    }
  }

  template <typename T, typename T_fp, typename T_sfp, typename T_bitplane, typename T_error, OPTION BinaryType, OPTION EncodingAlgorithm, OPTION ErrorColectingAlgorithm,typename DeviceType>
  class GroupedEncoderFunctor: public Functor<DeviceType> {
    public: 
    MGARDm_CONT GroupedEncoderFunctor(SIZE n,
                                      SIZE num_batches_per_TB,
                                      SIZE num_bitplanes,
                                      SIZE exp,
                                      SubArray<1, T> v,
                                      SubArray<2, T_bitplane> encoded_bitplanes,
                                      SubArray<2, T_error> level_errors_workspace):
                                      n(n), num_bitplanes(num_bitplanes), num_batches_per_TB(num_batches_per_TB),
                                      exp(exp), encoded_bitplanes(encoded_bitplanes),
                                      v(v), level_errors_workspace(level_errors_workspace) {
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
    MGARDm_EXEC void
    Operation1() {

      debug = false;
      if (this->blockz == 0 && this->blocky == 0 && this->blockx == 0 &&
            this->threadx == 0 && this->thready == 0 && this->threadz == 0) 
        debug = true;

      int8_t * sm_p = (int8_t *)this->shared_memory;
      sm_temp_errors =  (T_error*)sm_p;    sm_p += (num_bitplanes + 1) * num_elems_per_TB * sizeof(T_error);
      sm_errors =       (T_error*)sm_p;    sm_p += (num_bitplanes + 1)                    * sizeof(T_error);
      sm_fix_point =    (T_fp*)sm_p;       sm_p += num_elems_per_TB                       * sizeof(T_fp);
      if (BinaryType == BINARY) {
        sm_signs =        (T_fp*)sm_p;       sm_p += num_elems_per_TB                       * sizeof(T_fp);
      }
      sm_shifted =      (T*)sm_p;          sm_p += num_elems_per_TB                       * sizeof(T);
      sm_bitplanes =    (T_bitplane*)sm_p; sm_p += (num_bitplanes + 1) * num_batches_per_TB * sizeof(T_bitplane);
      
      // sm_reduce =  (blockReduce_error.TempStorageType*) sm_p;
      // blockReduce_error.AllocateTempStorage();
      // thread orginal data mapping
      local_data_idx = this->thready * this->nblockx + this->threadx;
      global_data_idx = this->blockx * num_elems_per_TB + local_data_idx;

      local_bitplane_idx = this->thready * this->nblockx + this->threadx;
      // // thread bitplane mapping (transposed of data mapping for more efficient )
      // bitplane_idx = this->threadx / B;
      // block_idx = this->threadx % B;

      // if (local_data_idx < num_elems_per_TB) {
      //   sm_fix_point[local_data_idx] = 0;
      //   sm_shifted[local_data_idx] = 0;
      //   sm_signs[local_data_idx] = 0;
      // }

      for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes+1; bitplane_idx++) {
        if (local_data_idx < num_elems_per_TB) {
          sm_temp_errors[bitplane_idx * num_elems_per_TB + local_data_idx] = 0;
        }
      }

      if (local_bitplane_idx < num_bitplanes + 1) {
        sm_errors[local_bitplane_idx] = 0; 
      }

      __syncthreads();
      if (BinaryType == NEGABINARY) exp += 2;
      // convert to fixpoint data
      if (local_data_idx < num_elems_per_TB && global_data_idx < n) {
        T cur_data = *v(global_data_idx); 
        T shifted_data = ldexp(cur_data, (int)num_bitplanes - (int)exp);
        T_fp fp_data;
        if (BinaryType == BINARY) {
          fp_data = (T_fp) fabs(shifted_data);
        } else if (BinaryType == NEGABINARY) {
          fp_data = binary2negabinary((T_sfp)shifted_data);
          // printf("2^%d %f->%u\n", (int)num_bitplanes - (int)exp, shifted_data, fp_data);
        }
        // save fp_data to shared memory
        sm_fix_point[local_data_idx] = fp_data;
        sm_shifted  [local_data_idx] = shifted_data;
        if (BinaryType == BINARY) {
          sm_signs    [local_data_idx] = signbit(cur_data) << (sizeof(T_fp)*8 - 1); 
        }
        // printf("%llu, %f -> %f-> %u\n", global_data_idx, cur_data, shifted_data, sm_fix_point[local_data_idx] );
        // printf("sm_fix_point[%llu]: %u\n", local_data_idx, sm_fix_point[local_data_idx]);
      }
    }

    // convert fix point to bit-planes
    // level error reduction (intra block)
    MGARDm_EXEC void
    Operation2() {
      // data
      BlockBitTranspose<T_fp, T_bitplane, 32, 32, 1, ALIGN_LEFT, EncodingAlgorithm, DeviceType> blockBitTranspose;
      for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx++) {
        blockBitTranspose.Transpose(sm_fix_point + batch_idx * num_elems_per_batch, 
                                    sm_bitplanes + batch_idx * num_bitplanes, 
                                    num_elems_per_batch, num_bitplanes);
      }
      if (BinaryType == BINARY) {
        // sign
        for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx++) {
          blockBitTranspose.Transpose(sm_signs + batch_idx * num_elems_per_batch, 
                                      sm_bitplanes + num_batches_per_TB * num_bitplanes + batch_idx,
                                      num_elems_per_batch, 1);
        }
      }
      // error
      ErrorCollect<T, T_fp, T_sfp, T_error, 32, 32, 1, ErrorColectingAlgorithm, BinaryType, DeviceType>().Collect(sm_shifted, sm_temp_errors, sm_errors, num_elems_per_TB, num_bitplanes);
    }

    // get max bit-plane length 
    MGARDm_EXEC void
    Operation3() {
      // data
      block_offset = max_length_per_TB * this->blockx;
      for (SIZE bitplane_idx = this->thready; bitplane_idx < num_bitplanes; bitplane_idx += 32) {
        for (SIZE batch_idx = this->threadx; batch_idx < num_batches_per_TB; batch_idx += 32) {
          *encoded_bitplanes(bitplane_idx, block_offset + batch_idx) = 
              sm_bitplanes[batch_idx * num_bitplanes + bitplane_idx];
        }
      }

      if (BinaryType == BINARY) {
        // sign
        if (local_data_idx < num_batches_per_TB) {
          *encoded_bitplanes(0, block_offset + num_batches_per_TB + local_data_idx) = 
              sm_bitplanes[num_batches_per_TB*num_bitplanes + local_data_idx];
        }
      }

      // error
      if (local_bitplane_idx < num_bitplanes + 1) {
        sm_errors[local_bitplane_idx] = ldexp(sm_errors[local_bitplane_idx], 2*(- (int)num_bitplanes + exp));
      }

      if (local_bitplane_idx < num_bitplanes + 1) {
        *level_errors_workspace(local_bitplane_idx, this->blockx) = sm_errors[local_bitplane_idx];
      }

    }

    MGARDm_EXEC void
    Operation4() {
      if (debug) {
        // for (int i = 0; i < num_elems_per_TB; i++) {
        //   printf("input[%u]\torg\t%f\t2^%d\tfp\t%llu:\t", i, *v(this->blockx*num_elems_per_TB+i), (int)num_bitplanes - (int)exp, sm_fix_point[i]);
        //   print_bits(sm_fix_point[i], num_bitplanes);
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
        //     printf (" %f ", sm_errors[i]);
        //   printf("\n");
        // }

      }
    }

    MGARDm_EXEC void
    Operation5() {}

    MGARDm_CONT size_t
    shared_memory_size() {
      size_t size = 0;
      size += (num_bitplanes + 1) * num_elems_per_TB  * sizeof(T_error);
      size += (num_bitplanes + 1)                     * sizeof(T_error);
      size += num_elems_per_TB                        * sizeof(T_fp);
      size += (num_bitplanes + 1) * num_batches_per_TB * sizeof(T_bitplane);
      size += num_elems_per_TB                        * sizeof(T);
      if (BinaryType == BINARY) {
        size += num_elems_per_TB                        * sizeof(T_fp);
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
    SubArray<1, T> v;
    SubArray<2, T_bitplane> encoded_bitplanes;
    SubArray<2, T_error> level_errors_workspace;

    // stateful thread local variables
    
    bool debug;
    IDX local_data_idx, global_data_idx, local_bitplane_idx;

    SIZE num_elems_per_batch = sizeof(T_bitplane) * 8;
    SIZE num_elems_per_TB = num_elems_per_batch * num_batches_per_TB;
    SIZE max_length_per_TB;
    SIZE block_offset;
    T_error * sm_temp_errors;
    T_error * sm_errors;
    T_fp * sm_fix_point;
    T * sm_shifted;
    T_bitplane * sm_bitplanes;
    T_fp * sm_signs;
  };


  template <typename HandleType, typename T, typename T_bitplane, typename T_error, OPTION BinaryType, OPTION EncodingAlgorithm, OPTION ErrorColectingAlgorithm, typename DeviceType>
  class GroupedEncoder: public AutoTuner<HandleType, DeviceType> {
  public:
    MGARDm_CONT
    GroupedEncoder(HandleType& handle):AutoTuner<HandleType, DeviceType>(handle) {}

    using T_sfp = typename std::conditional<std::is_same<T, double>::value, int64_t, int32_t>::type;
    using T_fp = typename std::conditional<std::is_same<T, double>::value, uint64_t, uint32_t>::type;
    using FunctorType = GroupedEncoderFunctor<T, T_fp, T_sfp, T_bitplane, T_error, BinaryType, EncodingAlgorithm, ErrorColectingAlgorithm, DeviceType>;
    using TaskType = Task<FunctorType>;


    template <typename T_fp, typename T_sfp>
    MGARDm_CONT
    TaskType GenTask(
                    SIZE n,
                    SIZE num_batches_per_TB,
                    SIZE num_bitplanes,
                    SIZE exp,
                    SubArray<1, T> v,
                    SubArray<2, T_bitplane> encoded_bitplanes,
                    SubArray<2, T_error> level_errors_workspace,
                    int queue_idx) 
    {
      FunctorType functor(n, num_batches_per_TB, num_bitplanes, exp, v, encoded_bitplanes, level_errors_workspace);
        SIZE tbx, tby, tbz, gridx, gridy, gridz;
        size_t sm_size = functor.shared_memory_size();
        const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
        tbz = 1;
        tby = 32;
        tbx = 32;
        gridz = 1;
        gridy = 1;
        gridx = (n-1) / num_elems_per_TB + 1;
        // printf("GroupedEncoder config(%u %u %u) (%u %u %u), sm_size: %llu\n", tbx, tby, tbz, gridx, gridy, gridz, sm_size);
        return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx); 
    }

    MGARDm_CONT
    void Execute(SIZE n,
                 SIZE num_batches_per_TB,
                 SIZE num_bitplanes,
                 SIZE exp,
                 SubArray<1, T> v,
                 SubArray<2, T_bitplane> encoded_bitplanes,
                 SubArray<1, T_error> level_errors,
                 SubArray<2, T_error> level_errors_workspace,
                 int queue_idx) {
      
      // PrintSubarray("v", v);
      TaskType task = GenTask<T_fp, T_sfp>(n, num_batches_per_TB, num_bitplanes, exp, v, encoded_bitplanes, level_errors_workspace, queue_idx);
      DeviceAdapter<HandleType, TaskType, DeviceType> adapter(this->handle);
      adapter.Execute(task);
      this->handle.sync_all();
      // PrintSubarray("level_errors_workspace", level_errors_workspace);
      // get level error
      const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
      SIZE reduce_size = (n-1)/num_elems_per_TB+1;
      DeviceReduce<HandleType, T_error, DeviceType> deviceReduce(this->handle);
      for (int i = 0; i < num_bitplanes + 1; i++) {
        SubArray<1, T_error> curr_errors({reduce_size}, level_errors_workspace(i, 0));
        SubArray<1, T_error> sum_error({1}, level_errors(i));
        deviceReduce.Sum(reduce_size, curr_errors, sum_error, queue_idx);
      }
      this->handle.sync_all();

      // PrintSubarray("v", v);

      // PrintSubarray("level_errors", level_errors);
      // PrintSubarray("encoded_bitplanes", encoded_bitplanes);

    }

    MGARDm_CONT
    SIZE MaxBitplaneLength(LENGTH n) {
      mgard_cuda::SIZE num_batches_per_TB = 2;
      const mgard_cuda::SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
      const mgard_cuda::SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
      mgard_cuda::SIZE num_blocks = (n-1)/num_elems_per_TB+1;
      mgard_cuda::SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_blocks;
      return bitplane_max_length_total;

    }
  };


  template <typename T, typename T_fp, typename T_sfp, typename T_bitplane, OPTION BinaryType, OPTION DecodingAlgorithm, typename DeviceType>
  class GroupedDecoderFunctor: public Functor<DeviceType> 
  {
      public: 
      MGARDm_CONT GroupedDecoderFunctor(SIZE n,
                                        SIZE num_batches_per_TB,
                                        SIZE starting_bitplane,
                                        SIZE num_bitplanes,
                                        SIZE exp,
                                        SubArray<2, T_bitplane> encoded_bitplanes,
                                        SubArray<1, bool> signs,
                                        SubArray<1, T> v):
                                        n(n), num_batches_per_TB(num_batches_per_TB),
                                        starting_bitplane(starting_bitplane), num_bitplanes(num_bitplanes),
                                        exp(exp), encoded_bitplanes(encoded_bitplanes), signs(signs),
                                        v(v) {
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
      MGARDm_EXEC void
      Operation1() {
        debug = false;
        if (this->blockz == 0 && this->blocky == 0 && this->blockx == 0 &&
              this->threadx == 0 && this->thready == 0 && this->threadz == 0) 
          debug = true;

        debug2 = false;
        if (this->blockz == 0 && this->blocky == 0 && this->blockx == 0) 
          debug2 = true;

        int8_t * sm_p = (int8_t *)this->shared_memory;
        sm_fix_point =  (T_fp*)sm_p;       sm_p += num_elems_per_TB                       * sizeof(T_fp);
        if (BinaryType == BINARY) {
          sm_signs =        (T_fp*)sm_p;       sm_p += num_elems_per_TB                       * sizeof(T_fp);
        }
        sm_bitplanes  = (T_bitplane*)sm_p; sm_p += num_batches_per_TB * (num_bitplanes+1) * sizeof(T_bitplane);
        

        local_data_idx = this->thready * this->nblockx + this->threadx;
        global_data_idx = this->blockx * num_elems_per_TB + local_data_idx;

        ending_bitplane = starting_bitplane + num_bitplanes;

        if (BinaryType == NEGABINARY) exp += 2;
        // data
        block_offset = max_length_per_TB * this->blockx;
        for (SIZE bitplane_idx = this->thready; bitplane_idx < num_bitplanes; bitplane_idx += 32) {
          for (SIZE batch_idx = this->threadx; batch_idx < num_batches_per_TB; batch_idx += 32) {
            sm_bitplanes[batch_idx * num_bitplanes + bitplane_idx] = 
              *encoded_bitplanes(bitplane_idx, block_offset + batch_idx);
          }
        }

        if (BinaryType == BINARY) {
          //sign
          sign = 0;  // 0: positive
                     // 1: negative
          if (starting_bitplane == 0) {
            if (local_data_idx < num_batches_per_TB) {
              sm_bitplanes[num_batches_per_TB*num_bitplanes + local_data_idx] = 
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
      MGARDm_EXEC void
      Operation2() {
        //data
        BlockBitTranspose<T_bitplane, T_fp, 32, 32, 1, ALIGN_RIGHT, DecodingAlgorithm, DeviceType> blockBitTranspose;
        for (SIZE i = 0; i < num_batches_per_TB; i++) {
          blockBitTranspose.Transpose(sm_bitplanes + i * num_bitplanes, sm_fix_point + i * num_elems_per_batch,  num_bitplanes, num_elems_per_batch);
        }

        if (BinaryType == BINARY) {
          // sign
          if (starting_bitplane == 0) {
            for (SIZE batch_idx = 0; batch_idx < num_batches_per_TB; batch_idx ++) {
              blockBitTranspose.Transpose(sm_bitplanes + num_batches_per_TB*num_bitplanes + batch_idx,
                                          sm_signs + batch_idx * num_elems_per_batch,
                                          1, num_elems_per_batch);
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
        //     bit = (sm_bitplanes[encoding_block_idx*num_bitplanes + i] >> encoding_bit_idx) & 1u ;
        //     fp_data += bit << num_bitplanes - 1 - i;
        //   }

        //   if (starting_bitplane == 0) {
        //     // decoding signs (total B blocks)
        //       T_bitplane sign_bitplane = *encoded_bitplanes(0, block_offset + B + encoding_block_idx);
        //       sign = (sign_bitplane >> encoding_bit_idx) & 1u;
        //       if (encoding_bit_idx == 0) {
        //         *signs(local_data_idx) = sign;
        //       }
        //   } else {
        //     sign = *signs(local_data_idx);
        //   }
        // }

        // T cur_data = ldexp((T)fp_data, - ending_bitplane + exp);

        // // if (debug) printf("fp[%llu]: %u -> 2^%u %f\n", local_data_idx, fp_data, - ending_bitplane + exp, cur_data);
        // *v(local_data_idx) = sign ? -cur_data : cur_data;
      }


      // store bit-plane
      MGARDm_EXEC void
      Operation3() {
        if (local_data_idx < num_elems_per_TB) {
          T_fp fp_data = sm_fix_point[local_data_idx];
          if (BinaryType == BINARY) {
            T cur_data = ldexp((T)fp_data, - ending_bitplane + exp);
            *v(global_data_idx) = sm_signs[local_data_idx] ? -cur_data : cur_data;
            *signs(global_data_idx) = sm_signs[local_data_idx];
          } else if (BinaryType == NEGABINARY) {
            T cur_data = ldexp((T)negabinary2binary(fp_data), - ending_bitplane + exp);
            *v(global_data_idx) = ending_bitplane % 2 != 0 ? -cur_data : cur_data;
          }
        }
      }

      MGARDm_EXEC void
      Operation4() {

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
        //     printf("%f\t", *v(this->blockx * num_elems_per_TB + i));
        //   }
        //   printf("\n");
        // }
      }

      

      MGARDm_EXEC void
      Operation5() {

      }
      MGARDm_CONT size_t
      shared_memory_size() {
        size_t size = 0;
        size += num_batches_per_TB * (num_bitplanes+1) * sizeof(T_bitplane);
        size += num_elems_per_TB                       * sizeof(T_fp);
        if (BinaryType == BINARY) {
          size += num_elems_per_TB                        * sizeof(T_fp);
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
      SubArray<2, T_bitplane> encoded_bitplanes;
      SubArray<1, bool> signs;
      SubArray<1, T> v;

      // stateful thread local variables
      bool debug, debug2;
      IDX local_data_idx, global_data_idx;

      SIZE num_elems_per_batch = sizeof(T_bitplane) * 8;
      SIZE num_elems_per_TB = num_elems_per_batch * num_batches_per_TB;
      SIZE max_length_per_TB;
      SIZE block_offset;
      SIZE ending_bitplane;
      SIZE bitplane_max_length;
      T_bitplane * sm_bitplanes;
      T_fp * sm_fix_point;
      bool sign;
      T_fp * sm_signs;


  };


  template <typename HandleType, typename T, typename T_bitplane, OPTION BinaryType, OPTION DecodingAlgorithm, typename DeviceType>
  class GroupedDecoder: public AutoTuner<HandleType, DeviceType> {
  public:
    MGARDm_CONT
    GroupedDecoder(HandleType& handle):AutoTuner<HandleType, DeviceType>(handle) {}

    using T_sfp = typename std::conditional<std::is_same<T, double>::value, int64_t, int32_t>::type;
    using T_fp = typename std::conditional<std::is_same<T, double>::value, uint64_t, uint32_t>::type;
    using FunctorType = GroupedDecoderFunctor<T, T_fp, T_sfp, T_bitplane, BinaryType, DecodingAlgorithm, DeviceType>;
    using TaskType = Task<FunctorType>;

    template <typename T_fp, typename T_sfp>
    MGARDm_CONT
    TaskType GenTask(SIZE n,
                     SIZE num_batches_per_TB,
                     SIZE starting_bitplane,
                     SIZE num_bitplanes,
                     SIZE exp,
                     SubArray<2, T_bitplane> encoded_bitplanes,
                     SubArray<1, bool> signs,
                     SubArray<1, T> v,
                     int queue_idx) 
{
      // using FunctorType = GroupedDecoderFunctor<T, T_fp, T_sfp, T_bitplane, BinaryType, DeviceType>;
      FunctorType functor(n, num_batches_per_TB, starting_bitplane, num_bitplanes, exp, encoded_bitplanes, signs, v);
        SIZE tbx, tby, tbz, gridx, gridy, gridz;
        size_t sm_size = functor.shared_memory_size();
        const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
        tbz = 1;
        tby = 32;
        tbx = 32;
        gridz = 1;
        gridy = 1;
        gridx = (n-1) / num_elems_per_TB + 1;
        printf("GroupedDecoder config(%u %u %u) (%u %u %u)\n", tbx, tby, tbz, gridx, gridy, gridz);
        return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx); 
    }

    MGARDm_CONT
    void Execute(SIZE n,
                 SIZE num_batches_per_TB,
                 SIZE starting_bitplane,
                 SIZE num_bitplanes,
                 SIZE exp,
                 SubArray<2, T_bitplane> encoded_bitplanes,
                 SubArray<1, bool> signs,
                 SubArray<1, T> v,
                 int queue_idx) 
    {
      TaskType task = GenTask<T_fp, T_sfp>(n, 
                                          num_batches_per_TB, 
                                          starting_bitplane, 
                                          num_bitplanes, 
                                          exp, 
                                          encoded_bitplanes, 
                                          signs, 
                                          v, 
                                          queue_idx);
      DeviceAdapter<HandleType, TaskType, DeviceType> adapter(this->handle);
      adapter.Execute(task);
      this->handle.sync_all();
      // PrintSubarray("v", v);
    }
  };

    // general bitplane encoder that encodes data by block using T_stream type buffer
    template<DIM D, typename T_data, typename T_stream>
    class GroupedBPEncoderGPU : public concepts::BitplaneEncoderInterface<D, T_data> {
    public:
        GroupedBPEncoderGPU(Handle<D, T_data> &handle): _handle(handle) {
            std::cout <<  "GroupedBPEncoder\n";
            static_assert(std::is_floating_point<T_data>::value, "GeneralBPEncoder: input data must be floating points.");
            static_assert(!std::is_same<T_data, long double>::value, "GeneralBPEncoder: long double is not supported.");
            static_assert(std::is_unsigned<T_stream>::value, "GroupedBPBlockEncoder: streams must be unsigned integers.");
            static_assert(std::is_integral<T_stream>::value, "GroupedBPBlockEncoder: streams must be unsigned integers.");
        }

        std::vector<uint8_t *> encode(T_data const * data, int32_t n, int32_t exp, uint8_t num_bitplanes, std::vector<uint32_t>& stream_sizes) const {
            
            assert(num_bitplanes > 0);
            // determine block size based on bitplane integer type
            uint32_t block_size = block_size_based_on_bitplane_int_type<T_stream>();
            std::vector<uint8_t> starting_bitplanes = std::vector<uint8_t>((n - 1)/block_size + 1, 0);
            stream_sizes = std::vector<uint32_t>(num_bitplanes, 0);
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            std::vector<uint8_t *> streams;
            for(int i=0; i<num_bitplanes; i++){
                streams.push_back((uint8_t *) malloc(2 * n / UINT8_BITS + sizeof(T_stream)));
            }
            std::vector<T_fp> int_data_buffer(block_size, 0);
            std::vector<T_stream *> streams_pos(streams.size());
            for(int i=0; i<streams.size(); i++){
                streams_pos[i] = reinterpret_cast<T_stream*>(streams[i]);
            }
            T_data const * data_pos = data;
            int block_id=0;
            for(int i=0; i<n - block_size; i+=block_size){
                T_stream sign_bitplane = 0;
                for(int j=0; j<block_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    int64_t fix_point = (int64_t) shifted_data;
                    T_stream sign = cur_data < 0;
                    int_data_buffer[j] = sign ? -fix_point : +fix_point;
                    sign_bitplane += sign << j;
                }
                starting_bitplanes[block_id ++] = encode_block(int_data_buffer.data(), block_size, num_bitplanes, sign_bitplane, streams_pos);
            }
            // leftover
            {
                int rest_size = n - block_size * block_id;
                T_stream sign_bitplane = 0;
                for(int j=0; j<rest_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    int64_t fix_point = (int64_t) shifted_data;
                    T_stream sign = cur_data < 0;
                    int_data_buffer[j] = sign ? -fix_point : +fix_point;
                    sign_bitplane += sign << j;
                }
                starting_bitplanes[block_id ++] = encode_block(int_data_buffer.data(), rest_size, num_bitplanes, sign_bitplane, streams_pos);
            }
            for(int i=0; i<num_bitplanes; i++){
                stream_sizes[i] = reinterpret_cast<uint8_t*>(streams_pos[i]) - streams[i];
            }
            // merge starting_bitplane with the first bitplane
            uint32_t merged_size = 0;
            uint8_t * merged = merge_arrays(reinterpret_cast<uint8_t const*>(starting_bitplanes.data()), starting_bitplanes.size() * sizeof(uint8_t), reinterpret_cast<uint8_t*>(streams[0]), stream_sizes[0], merged_size);
            free(streams[0]);
            streams[0] = merged;
            stream_sizes[0] = merged_size;
            return streams;
        }

        // only differs in error collection
        std::vector<uint8_t *> encode(T_data const * data, int32_t n, int32_t exp, uint8_t num_bitplanes, std::vector<uint32_t>& stream_sizes, std::vector<double>& level_errors) const {
            assert(num_bitplanes > 0);
            // init level errors
            level_errors.clear();
            level_errors.resize(num_bitplanes + 1);
            for(int i=0; i<level_errors.size(); i++){
                level_errors[i] = 0;
            }
            stream_sizes = std::vector<uint32_t>(num_bitplanes, 0);
            

            Array<1, T_data> v_array({(SIZE)n});
            v_array.loadData(data);
            SubArray<1, T_data> v(v_array);

            Array<1, double> level_errors_array({(SIZE)num_bitplanes+1});
            SubArray<1, double> level_errors_subarray(level_errors_array);


            SIZE num_batches_per_TB = 2;
            using T_bitplane = uint32_t;
            const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
            const SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
            SIZE num_blocks = (n-1)/num_elems_per_TB+1;
            SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_blocks;

            Array<2, double> level_errors_work_array({(SIZE)num_bitplanes+1, num_blocks});
            SubArray<2, double> level_errors_work_subarray(level_errors_work_array);

            Array<2, T_bitplane> encoded_bitplanes_array({(SIZE)num_bitplanes, (SIZE)bitplane_max_length_total});
            SubArray<2, T_bitplane> encoded_bitplanes_subarray(encoded_bitplanes_array);
            
            GroupedEncoder<Handle<D, T_data>, T_data, T_bitplane, double, BINARY_TYPE, DATA_ENCODING_ALGORITHM, ERROR_COLLECTING_ALGORITHM, CUDA>(_handle).Execute(n, num_batches_per_TB, num_bitplanes, exp, v, encoded_bitplanes_subarray, level_errors_subarray, level_errors_work_subarray, 0);

            cudaMemcpyAsyncHelper(_handle, level_errors.data(), level_errors_subarray.data(), (num_bitplanes+1)* sizeof(double), AUTO, 0);
            _handle.sync_all();

            T_bitplane * encoded_bitplanes = encoded_bitplanes_array.getDataHost();

            std::vector<uint8_t *> streams2;
            for(int i=0; i<num_bitplanes; i++){
                stream_sizes[i] = bitplane_max_length_total*sizeof(T_bitplane);
                streams2.push_back((uint8_t *) malloc(bitplane_max_length_total*sizeof(T_bitplane)));
                memcpy(streams2[i], encoded_bitplanes + i * bitplane_max_length_total, bitplane_max_length_total*sizeof(T_bitplane)); 
            }
            return streams2;

            for(int i=0; i<level_errors.size(); i++){
                level_errors[i] = 0;
            }
            // determine block size based on bitplane integer type
            uint32_t block_size = block_size_based_on_bitplane_int_type<T_stream>();
            std::vector<uint8_t> starting_bitplanes = std::vector<uint8_t>((n - 1)/block_size + 1, 0);
            
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            std::vector<uint8_t *> streams;
            for(int i=0; i<num_bitplanes; i++){
                streams.push_back((uint8_t *) malloc(2 * n / UINT8_BITS + sizeof(T_stream)));
            }
            std::vector<T_fp> int_data_buffer(block_size, 0);
            std::vector<T_stream *> streams_pos(streams.size());
            for(int i=0; i<streams.size(); i++){
                streams_pos[i] = reinterpret_cast<T_stream*>(streams[i]);
            }
            
            T_data const * data_pos = data;
            int block_id=0;

            for(int i=0; i<(int)n - (int)block_size; i+=block_size){
                T_stream sign_bitplane = 0;
                for(int j=0; j<block_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    // compute level errors
                    collect_level_errors(level_errors, fabs(shifted_data), num_bitplanes);
                    int64_t fix_point = (int64_t) shifted_data;
                    T_stream sign = cur_data < 0;
                    int_data_buffer[j] = sign ? -fix_point : +fix_point;
                    sign_bitplane += sign << j;
                }
                starting_bitplanes[block_id ++] = encode_block(int_data_buffer.data(), block_size, num_bitplanes, sign_bitplane, streams_pos);
            }
            // leftover

            {
                int rest_size = n - block_size * block_id;
                T_stream sign_bitplane = 0;
                for(int j=0; j<rest_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    // compute level errors
                    collect_level_errors(level_errors, fabs(shifted_data), num_bitplanes);
                    int64_t fix_point = (int64_t) shifted_data;
                    T_stream sign = cur_data < 0;
                    int_data_buffer[j] = sign ? -fix_point : +fix_point;
                    sign_bitplane += sign << j;
                }
                starting_bitplanes[block_id ++] = encode_block(int_data_buffer.data(), rest_size, num_bitplanes, sign_bitplane, streams_pos);
            }


            for(int i=0; i<num_bitplanes; i++){
                stream_sizes[i] = reinterpret_cast<uint8_t*>(streams_pos[i]) - streams[i];
            }
            // merge starting_bitplane with the first bitplane
            uint32_t merged_size = 0;
            uint8_t * merged = merge_arrays(reinterpret_cast<uint8_t const*>(starting_bitplanes.data()), starting_bitplanes.size() * sizeof(uint8_t), reinterpret_cast<uint8_t*>(streams[0]), stream_sizes[0], merged_size);
            free(streams[0]);
            streams[0] = merged;
            stream_sizes[0] = merged_size;
            // translate level errors
            printf("error: ");
            for(int i=0; i<level_errors.size(); i++){
                level_errors[i] = ldexp(level_errors[i], 2*(- num_bitplanes + exp));
                printf("%f ", level_errors[i]);
            }
            printf("\n");


            return streams;
        }

        T_data * decode(const std::vector<uint8_t const *>& streams, int32_t n, int exp, uint8_t num_bitplanes) {
            uint32_t block_size = block_size_based_on_bitplane_int_type<T_stream>();
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            T_data * data = (T_data *) malloc(n * sizeof(T_data));
            if(num_bitplanes == 0){
                memset(data, 0, n * sizeof(T_data));
                return data;
            }
            std::vector<T_stream const *> streams_pos(streams.size());
            for(int i=0; i<streams.size(); i++){
                streams_pos[i] = reinterpret_cast<T_stream const *>(streams[i]);
            }
            // deinterleave the first bitplane
            uint32_t recording_bitplane_size = *reinterpret_cast<int32_t const*>(streams_pos[0]);
            uint8_t const * recording_bitplanes = reinterpret_cast<uint8_t const*>(streams_pos[0]) + sizeof(uint32_t);
            streams_pos[0] = reinterpret_cast<T_stream const *>(recording_bitplanes + recording_bitplane_size);

            std::vector<T_fp> int_data_buffer(block_size, 0);
            // decode
            T_data * data_pos = data;
            int block_id = 0;
            for(int i=0; i<n - block_size; i+=block_size){
                uint8_t recording_bitplane = recording_bitplanes[block_id ++];
                if(recording_bitplane < num_bitplanes){
                    memset(int_data_buffer.data(), 0, block_size * sizeof(T_fp));
                    T_stream sign_bitplane = *(streams_pos[recording_bitplane] ++);
                    decode_block(streams_pos, block_size, recording_bitplane, num_bitplanes - recording_bitplane, int_data_buffer.data());
                    for(int j=0; j<block_size; j++, sign_bitplane >>= 1){
                        T_data cur_data = ldexp((T_data)int_data_buffer[j], - num_bitplanes + exp);
                        *(data_pos++) = (sign_bitplane & 1u) ? -cur_data : cur_data;
                    }
                }
                else{
                    for(int j=0; j<block_size; j++){
                        *(data_pos ++) = 0;
                    }
                }
            }
            // leftover
            {
                int rest_size = n - block_size * block_id;
                int recording_bitplane = recording_bitplanes[block_id];
                T_stream sign_bitplane = 0;
                if(recording_bitplane < num_bitplanes){
                    memset(int_data_buffer.data(), 0, block_size * sizeof(T_fp));
                    sign_bitplane = *(streams_pos[recording_bitplane] ++);
                    decode_block(streams_pos, block_size, recording_bitplane, num_bitplanes - recording_bitplane, int_data_buffer.data());
                    for(int j=0; j<rest_size; j++, sign_bitplane >>= 1){
                        T_data cur_data = ldexp((T_data)int_data_buffer[j], - num_bitplanes + exp);
                        *(data_pos++) = (sign_bitplane & 1u) ? -cur_data : cur_data;
                    }
                }
                else{
                    for(int j=0; j<block_size; j++){
                        *(data_pos ++) = 0;
                    }
                }
            }
            return data;
        }

        // decode the data and record necessary information for progressiveness
        T_data * progressive_decode(const std::vector<uint8_t const *>& streams, int32_t n, int exp, uint8_t starting_bitplane, uint8_t num_bitplanes, int level) {
            T_data * data = (T_data *) malloc(n * sizeof(T_data));
            if(num_bitplanes == 0){
                memset(data, 0, n * sizeof(T_data));
                return data;
            }

            if(level_signs.size() == level){
              level_signs.push_back(std::vector<bool>(n, false));
            }
            std::vector<bool>& signs = level_signs[level];
            bool * new_signs = new bool[n];
            for (int i = 0; i < n; i++) {
              new_signs[i] = signs[i];
            }
            
            SIZE num_batches_per_TB = 2;
            using T_bitplane = uint32_t;
            const SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
            const SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
            SIZE num_blocks = (n-1)/num_elems_per_TB+1;
            SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_blocks;

            T_bitplane * encoded_bitplanes = new T_bitplane[bitplane_max_length_total * num_bitplanes];
            for (int i = 0; i < num_bitplanes; i++) {
               memcpy(encoded_bitplanes + i * bitplane_max_length_total, streams[i], bitplane_max_length_total * sizeof(T_bitplane));
            }
            Array<2, T_bitplane> encoded_bitplanes_array({(SIZE)num_bitplanes, (SIZE)bitplane_max_length_total});
            encoded_bitplanes_array.loadData(encoded_bitplanes);
            SubArray<2, T_bitplane> encoded_bitplanes_subarray(encoded_bitplanes_array);

            Array<1, bool> signs_array({(SIZE)n});
            signs_array.loadData(new_signs);
            SubArray<1, bool> signs_subarray(signs_array);

            delete [] new_signs;

            Array<1, T_data> v_array({(SIZE)n});
            SubArray<1, T_data> v(v_array);

            GroupedDecoder<Handle<D, T_data>, T_data, T_bitplane, BINARY_TYPE, DATA_DECODING_ALGORITHM, CUDA>(_handle).Execute(n, num_batches_per_TB, starting_bitplane, num_bitplanes, exp, encoded_bitplanes_subarray, signs_subarray, v, 0);

            new_signs = signs_array.getDataHost();

            for (int i = 0; i < n; i++) {
              signs[i] = new_signs[i];
            }

            T_data * temp_data = v_array.getDataHost();
            memcpy(data, temp_data, n * sizeof(T_data));
            return data;


            uint32_t block_size = block_size_based_on_bitplane_int_type<T_stream>();
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            
            std::vector<T_stream const *> streams_pos(streams.size());
            for(int i=0; i<streams.size(); i++){
                streams_pos[i] = reinterpret_cast<T_stream const *>(streams[i]);
            }
            if(level_recording_bitplanes.size() == level){
                // deinterleave the first bitplane
                uint32_t recording_bitplane_size = *reinterpret_cast<int32_t const*>(streams_pos[0]);
                uint8_t const * recording_bitplanes_pos = reinterpret_cast<uint8_t const*>(streams_pos[0]) + sizeof(uint32_t);
                auto recording_bitplanes = std::vector<uint8_t>(recording_bitplanes_pos, recording_bitplanes_pos + recording_bitplane_size);
                level_recording_bitplanes.push_back(recording_bitplanes);
                streams_pos[0] = reinterpret_cast<T_stream const *>(recording_bitplanes_pos + recording_bitplane_size);
            }

            std::vector<T_fp> int_data_buffer(block_size, 0);

            const std::vector<uint8_t>& recording_bitplanes = level_recording_bitplanes[level];
            const uint8_t ending_bitplane = starting_bitplane + num_bitplanes;
            // decode
            T_data * data_pos = data;
            int block_id = 0;
            for(int i=0; i<n - block_size; i+=block_size){
                uint8_t recording_bitplane = recording_bitplanes[block_id ++];
                if(recording_bitplane < ending_bitplane){
                    memset(int_data_buffer.data(), 0, block_size * sizeof(T_fp));
                    if(recording_bitplane >= starting_bitplane){
                        // have not recorded signs for this block
                        T_stream sign_bitplane = *(streams_pos[recording_bitplane - starting_bitplane] ++);
                        for(int j=0; j<block_size; j++, sign_bitplane >>= 1){
                            signs[i + j] = sign_bitplane & 1u;
                        }
                        decode_block(streams_pos, block_size, recording_bitplane - starting_bitplane, ending_bitplane - recording_bitplane, int_data_buffer.data());
                    }                    
                    else{
                        decode_block(streams_pos, block_size, 0, num_bitplanes, int_data_buffer.data());                    
                    }
                    for(int j=0; j<block_size; j++){
                        T_data cur_data = ldexp((T_data)int_data_buffer[j], - ending_bitplane + exp);
                        *(data_pos++) = signs[i + j] ? -cur_data : cur_data;
                    }
                }
                else{
                    for(int j=0; j<block_size; j++){
                        *(data_pos ++) = 0;
                    }
                }
            }
            // leftover
            {
                int rest_size = n - block_size * block_id;
                uint8_t recording_bitplane = recording_bitplanes[block_id];
                if(recording_bitplane < ending_bitplane){
                    memset(int_data_buffer.data(), 0, block_size * sizeof(T_fp));
                    if(recording_bitplane >= starting_bitplane){
                        // have not recorded signs for this block
                        T_stream sign_bitplane = *(streams_pos[recording_bitplane - starting_bitplane] ++);
                        for(int j=0; j<rest_size; j++, sign_bitplane >>= 1){
                            signs[block_size * block_id + j] = sign_bitplane & 1u;
                        }
                        decode_block(streams_pos, rest_size, recording_bitplane - starting_bitplane, ending_bitplane - recording_bitplane, int_data_buffer.data());
                    }
                    else{
                        decode_block(streams_pos, rest_size, 0, num_bitplanes, int_data_buffer.data());                    
                    }
                    for(int j=0; j<rest_size; j++){
                        T_data cur_data = ldexp((T_data)int_data_buffer[j], - ending_bitplane + exp);
                        *(data_pos++) = signs[block_size * block_id + j] ? -cur_data : cur_data;
                    }
                }
                else{
                    for(int j=0; j<rest_size; j++){
                        *(data_pos ++) = 0;
                    }
                }
            }
            return data;
        }

        void print() const {
            std::cout << "Grouped bitplane encoder" << std::endl;
        }
    private:
        template<class T>
        uint32_t block_size_based_on_bitplane_int_type() const {
            uint32_t block_size = 0;
            if(std::is_same<T, uint64_t>::value){
                block_size = 64;
            }
            else if(std::is_same<T, uint32_t>::value){
                block_size = 32;
            }
            else if(std::is_same<T, uint16_t>::value){
                block_size = 16;
            }
            else if(std::is_same<T, uint8_t>::value){
                block_size = 8;
            }
            else{
                std::cerr << "Integer type not supported." << std::endl;
                exit(0);
            }
            return block_size;
        }
        inline void collect_level_errors(std::vector<double>& level_errors, float data, int num_bitplanes) const {
            uint32_t fp_data = (uint32_t) data;
            double mantissa = data - (uint32_t) data;
            level_errors[num_bitplanes] += mantissa * mantissa;
            for(int k=1; k<num_bitplanes; k++){
                uint32_t mask = (1 << k) - 1;
                double diff = (double) (fp_data & mask) + mantissa;
                level_errors[num_bitplanes - k] += diff * diff;
            }
            double diff = fp_data + mantissa;
            level_errors[0] += data * data;
        }

        template <class T_int>
        inline uint8_t encode_block(T_int const * data, size_t n, uint8_t num_bitplanes, T_stream sign, std::vector<T_stream *>& streams_pos) const {
            bool recorded = false;
            uint8_t recording_bitplane = num_bitplanes;
            for(int k=num_bitplanes - 1; k>=0; k--){
                T_stream bitplane_value = 0;
                T_stream bitplane_index = num_bitplanes - 1 - k;
                for (int i=0; i<n; i++){
                    bitplane_value += (T_stream)((data[i] >> k) & 1u) << i;
                }
                if(bitplane_value || recorded){
                    if(!recorded){
                        recorded = true;
                        recording_bitplane = bitplane_index;
                        *(streams_pos[bitplane_index] ++) = sign;
                    }
                    *(streams_pos[bitplane_index] ++) = bitplane_value;
                }
            }
            return recording_bitplane;
        }

        template <class T_int>
        inline void decode_block(std::vector<T_stream const *>& streams_pos, size_t n, uint8_t recording_bitplane, uint8_t num_bitplanes, T_int * data) const {
            for(int k=num_bitplanes - 1; k>=0; k--){
                T_stream bitplane_index = recording_bitplane + num_bitplanes - 1 - k;
                T_stream bitplane_value = *(streams_pos[bitplane_index] ++);
                for (int i=0; i<n; i++){
                    data[i] += ((bitplane_value >> i) & 1u) << k;
                }
            }
        }

        uint8_t * merge_arrays(uint8_t const * array1, uint32_t size1, uint8_t const * array2, uint32_t size2, uint32_t& merged_size) const {
            merged_size = sizeof(uint32_t) + size1 + size2;
            uint8_t * merged_array = (uint8_t *) malloc(merged_size);
            *reinterpret_cast<uint32_t*>(merged_array) = size1;
            memcpy(merged_array + sizeof(uint32_t), array1, size1);
            memcpy(merged_array + sizeof(uint32_t) + size1, array2, size2);
            return merged_array;
        }

        Handle<D, T_data> &_handle;
        std::vector<std::vector<bool>> level_signs;
        std::vector<std::vector<uint8_t>> level_recording_bitplanes;
    };
}
}



namespace mgard_m {
namespace MDR {
// general bitplane encoder that encodes data by block using T_stream type buffer
template<typename HandleType, mgard_cuda::DIM D, typename T_data, typename T_bitplane, typename T_error>
class GroupedBPEncoder : public concepts::BitplaneEncoderInterface<HandleType, D, T_data, T_bitplane, T_error> {
public:
  GroupedBPEncoder(HandleType& handle): handle(handle) {
    std::cout <<  "GroupedBPEncoder\n";
    static_assert(std::is_floating_point<T_data>::value, "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value, "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value, "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value, "GroupedBPBlockEncoder: streams must be unsigned integers.");
  }

  mgard_cuda::Array<2, T_bitplane> encode(mgard_cuda::SIZE n, mgard_cuda::SIZE num_bitplanes, int32_t exp, 
              mgard_cuda::SubArray<1, T_data> v,
              mgard_cuda::SubArray<1, T_error> level_errors,
              std::vector<mgard_cuda::SIZE>& streams_sizes, int queue_idx) const {

    mgard_cuda::SIZE num_batches_per_TB = 2;
    const mgard_cuda::SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const mgard_cuda::SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    mgard_cuda::SIZE num_blocks = (n-1)/num_elems_per_TB+1;
    mgard_cuda::SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_blocks;

    mgard_cuda::Array<2, T_error> level_errors_work_array({(mgard_cuda::SIZE)num_bitplanes+1, num_blocks});
    mgard_cuda::SubArray<2, T_error> level_errors_work(level_errors_work_array);
    
    mgard_cuda::Array<2, T_bitplane> encoded_bitplanes_array({(mgard_cuda::SIZE)num_bitplanes, (mgard_cuda::SIZE)bitplane_max_length_total});
    mgard_cuda::SubArray<2, T_bitplane> encoded_bitplanes_subarray(encoded_bitplanes_array);


    mgard_cuda::MDR::GroupedEncoder<HandleType, T_data, T_bitplane, T_error, BINARY_TYPE, DATA_ENCODING_ALGORITHM, ERROR_COLLECTING_ALGORITHM, mgard_cuda::CUDA>(handle).
                  Execute(n, num_batches_per_TB, num_bitplanes, exp, 
                          v, encoded_bitplanes_subarray, level_errors, level_errors_work, 
                          queue_idx);

    for(int i=0; i<num_bitplanes; i++){
        streams_sizes[i] = bitplane_max_length_total*sizeof(T_bitplane);
    }

    return encoded_bitplanes_array;
  }

  mgard_cuda::Array<1, T_data> decode(mgard_cuda::SIZE n, mgard_cuda::SIZE num_bitplanes, int32_t exp, 
                  mgard_cuda::SubArray<2, T_bitplane> encoded_bitplanes, int level,
                  int queue_idx) {

  }

  // decode the data and record necessary information for progressiveness
  mgard_cuda::Array<1, T_data> progressive_decode(mgard_cuda::SIZE n, mgard_cuda::SIZE starting_bitplane, mgard_cuda::SIZE num_bitplanes, int32_t exp, 
                          mgard_cuda::SubArray<2, T_bitplane> encoded_bitplanes, int level,
                          int queue_idx) {

    mgard_cuda::SIZE num_batches_per_TB = 2;
    const mgard_cuda::SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const mgard_cuda::SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    mgard_cuda::SIZE num_blocks = (n-1)/num_elems_per_TB+1;
    mgard_cuda::SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_blocks;

    if(level_signs.size() == level){
      level_signs.push_back(mgard_cuda::Array<1, bool>({n}));
    }


    // uint8_t * encoded_bitplanes = new uint8_t[bitplane_max_length_total * num_bitplanes];
    // for (int i = 0; i < num_bitplanes; i++) {
       // memcpy(encoded_bitplanes + i * bitplane_max_length_total, streams[i], bitplane_max_length_total * sizeof(uint8_t));
    // }
    // Array<2, uint8_t> encoded_bitplanes_array({(SIZE)num_bitplanes, (SIZE)bitplane_max_length_total});
    // encoded_bitplanes_array.loadData(encoded_bitplanes);
    // SubArray<2, uint8_t> encoded_bitplanes_subarray(encoded_bitplanes_array);

    // Array<1, bool> signs_array({(SIZE)n});
    // signs_array.loadData(new_signs);
    mgard_cuda::SubArray<1, bool> signs_subarray(level_signs[level]);

    mgard_cuda::Array<1, T_data> v_array({(mgard_cuda::SIZE)n});
    mgard_cuda::SubArray<1, T_data> v(v_array);


    // delete [] new_signs;

    // Array<1, T_data> v_array({(SIZE)n});
    // SubArray<1, T_data> v(v_array);

    mgard_cuda::MDR::GroupedDecoder<HandleType, T_data, T_bitplane, BINARY_TYPE, 
                  DATA_DECODING_ALGORITHM, mgard_cuda::CUDA>(handle).
                  Execute(n, num_batches_per_TB, starting_bitplane, num_bitplanes, exp, 
                  encoded_bitplanes, signs_subarray, v, queue_idx);

    // new_signs = signs_array.getDataHost();

    // for (int i = 0; i < n; i++) {
    //   signs[i] = new_signs[i];
    // }

    // T_data * temp_data = v_array.getDataHost();
    // memcpy(data, temp_data, n * sizeof(T_data));
    // return data;
    return v_array;
  }


  mgard_cuda::SIZE buffer_size(mgard_cuda::SIZE n) const {
    mgard_cuda::SIZE num_batches_per_TB = 2;
    const mgard_cuda::SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
    const mgard_cuda::SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
    mgard_cuda::SIZE num_blocks = (n-1)/num_elems_per_TB+1;
    mgard_cuda::SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_blocks;
    return bitplane_max_length_total;
  }


  void print() const {
    std::cout << "Grouped bitplane encoder" << std::endl;
  }
private:
HandleType& handle;
std::vector<mgard_cuda::Array<1, bool>> level_signs;
std::vector<std::vector<uint8_t>> level_recording_bitplanes;

};
}
}
#endif
