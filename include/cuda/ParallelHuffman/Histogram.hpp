/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_HUFFMAN_HISTOGRAM_TEMPLATE_HPP
#define MGRAD_CUDA_HUFFMAN_HISTOGRAM_TEMPLATE_HPP


#include "../CommonInternal.h"
#include "../Functor.h"
#include "../AutoTuners/AutoTuner.h"
#include "../Task.h"
#include "../DeviceAdapters/DeviceAdapter.h"

namespace mgard_cuda {
  template <typename T, typename Q, typename DeviceType>
  class HistogramFunctor: public Functor<DeviceType> {
    public:
    MGARDm_CONT HistogramFunctor(SubArray<1, T, DeviceType> input_data, 
                                 SubArray<1, Q, DeviceType> output, 
                                 SIZE N, 
                                 int bins, int R):
                                input_data(input_data), output(output), N(N), bins(bins), R(R) {
      Functor<DeviceType>();                            
    }

    MGARDm_EXEC void
    Operation1() {
      Hs = (int*)this->shared_memory;

      warpid = (int)(this->threadx / MGARDm_WARP_SIZE);
      lane = this->threadx % MGARDm_WARP_SIZE;
      warps_block = this->nblockx / MGARDm_WARP_SIZE;

      off_rep = (bins + 1) * (this->threadx % R);

      begin = (N / warps_block) * warpid + MGARDm_WARP_SIZE * this->blockx + lane;
      end = (N / warps_block) * (warpid + 1);
      step = MGARDm_WARP_SIZE * this->ngridx;

      // final warp handles data outside of the warps_block partitions
      if (warpid >= warps_block - 1)
        end = N;

      for (unsigned int pos = this->threadx; pos < (bins + 1) * R; pos += this->nblockx)
        Hs[pos] = 0;
    }

    MGARDm_EXEC void
    Operation2() {
      for (unsigned int i = begin; i < end; i += step) {
        int d = *input_data(i);
        atomicAdd(&Hs[off_rep + d], 1);
      }
    }

    MGARDm_EXEC void
    Operation3() {
      for (unsigned int pos = this->threadx; pos < bins; pos += this->nblockx) {
        int sum = 0;
        for (int base = 0; base < (bins + 1) * R; base += bins + 1) {
          sum += Hs[base + pos];
        }
        atomicAdd(output(pos), sum);
      }
    }

    MGARDm_EXEC void
    Operation4() { }

    MGARDm_EXEC void
    Operation5() { }

    MGARDm_CONT size_t
    shared_memory_size() {
      size_t size = 0;
      size = (bins + 1) * R * sizeof(int);
      return size;
    } 

    private:
    SubArray<1, T, DeviceType> input_data;
    SubArray<1, Q, DeviceType> output;
    SIZE N; 
    int bins;
    int R;

    int * Hs;

    unsigned int warpid;
    unsigned int lane;
    unsigned int warps_block;

    unsigned int off_rep;

    unsigned int begin;
    unsigned int end;
    unsigned int step;
  };

  template <typename T, typename Q, typename DeviceType>
  class Histogram: public AutoTuner<DeviceType> {
  public:
    MGARDm_CONT
    Histogram():AutoTuner<DeviceType>() {}

    MGARDm_CONT
    Task<HistogramFunctor<T, Q, DeviceType> > 
    GenTask(SubArray<1, T, DeviceType> input_data, SubArray<1, Q, DeviceType> output, SIZE len, 
            int dict_size, int queue_idx) {
      using FunctorType = HistogramFunctor<T, Q, DeviceType>;

      int maxbytes = DeviceRuntime<DeviceType>::GetMaxSharedMemorySize();
      int numSMs = DeviceRuntime<DeviceType>::GetNumSMs();

      int numBuckets = dict_size;
      int numValues = len;
      int itemsPerThread = 1;
      int RPerBlock = (maxbytes / (int)sizeof(int)) / (numBuckets + 1);
      int numBlocks = numSMs;

      int threadsPerBlock =
          ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
      while (threadsPerBlock > 1024) {
        if (RPerBlock <= 1) {
          threadsPerBlock = 1024;
        } else {
          RPerBlock /= 2;
          numBlocks *= 2;
          threadsPerBlock =
              ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
        }
      }

      FunctorType functor(input_data, output, len, dict_size, RPerBlock);

      SIZE tbx, tby, tbz, gridx, gridy, gridz;
      size_t sm_size = functor.shared_memory_size();
      tbz = 1;
      tby = 1;
      tbx = threadsPerBlock;
      gridz = 1;
      gridy = 1;
      gridx = numBlocks;
      // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
      // PrintSubarray("shape", shape);
      return Task(functor, gridz, gridy, gridx, 
                  tbz, tby, tbx, sm_size, queue_idx); 
    }

    MGARDm_CONT
    void Execute(SubArray<1, T, DeviceType> input_data, SubArray<1, Q, DeviceType> output, SIZE len, 
                  int dict_size, int queue_idx) {
      using FunctorType = HistogramFunctor<T, Q, DeviceType>;
      using TaskType = Task<FunctorType>;
      TaskType task = GenTask(input_data, output, len, dict_size, queue_idx); 
      DeviceAdapter<TaskType, DeviceType> adapter; 
      adapter.Execute(task);
    }
  };


}

#endif