/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HUFFMAN_HISTOGRAM_TEMPLATE_HPP
#define MGARD_X_HUFFMAN_HISTOGRAM_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {
template <typename T, typename Q, bool CACHE_HISTOGRAM, typename DeviceType>
class HistogramFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT HistogramFunctor() {}
  MGARDX_CONT HistogramFunctor(SubArray<1, T, DeviceType> input_data,
                               SubArray<1, int, DeviceType> local_histogram,
                               SubArray<1, Q, DeviceType> output, SIZE N,
                               int bins, int RPerBlock)
      : input_data(input_data), local_histogram(local_histogram),
        output(output), N(N), bins(bins), RPerBlock(RPerBlock) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    if (CACHE_HISTOGRAM) {
      Hs = (int *)FunctorBase<DeviceType>::GetSharedMemory();
    } else {
      Hs = local_histogram(FunctorBase<DeviceType>::GetBlockIdX() * RPerBlock *
                           bins);
    }

    warpid = (int)(FunctorBase<DeviceType>::GetThreadIdX() / MGARDX_WARP_SIZE);
    lane = FunctorBase<DeviceType>::GetThreadIdX() % MGARDX_WARP_SIZE;
    warps_block = FunctorBase<DeviceType>::GetBlockDimX() / MGARDX_WARP_SIZE;

    off_rep = (bins) * (FunctorBase<DeviceType>::GetThreadIdX() % RPerBlock);

    begin = (N / warps_block) * warpid +
            MGARDX_WARP_SIZE * FunctorBase<DeviceType>::GetBlockIdX() + lane;
    end = (N / warps_block) * (warpid + 1);
    step = MGARDX_WARP_SIZE * FunctorBase<DeviceType>::GetGridDimX();

    // final warp handles data outside of the warps_block partitions
    if (warpid >= warps_block - 1)
      end = N;

    if (CACHE_HISTOGRAM) {
      for (unsigned int pos = FunctorBase<DeviceType>::GetThreadIdX();
           pos < (bins)*RPerBlock;
           pos += FunctorBase<DeviceType>::GetBlockDimX()) {
        Hs[pos] = 0;
      }
    }
  }

  MGARDX_EXEC void Operation2() {
    for (unsigned int i = begin; i < end; i += step) {
      int d = *input_data(i);
      if (CACHE_HISTOGRAM) {
        Atomic<int, AtomicSharedMemory, AtomicDeviceScope, DeviceType>::Add(
            &Hs[off_rep + d], 1);
      } else {
        Atomic<int, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>::Add(
            &Hs[off_rep + d], 1);
      }
    }
  }

  MGARDX_EXEC void Operation3() {
    for (unsigned int pos = FunctorBase<DeviceType>::GetThreadIdX(); pos < bins;
         pos += FunctorBase<DeviceType>::GetBlockDimX()) {
      int sum = 0;
      for (int base = 0; base < (bins)*RPerBlock; base += bins) {
        sum += Hs[base + pos];
      }
      Atomic<Q, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>::Add(
          output(pos), (Q)sum);
    }
  }

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    if (CACHE_HISTOGRAM) {
      size_t size = 0;
      size = (bins)*RPerBlock * sizeof(int);
      return size;
    } else {
      return 0;
    }
  }

private:
  SubArray<1, T, DeviceType> input_data;
  SubArray<1, int, DeviceType> local_histogram;
  SubArray<1, Q, DeviceType> output;
  SIZE N;
  int bins;
  int RPerBlock;

  int *Hs;

  unsigned int warpid;
  unsigned int lane;
  unsigned int warps_block;

  unsigned int off_rep;

  unsigned int begin;
  unsigned int end;
  unsigned int step;
};

template <typename T, typename Q, bool CACHE_HISTOGRAM, typename DeviceType>
class HistogramKernel : public Kernel {
public:
  constexpr static std::string_view Name = "histogram";
  constexpr static bool EnableAutoTuning() { return false; }

  MGARDX_CONT
  HistogramKernel(SubArray<1, T, DeviceType> input_data,
                  SubArray<1, int, DeviceType> local_histogram,
                  SubArray<1, Q, DeviceType> output, SIZE N, int bins,
                  int RPerBlock, int threadsPerBlock, int numBlocks)
      : input_data(input_data), local_histogram(local_histogram),
        output(output), N(N), bins(bins), RPerBlock(RPerBlock),
        threadsPerBlock(threadsPerBlock), numBlocks(numBlocks) {}

  MGARDX_CONT Task<HistogramFunctor<T, Q, CACHE_HISTOGRAM, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = HistogramFunctor<T, Q, CACHE_HISTOGRAM, DeviceType>;

    FunctorType functor(input_data, local_histogram, output, N, bins,
                        RPerBlock);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = threadsPerBlock;
    gridz = 1;
    gridy = 1;
    gridx = numBlocks;

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> input_data;
  SubArray<1, int, DeviceType> local_histogram;
  SubArray<1, Q, DeviceType> output;
  SIZE N;
  int bins;
  int RPerBlock;
  int threadsPerBlock;
  int numBlocks;
};

template <typename DeviceType>
MGARDX_CONT void ExecutionConfig(int N, int bins, int RPerBlock,
                                 int &threadsPerBlock, int &numBlocks) {
  int numSMs = DeviceRuntime<DeviceType>::GetNumSMs();
  int numBuckets = bins;
  int numValues = N;
  int itemsPerThread = 1;
  numBlocks = numSMs;

  threadsPerBlock =
      ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
  while (threadsPerBlock > DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB()) {
    if (RPerBlock <= 1) {
      threadsPerBlock = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB();
    } else {
      RPerBlock /= 2;
      numBlocks *= 2;
      threadsPerBlock =
          ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
    }
  }
}

template <typename T, typename Q, typename DeviceType>
MGARDX_CONT void Histogram(SubArray<1, T, DeviceType> input_data,
                           SubArray<1, Q, DeviceType> output, SIZE N, int bins,
                           int queue_idx) {
  int maxbytes = DeviceRuntime<DeviceType>::GetMaxSharedMemorySize();
  SubArray<1, int, DeviceType> local_histogram;
  if (bins * sizeof(int) < maxbytes) {
    if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
      std::cout << log::log_info
                << "Histogram: using shared memory for local histogram\n";
    }
    using FunctorType = HistogramFunctor<T, Q, true, DeviceType>;
    using TaskType = Task<FunctorType>;
    int RPerBlock = (maxbytes / (int)sizeof(int)) / (bins);
    if (std::is_same<DeviceType, SERIAL>::value ||
        std::is_same<DeviceType, OPENMP>::value) {
      RPerBlock = 1;
    }
    int threadsPerBlock, numBlocks;
    ExecutionConfig<DeviceType>(N, bins, RPerBlock, threadsPerBlock, numBlocks);
    DeviceLauncher<DeviceType>::Execute(
        HistogramKernel<T, Q, true, DeviceType>(input_data, local_histogram,
                                                output, N, bins, RPerBlock,
                                                threadsPerBlock, numBlocks),
        queue_idx);
  } else {
    if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
      std::cout << log::log_info
                << "Histogram: using global memory for local histogram\n";
    }
    using FunctorType = HistogramFunctor<T, Q, false, DeviceType>;
    using TaskType = Task<FunctorType>;
    int RPerBlock = 2;
    int threadsPerBlock, numBlocks;
    ExecutionConfig<DeviceType>(N, bins, RPerBlock, threadsPerBlock, numBlocks);
    Array<1, int, DeviceType> local_histogram_array(
        {(SIZE)RPerBlock * bins * numBlocks}, false, true);
    local_histogram_array.memset(0);
    // TODO: can we not sync all queues?
    DeviceRuntime<DeviceType>::SyncAllQueues();
    local_histogram = SubArray(local_histogram_array);
    DeviceLauncher<DeviceType>::Execute(
        HistogramKernel<T, Q, false, DeviceType>(input_data, local_histogram,
                                                 output, N, bins, RPerBlock,
                                                 threadsPerBlock, numBlocks),
        queue_idx);
    // Make sure the work is does before we free local_histogram_array
    DeviceRuntime<DeviceType>::SyncAllQueues();
  }
}

} // namespace mgard_x

#endif