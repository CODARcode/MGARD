/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_ENCODE_FIXED_LEN_TEMPLATE_HPP
#define MGARD_X_ENCODE_FIXED_LEN_TEMPLATE_HPP

#include "../CommonInternal.h"

namespace mgard_x {

// Jieyang: this kernel rely on whole grid sychronized execution
// For example, adding
//   if (thread == 0) {
//      __nanosleep(1e9);
//   }
// will cause incorrect results

template <typename Q, typename H, typename DeviceType>
class EncodeFixedLenFunctor: public Functor<DeviceType> {
  public:
  MGARDm_CONT EncodeFixedLenFunctor(SubArray<1, Q, DeviceType> data, 
                                    SubArray<1, H, DeviceType> hcoded, 
                                    SIZE data_len,
                                    SubArray<1, H, DeviceType> codebook):
                                  data(data), hcoded(hcoded), data_len(data_len),
                                  codebook(codebook) {
    Functor<DeviceType>();                            
  }

  MGARDm_EXEC void
  Operation1() {
    unsigned int gid = (this->blockx * this->nblockx) + this->threadx;
    if (gid >= data_len)
      return;
    *hcoded(gid) = *codebook(*data(gid)); // try to exploit cache?
  }

  MGARDm_EXEC void
  Operation2() { }

  MGARDm_EXEC void
  Operation3() { }

  MGARDm_EXEC void
  Operation4() { }

  MGARDm_EXEC void
  Operation5() { }

  MGARDm_CONT size_t
  shared_memory_size() { return 0; }

  private:
  SubArray<1, Q, DeviceType> data;
  SubArray<1, H, DeviceType> hcoded;
  SIZE data_len; 
  SubArray<1, H, DeviceType> codebook;
};


template <typename Q, typename H, typename DeviceType>
class EncodeFixedLen: public AutoTuner<DeviceType> {
public:
  MGARDm_CONT
  EncodeFixedLen():AutoTuner<DeviceType>() {}

  MGARDm_CONT
  Task<EncodeFixedLenFunctor<Q, H, DeviceType> > 
  GenTask(SubArray<1, Q, DeviceType> data, SubArray<1, H, DeviceType> hcoded, 
              SIZE data_len, SubArray<1, H, DeviceType> codebook, int queue_idx) {
    using FunctorType = EncodeFixedLenFunctor<Q, H, DeviceType>;
    FunctorType functor(data, hcoded, data_len, codebook);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerSM();
    gridz = 1;
    gridy = 1;
    gridx = (data_len - 1) / tbx + 1;
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "EncodeFixedLen"); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, Q, DeviceType> data, SubArray<1, H, DeviceType> hcoded, 
              SIZE data_len, SubArray<1, H, DeviceType> codebook, int queue_idx) {
    using FunctorType = EncodeFixedLenFunctor<Q, H, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(data, hcoded, data_len, codebook, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endif