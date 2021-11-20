/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_DECODE_TEMPLATE_HPP
#define MGRAD_CUDA_DECODE_TEMPLATE_HPP

#include "../CommonInternal.h"

namespace mgard_cuda {

template <typename Q, typename H, typename DeviceType>
class DecodeFunctor: public Functor<DeviceType> {
  public:
  MGARDm_CONT DecodeFunctor(SubArray<1, H, DeviceType> densely, 
                            SubArray<1, size_t, DeviceType> dH_meta,
                            SubArray<1, Q, DeviceType> bcode,
                            SIZE len, int chunk_size, int n_chunk, 
                            SubArray<1, uint8_t, DeviceType> singleton, 
                            size_t singleton_size
                            ): densely(densely), dH_meta(dH_meta),
                             bcode(bcode), len(len), chunk_size(chunk_size),
                             n_chunk(n_chunk), singleton(singleton),
                             singleton_size(singleton_size){
    Functor<DeviceType>();                            
  }

  MGARDm_EXEC void
  Operation1() {
    _s_singleton = (uint8_t*)this->shared_memory;
    if (this->threadx == 0) {
      memcpy(_s_singleton, singleton((IDX)0), singleton_size);
    }
  }

  MGARDm_EXEC void
  Operation2() { 
    size_t chunk_id = this->blockx * this->nblockx + this->threadx;
    // if (chunk_id == 0) printf("n_chunk: %lu\n", n_chunk);
    if (chunk_id >= n_chunk)
      return;

    SIZE densely_offset = *dH_meta(n_chunk + chunk_id);
    SIZE bcode_offset = chunk_size * chunk_id;
    size_t total_bw = *dH_meta(chunk_id);

    uint8_t next_bit;
    size_t idx_bit;
    size_t idx_byte = 0;
    size_t idx_bcoded = 0;
    auto first = reinterpret_cast<H *>(_s_singleton);
    auto entry = first + sizeof(H) * 8;
    auto keys =
        reinterpret_cast<Q *>(_s_singleton + sizeof(H) * (2 * sizeof(H) * 8));
    H v = (*densely(densely_offset + idx_byte) >> (sizeof(H) * 8 - 1)) & 0x1; // get the first bit
    size_t l = 1;
    size_t i = 0;
    while (i < total_bw) {
      while (v < first[l]) { // append next i_cb bit
        ++i;
        idx_byte = i / (sizeof(H) * 8);
        idx_bit = i % (sizeof(H) * 8);
        next_bit = ((*densely(densely_offset+idx_byte) >> (sizeof(H) * 8 - 1 - idx_bit)) & 0x1);
        v = (v << 1) | next_bit;
        ++l;
      }
      *bcode(bcode_offset + idx_bcoded) = keys[entry[l] + v - first[l]];
      idx_bcoded++;
      {
        ++i;
        idx_byte = i / (sizeof(H) * 8);
        idx_bit = i % (sizeof(H) * 8);
        next_bit = ((*densely(densely_offset+idx_byte) >> (sizeof(H) * 8 - 1 - idx_bit)) & 0x1);
        v = 0x0 | next_bit;
      }
      l = 1;
    }
  }

  MGARDm_EXEC void
  Operation3() { }

  MGARDm_EXEC void
  Operation4() { }

  MGARDm_EXEC void
  Operation5() { }

  MGARDm_CONT size_t
  shared_memory_size() { return singleton_size; }

  private:
  SubArray<1, H, DeviceType> densely;
  SubArray<1, size_t, DeviceType> dH_meta;
  SubArray<1, Q, DeviceType> bcode;
  SIZE len;
  int chunk_size;
  int n_chunk; 
  SubArray<1, uint8_t, DeviceType> singleton; 
  size_t singleton_size;

  uint8_t * _s_singleton;
};


template <typename Q, typename H, typename DeviceType>
class Decode: public AutoTuner<DeviceType> {
public:
  MGARDm_CONT
  Decode():AutoTuner<DeviceType>() {}

  MGARDm_CONT
  Task<DecodeFunctor<Q, H, DeviceType> > 
  GenTask(SubArray<1, H, DeviceType> densely, 
          SubArray<1, size_t, DeviceType> dH_meta,
          SubArray<1, Q, DeviceType> bcode,
          SIZE len, int chunk_size, int n_chunk, 
          SubArray<1, uint8_t, DeviceType> singleton, 
          size_t singleton_size, int queue_idx) {
    using FunctorType = DecodeFunctor<Q, H, DeviceType>;
    FunctorType functor(densely, dH_meta, bcode, len, chunk_size,
                        n_chunk, singleton, singleton_size);

    int nchunk = (len - 1) / chunk_size + 1;
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerSM();
    gridz = 1;
    gridy = 1;
    gridx = (nchunk - 1) / tbx + 1;
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "Decode"); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, H, DeviceType> densely, 
              SubArray<1, size_t, DeviceType> dH_meta,
              SubArray<1, Q, DeviceType> bcode,
              SIZE len, int chunk_size, int n_chunk, 
              SubArray<1, uint8_t, DeviceType> singleton, 
              size_t singleton_size, int queue_idx) {
    using FunctorType = DecodeFunctor<Q, H, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(densely, dH_meta, bcode, len, chunk_size,
                        n_chunk, singleton, singleton_size, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endif