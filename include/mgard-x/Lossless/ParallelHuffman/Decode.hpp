/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_DECODE_TEMPLATE_HPP
#define MGARD_X_DECODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

template <typename Q, typename H, typename DeviceType>
class DecodeFunctor: public Functor<DeviceType> {
  public:
  MGARDX_CONT DecodeFunctor(){}
  MGARDX_CONT DecodeFunctor(SubArray<1, H, DeviceType> densely, 
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

  MGARDX_EXEC void
  Operation1() {
    _s_singleton = (uint8_t*)FunctorBase<DeviceType>::GetSharedMemory();
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      memcpy(_s_singleton, singleton((IDX)0), singleton_size);
    }
  }

  MGARDX_EXEC void
  Operation2() { 
    size_t chunk_id = FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX() + FunctorBase<DeviceType>::GetThreadIdX();
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
    auto first = reinterpret_cast<H *>(singleton((IDX)0));
    auto entry = first + sizeof(H) * 8;
    auto keys =
        reinterpret_cast<Q *>(singleton((IDX)0) + sizeof(H) * (2 * sizeof(H) * 8));
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

      // debug - start
      // if (!chunk_id) {
      // // if ((entry[l] + v - first[l])*sizeof(Q) + sizeof(H) * (2 * sizeof(H) * 8) >= 1280) {
      //   printf("out of range: %llu\n", (entry[l] + v - first[l])*sizeof(Q) + sizeof(H) * (2 * sizeof(H) * 8));
      //   printf("l: %llu\n", l);
      //   printf("entry[l]: %llu\n", entry[l]);
      //   printf("v: %llu\n", v);
      //   printf("first[l]: %llu\n", first[l]);
      //   printf("entry:");
      //   for (int i = 0; i < 64; i++) {
      //     printf("%llu ", entry[i]);
      //   }
      //   printf("\n");
      //   printf("first:");
      //   for (int i = 0; i < 64; i++) {
      //     printf("%llu ", first[i]);
      //   }
      //   printf("\n");
      // } 
      //debug - end
      // if (entry[l] + v - first[l] > 100000) {
      //   printf("offset: %llu + %llu i: %llu l: %llu (%llu, %llu, %llu)\n", sizeof(H) * (2 * sizeof(H) * 8), entry[l] + v - first[l], i, l, entry[l], v, first[l]);
      // }
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

  MGARDX_EXEC void
  Operation3() { }

  MGARDX_EXEC void
  Operation4() { }

  MGARDX_EXEC void
  Operation5() { }

  MGARDX_CONT size_t
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
  MGARDX_CONT
  Decode():AutoTuner<DeviceType>() {}

  MGARDX_CONT
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
    tbx = tBLK_DEFLATE;
    gridz = 1;
    gridy = 1;
    gridx = (nchunk - 1) / tbx + 1;
    // printf("sm_size: %llu\n", sm_size);
    // SubArray<1, H, DeviceType> temp({(SIZE)(singleton_size/sizeof(H))}, (H*)singleton.data());
    // PrintSubarray("singleton", temp);
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "Decode"); 
  }

  MGARDX_CONT
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