/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DECODE_TEMPLATE_HPP
#define MGARD_X_DECODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

template <typename Q, typename H, bool CACHE_SINGLETION, typename DeviceType>
class DecodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DecodeFunctor() {}
  MGARDX_CONT DecodeFunctor(SubArray<1, H, DeviceType> densely,
                            SubArray<1, size_t, DeviceType> dH_meta,
                            SubArray<1, Q, DeviceType> bcode, SIZE len,
                            int chunk_size, int n_chunk,
                            SubArray<1, uint8_t, DeviceType> singleton,
                            size_t singleton_size)
      : densely(densely), dH_meta(dH_meta), bcode(bcode), len(len),
        chunk_size(chunk_size), n_chunk(n_chunk), singleton(singleton),
        singleton_size(singleton_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    if (CACHE_SINGLETION) {
      _s_singleton = (uint8_t *)FunctorBase<DeviceType>::GetSharedMemory();
      if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
        memcpy(_s_singleton, singleton((IDX)0), singleton_size);
      }
    } else {
      _s_singleton = singleton((IDX)0);
    }
  }

  MGARDX_EXEC void Operation2() {
    size_t chunk_id = FunctorBase<DeviceType>::GetBlockIdX() *
                          FunctorBase<DeviceType>::GetBlockDimX() +
                      FunctorBase<DeviceType>::GetThreadIdX();
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
    H v = (*densely(densely_offset + idx_byte) >> (sizeof(H) * 8 - 1)) &
          0x1; // get the first bit
    size_t l = 1;
    size_t i = 0;
    while (i < total_bw) {
      while (v < first[l]) { // append next i_cb bit
        ++i;
        idx_byte = i / (sizeof(H) * 8);
        idx_bit = i % (sizeof(H) * 8);
        next_bit = ((*densely(densely_offset + idx_byte) >>
                     (sizeof(H) * 8 - 1 - idx_bit)) &
                    0x1);
        v = (v << 1) | next_bit;
        ++l;
      }

      // debug - start
      // if (!chunk_id) {
      // // if ((entry[l] + v - first[l])*sizeof(Q) + sizeof(H) * (2 * sizeof(H)
      // * 8) >= 1280) {
      //   printf("out of range: %llu\n", (entry[l] + v - first[l])*sizeof(Q) +
      //   sizeof(H) * (2 * sizeof(H) * 8)); printf("l: %llu\n", l);
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
      // debug - end
      // if (entry[l] + v - first[l] > 100000) {
      //   printf("offset: %llu + %llu i: %llu l: %llu (%llu, %llu, %llu)\n",
      //   sizeof(H) * (2 * sizeof(H) * 8), entry[l] + v - first[l], i, l,
      //   entry[l], v, first[l]);
      // }
      *bcode(bcode_offset + idx_bcoded) = keys[entry[l] + v - first[l]];
      idx_bcoded++;
      {
        ++i;
        idx_byte = i / (sizeof(H) * 8);
        idx_bit = i % (sizeof(H) * 8);
        next_bit = ((*densely(densely_offset + idx_byte) >>
                     (sizeof(H) * 8 - 1 - idx_bit)) &
                    0x1);
        v = 0x0 | next_bit;
      }
      l = 1;
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    if (CACHE_SINGLETION) {
      return singleton_size;
    } else {
      return 0;
    }
  }

private:
  SubArray<1, H, DeviceType> densely;
  SubArray<1, size_t, DeviceType> dH_meta;
  SubArray<1, Q, DeviceType> bcode;
  SIZE len;
  int chunk_size;
  int n_chunk;
  SubArray<1, uint8_t, DeviceType> singleton;
  size_t singleton_size;

  uint8_t *_s_singleton;
};

template <typename Q, typename H, bool CACHE_SINGLETION, typename DeviceType>
class DecodeKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  using DataType = H;
  constexpr static std::string_view Name = "decode";
  MGARDX_CONT
  DecodeKernel(SubArray<1, H, DeviceType> densely,
               SubArray<1, size_t, DeviceType> dH_meta,
               SubArray<1, Q, DeviceType> bcode, SIZE len, int chunk_size,
               int n_chunk, SubArray<1, uint8_t, DeviceType> singleton,
               size_t singleton_size)
      : densely(densely), dH_meta(dH_meta), bcode(bcode), len(len),
        chunk_size(chunk_size), n_chunk(n_chunk), singleton(singleton),
        singleton_size(singleton_size) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<DecodeFunctor<Q, H, CACHE_SINGLETION, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DecodeFunctor<Q, H, CACHE_SINGLETION, DeviceType>;
    FunctorType functor(densely, dH_meta, bcode, len, chunk_size, n_chunk,
                        singleton, singleton_size);

    int nchunk = (len - 1) / chunk_size + 1;
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = 1;
    gridy = 1;
    gridx = (nchunk - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, H, DeviceType> densely;
  SubArray<1, size_t, DeviceType> dH_meta;
  SubArray<1, Q, DeviceType> bcode;
  SIZE len;
  int chunk_size;
  int n_chunk;
  SubArray<1, uint8_t, DeviceType> singleton;
  size_t singleton_size;
};

template <typename Q, typename H, typename DeviceType>
void Decode(SubArray<1, H, DeviceType> densely,
            SubArray<1, size_t, DeviceType> dH_meta,
            SubArray<1, Q, DeviceType> bcode, SIZE len, int chunk_size,
            int n_chunk, SubArray<1, uint8_t, DeviceType> singleton,
            size_t singleton_size, int queue_idx) {
  int maxbytes = DeviceRuntime<DeviceType>::GetMaxSharedMemorySize();
  // Shared memory is disabled as it does not provide better performance
  // if (singleton_size <= maxbytes) {
  //   if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
  //     std::cout << log::log_info
  //               << "Decode: using share memory to cache decodebook\n";
  //   }
  //   DeviceLauncher<DeviceType>::Execute(
  //       DecodeKernel<Q, H, true, DeviceType>(densely, dH_meta, bcode, len,
  //                                            chunk_size, n_chunk, singleton,
  //                                            singleton_size),
  //       queue_idx);
  // } else {
  if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
    std::cout << log::log_info
              << "Decode: not using share memory to cache decodebook\n";
  }
  DeviceLauncher<DeviceType>::Execute(
      DecodeKernel<Q, H, false, DeviceType>(densely, dH_meta, bcode, len,
                                            chunk_size, n_chunk, singleton,
                                            singleton_size),
      queue_idx);
  // }
}

} // namespace mgard_x

#endif