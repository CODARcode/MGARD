/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_DEFLATE_TEMPLATE_HPP
#define MGARD_X_DEFLATE_TEMPLATE_HPP
#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

// Jieyang: this kernel rely on whole grid sychronized execution
// For example, adding
//   if (thread == 0) {
//      __nanosleep(1e9);
//   }
// will cause incorrect results

template <typename H, typename DeviceType>
class DeflateFunctor: public Functor<DeviceType> {
  public:
  MGARDX_CONT DeflateFunctor(){}
  MGARDX_CONT DeflateFunctor(SubArray<1, H, DeviceType> hcoded, 
                             SIZE len,
                             SubArray<1, size_t, DeviceType> densely_meta,
                             int PART_SIZE):
                             hcoded(hcoded), len(len),
                             densely_meta(densely_meta), PART_SIZE(PART_SIZE) {
    Functor<DeviceType>();                            
  }

  MGARDX_EXEC void
  Operation1() {
    size_t gid = FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX() + FunctorBase<DeviceType>::GetThreadIdX();
    if (gid >= (len - 1) / PART_SIZE + 1)
      return;
    uint8_t bitwidth;
    size_t densely_coded_lsb_pos = sizeof(H) * 8, total_bitwidth = 0;
    size_t ending =
        (gid + 1) * PART_SIZE <= len ? PART_SIZE : len - gid * PART_SIZE;
    //    if ((gid + 1) * PART_SIZE > len) printf("\n\ngid %lu\tending %lu\n\n",
    //    gid, ending);
    H msb_bw_word_lsb, _1, _2;
    H *current = hcoded(gid * PART_SIZE);
    for (size_t i = 0; i < ending; i++) {
      msb_bw_word_lsb = *hcoded(gid * PART_SIZE + i);
      bitwidth = *((uint8_t *)&msb_bw_word_lsb + (sizeof(H) - 1));

      *((uint8_t *)&msb_bw_word_lsb + sizeof(H) - 1) = 0x0;
      if (densely_coded_lsb_pos == sizeof(H) * 8)
        *current = 0x0; // a new unit of data type
      if (bitwidth <= densely_coded_lsb_pos) {
        densely_coded_lsb_pos -= bitwidth;
        *current |= msb_bw_word_lsb << densely_coded_lsb_pos;
        if (densely_coded_lsb_pos == 0) {
          densely_coded_lsb_pos = sizeof(H) * 8;
          ++current;
        }
      } else {
        // example: we have 5-bit code 11111 but 3 bits left for (*current)
        // we put first 3 bits of 11111 to the last 3 bits of (*current)
        // and put last 2 bits from MSB of (*(++current))
        // the comment continues with the example
        _1 = msb_bw_word_lsb >> (bitwidth - densely_coded_lsb_pos);
        _2 = msb_bw_word_lsb << (sizeof(H) * 8 -
                                 (bitwidth - densely_coded_lsb_pos));
        *current |= _1;
        *(++current) = 0x0;
        *current |= _2;
        densely_coded_lsb_pos =
            sizeof(H) * 8 - (bitwidth - densely_coded_lsb_pos);
      }
      total_bitwidth += bitwidth;
    }
    *densely_meta(gid) = total_bitwidth;
  }

  MGARDX_EXEC void
  Operation2() { }

  MGARDX_EXEC void
  Operation3() { }

  MGARDX_EXEC void
  Operation4() { }

  MGARDX_EXEC void
  Operation5() { }

  MGARDX_CONT size_t
  shared_memory_size() { return 0; }

  private:
  SubArray<1, H, DeviceType> hcoded;
  SIZE len;
  SubArray<1, size_t, DeviceType> densely_meta;
  int PART_SIZE;
};


template <typename H, typename DeviceType>
class Deflate: public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Deflate():AutoTuner<DeviceType>() {}

  MGARDX_CONT
  Task<DeflateFunctor<H, DeviceType> > 
  GenTask(SubArray<1, H, DeviceType> hcoded, 
                   SIZE len,
                   SubArray<1, size_t, DeviceType> densely_meta,
                   int PART_SIZE, int queue_idx) {
    using FunctorType = DeflateFunctor<H, DeviceType>;
    FunctorType functor(hcoded, len, densely_meta, PART_SIZE);

    auto nchunk = (len - 1) / PART_SIZE + 1;
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = tBLK_DEFLATE;
    gridz = 1;
    gridy = 1;
    gridx = (nchunk - 1) / tbx + 1;
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "Deflate"); 
  }

  MGARDX_CONT
  void Execute(SubArray<1, H, DeviceType> hcoded, 
               SIZE len,
               SubArray<1, size_t, DeviceType> densely_meta,
               int PART_SIZE, int queue_idx) {
    using FunctorType = DeflateFunctor<H, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(hcoded, len, densely_meta, PART_SIZE, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endif