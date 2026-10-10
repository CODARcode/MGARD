/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_LOSSLESS_COMPRESSOR_INTERFACE_HPP
#define MGARD_X_160_LOSSLESS_COMPRESSOR_INTERFACE_HPP
namespace mgard_x_160 {
template <typename T, typename DeviceType> class LosslessCompressorInterface {
  virtual void Compress(Array<1, T, DeviceType> &original_data,
                        Array<1, Byte, DeviceType> &compressed_data,
                        int queue_idx) = 0;
  virtual void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                          Array<1, T, DeviceType> &decompressed_data,
                          int queue_idx) = 0;
};
} // namespace mgard_x_160

#endif