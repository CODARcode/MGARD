/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_LOSSLESS_COMPRESSOR_INTERFACE_HPP
#define MGARD_X_LOSSLESS_COMPRESSOR_INTERFACE_HPP
namespace mgard_x {
template <typename T, typename DeviceType> class LosslessCompressorInterface {
  virtual void Compress(Array<1, T, DeviceType> &original_data,
                        Array<1, Byte, DeviceType> &compressed_data,
                        int queue_idx) = 0;
  virtual void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                          Array<1, T, DeviceType> &decompressed_data,
                          int queue_idx) = 0;
};
} // namespace mgard_x

#endif