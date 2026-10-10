/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_LOSSY_COMPRESSOR_INTERFACE_HPP
#define MGARD_X_160_LOSSY_COMPRESSOR_INTERFACE_HPP
namespace mgard_x_160 {
template <DIM D, typename T, typename DeviceType>
class LossyCompressorInterface {
  virtual void Compress(Array<D, T, DeviceType> &original_data,
                        enum error_bound_type type, T tol, T s, T &norm,
                        Array<1, Byte, DeviceType> &compressed_data,
                        int queue_idx) = 0;
  virtual void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                          enum error_bound_type type, T tol, T s, T &norm,
                          Array<D, T, DeviceType> &decompressed_data,
                          int queue_idx) = 0;
};
} // namespace mgard_x_160

#endif