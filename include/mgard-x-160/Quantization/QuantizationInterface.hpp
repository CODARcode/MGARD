/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_QUANTIZATION_INTERFACE_HPP
#define MGARD_X_160_QUANTIZATION_INTERFACE_HPP
namespace mgard_x_160 {
template <DIM D, typename T, typename Q, typename DeviceType>
class QuantizationInterface {
  virtual void Quantize(SubArray<D, T, DeviceType> original_data,
                        enum error_bound_type ebtype, T tol, T s, T norm,
                        SubArray<D, Q, DeviceType> quantized_data,
                        int queue_idx) = 0;

  virtual void Dequantize(SubArray<D, T, DeviceType> original_data,
                          enum error_bound_type ebtype, T tol, T s, T norm,
                          SubArray<D, Q, DeviceType> quantized_data,
                          int queue_idx) = 0;
};
} // namespace mgard_x_160

#endif