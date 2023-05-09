/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_QUANTIZATION_INTERFACE_HPP
#define MGARD_X_QUANTIZATION_INTERFACE_HPP
namespace mgard_x {
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
} // namespace mgard_x

#endif