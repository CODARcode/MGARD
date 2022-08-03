/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "nvcomp.hpp"
#include "nvcomp/bitcomp.hpp"

#ifndef MGARD_X_BITCOMP_TEMPLATE_HPP
#define MGARD_X_BITCOMP_TEMPLATE_HPP

namespace mgard_x {

template <typename C, typename DeviceType>
Array<1, Byte, DeviceType>
BitcompCompress(SubArray<1, C, DeviceType> &input_data, int algorithm_type) {
  using Mem = MemoryManager<DeviceType>;
  nvcomp::BitcompCompressor compressor(nvcomp::TypeOf<C>(), algorithm_type);

  size_t *temp_bytes;
  size_t *output_bytes;
  Mem::MallocHost(temp_bytes, 1, 0);
  Mem::MallocHost(output_bytes, 1, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  size_t input_count = input_data.shape(0);

  compressor.configure(input_count * sizeof(C), temp_bytes, output_bytes);

  Array<1, Byte, DeviceType> temp_space({(SIZE)*temp_bytes});
  Array<1, Byte, DeviceType> output_data({(SIZE)*output_bytes});

  compressor.compress_async(input_data.data(), input_count * sizeof(C),
                            temp_space.data(), *temp_bytes, output_data.data(),
                            output_bytes,
                            DeviceRuntime<DeviceType>::GetQueue(0));
  DeviceRuntime<DeviceType>::SyncQueue(0);
  output_data.shape(0) = *output_bytes;
  Mem::FreeHost(temp_bytes);
  Mem::FreeHost(output_bytes);
  return output_data;
}

template <typename C, typename DeviceType>
Array<1, C, DeviceType>
BitcompDecompress(SubArray<1, Byte, DeviceType> &input_data) {
  using Mem = MemoryManager<DeviceType>;
  nvcomp::BitcompDecompressor decompressor;

  size_t *temp_bytes;
  size_t *output_bytes;
  Mem::MallocHost(temp_bytes, 1, 0);
  Mem::MallocHost(output_bytes, 1, 0);

  decompressor.configure(input_data.data(), input_size, temp_bytes,
                         output_bytes, DeviceRuntime<DeviceType>::GetQueue(0));

  Array<1, Byte, DeviceType> temp_space({(SIZE)*temp_bytes});
  Array<1, C, DeviceType> output_data({(SIZE)*output_bytes});

  decompressor.decompress_async(input_data.data(), input_size,
                                temp_space.data(), *temp_bytes,
                                output_data.data(), *output_bytes,
                                DeviceRuntime<DeviceType>::GetQueue(0));
  DeviceRuntime<DeviceType>::SyncQueue(0);
  output_data.shape(0) = (*output_bytes) / sizeof(C);
  Mem::FreeHost(temp_bytes);
  Mem::FreeHost(output_bytes);
  return output_data;
}
} // namespace mgard_x

#endif