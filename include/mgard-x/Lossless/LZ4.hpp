#ifndef MGARD_X_LZ4_TEMPLATE_HPP
#define MGARD_X_LZ4_TEMPLATE_HPP

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"

namespace mgard_x {

template <typename C, typename DeviceType>
Array<1, Byte, DeviceType> 
LZ4Compress(SubArray<1, C, DeviceType> &input_data, size_t chunk_size) {
  using Mem = MemoryManager<DeviceType>;
  nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
  nvcomp::LZ4Compressor compressor(chunk_size, dtype);

  size_t *temp_bytes;
  size_t *output_bytes;
  Mem::MallocHost(temp_bytes, 1, 0);
  Mem::MallocHost(output_bytes, 1, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  size_t input_count = input_data.getShape(0);

  compressor.configure(input_count * sizeof(C), temp_bytes, output_bytes);

  Array<1, Byte, DeviceType> temp_space({(SIZE)*temp_bytes});
  Array<1, Byte, DeviceType> output_data({(SIZE)*output_bytes});

  compressor.compress_async(input_data.data(), input_count * sizeof(C), temp_space.get_dv(),
                            *temp_bytes, output_data.get_dv(), output_bytes,
                            DeviceRuntime<DeviceType>::GetQueue(0));

  DeviceRuntime<DeviceType>::SyncQueue(0);
  output_data.getShape()[0] = *output_bytes;  
  Mem::FreeHost(temp_bytes);
  Mem::FreeHost(output_bytes);
  return output_data;
}

template <typename C, typename DeviceType>
Array<1, C, DeviceType>
LZ4Decompress(SubArray<1, Byte, DeviceType> &input_data) {
  using Mem = MemoryManager<DeviceType>;
  nvcomp::LZ4Decompressor decompressor;
  size_t *temp_bytes;
  size_t *output_bytes;
  Mem::MallocHost(temp_bytes, 1, 0);
  Mem::MallocHost(output_bytes, 1, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  size_t input_size = input_data.getShape(0);

  decompressor.configure(input_data.data(), input_size, temp_bytes, output_bytes,
                         DeviceRuntime<DeviceType>::GetQueue(0));

  Array<1, Byte, DeviceType> temp_space({(SIZE)*temp_bytes});
  Array<1, C, DeviceType> output_data({(SIZE)*output_bytes});

  decompressor.decompress_async(input_data.data(), input_size, temp_space.get_dv(), *temp_bytes,
                                output_data.get_dv(), *output_bytes,
                                DeviceRuntime<DeviceType>::GetQueue(0));

  DeviceRuntime<DeviceType>::SyncQueue(0);
  output_data.getShape()[0] = (*output_bytes)/sizeof(C);
  Mem::FreeHost(temp_bytes);
  Mem::FreeHost(output_bytes);
  return output_data;
}

}


#endif

