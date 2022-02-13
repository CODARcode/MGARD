#ifndef MGARD_X_LZ4_TEMPLATE_HPP
#define MGARD_X_LZ4_TEMPLATE_HPP

#ifdef MGARDX_COMPILE_CUDA

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

namespace mgard_x {

template <typename C, typename DeviceType>
Array<1, Byte, DeviceType> LZ4Compress(SubArray<1, C, DeviceType> &input_data,
                                       size_t chunk_size) {
  // using Mem = MemoryManager<DeviceType>;
  nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
  // nvcomp::LZ4Compressor compressor(chunk_size, dtype);
  nvcomp::LZ4Manager nvcomp_manager{chunk_size, dtype,
                                    DeviceRuntime<DeviceType>::GetQueue(0)};

  // size_t *temp_bytes;
  // size_t *output_bytes;
  // Mem::MallocHost(temp_bytes, 1, 0);
  // Mem::MallocHost(output_bytes, 1, 0);
  // DeviceRuntime<DeviceType>::SyncQueue(0);

  size_t input_count = input_data.getShape(0);

  // compressor.configure(input_count * sizeof(C), temp_bytes, output_bytes);
  nvcomp::CompressionConfig comp_config =
      nvcomp_manager.configure_compression(input_count * sizeof(C));

  // Array<1, Byte, DeviceType> temp_space({(SIZE)*temp_bytes});
  Array<1, Byte, DeviceType> output_data(
      {(SIZE)comp_config.max_compressed_buffer_size});

  // compressor.compress_async(input_data.data(), input_count * sizeof(C),
  //                           temp_space.get_dv(), *temp_bytes,
  //                           output_data.get_dv(), output_bytes,
  //                           DeviceRuntime<DeviceType>::GetQueue(0));
  nvcomp_manager.compress((uint8_t *)input_data.data(), output_data.get_dv(),
                          comp_config);
  output_data.getShape()[0] =
      nvcomp_manager.get_compressed_output_size(output_data.get_dv());

  DeviceRuntime<DeviceType>::SyncQueue(0);
  // output_data.getShape()[0] = *output_bytes;
  // Mem::FreeHost(temp_bytes);
  // Mem::FreeHost(output_bytes);
  return output_data;
}

template <typename C, typename DeviceType>
Array<1, C, DeviceType>
LZ4Decompress(SubArray<1, Byte, DeviceType> &input_data) {
  // using Mem = MemoryManager<DeviceType>;
  // nvcomp::LZ4Decompressor decompressor;
  auto decomp_nvcomp_manager = nvcomp::create_manager(
      input_data.data(), DeviceRuntime<DeviceType>::GetQueue(0));
  // size_t *temp_bytes;
  // size_t *output_bytes;
  // Mem::MallocHost(temp_bytes, 1, 0);
  // Mem::MallocHost(output_bytes, 1, 0);
  // DeviceRuntime<DeviceType>::SyncQueue(0);

  size_t input_size = input_data.getShape(0);

  // decompressor.configure(input_data.data(), input_size, temp_bytes,
  //                        output_bytes,
  //                        DeviceRuntime<DeviceType>::GetQueue(0));
  nvcomp::DecompressionConfig decomp_config =
      decomp_nvcomp_manager->configure_decompression(input_data.data());

  // Array<1, Byte, DeviceType> temp_space({(SIZE)*temp_bytes});
  Array<1, C, DeviceType> output_data({(SIZE)decomp_config.decomp_data_size});

  // decompressor.decompress_async(input_data.data(), input_size,
  //                               temp_space.get_dv(), *temp_bytes,
  //                               output_data.get_dv(), *output_bytes,
  //                               DeviceRuntime<DeviceType>::GetQueue(0));

  decomp_nvcomp_manager->decompress(output_data.get_dv(), input_data.data(),
                                    decomp_config);
  output_data.getShape()[0] = decomp_config.decomp_data_size / sizeof(C);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  // output_data.getShape()[0] = (*output_bytes) / sizeof(C);
  // Mem::FreeHost(temp_bytes);
  // Mem::FreeHost(output_bytes);
  return output_data;
}

} // namespace mgard_x

#endif
#endif
