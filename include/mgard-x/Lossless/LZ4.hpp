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
  Timer timer;
  if (log::level & log::TIME) timer.start();
  nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
  nvcomp::LZ4Manager nvcomp_manager{chunk_size, dtype,
                                    DeviceRuntime<DeviceType>::GetQueue(0)};
  size_t input_count = input_data.shape(0);
  nvcomp::CompressionConfig comp_config =
      nvcomp_manager.configure_compression(input_count * sizeof(C));
  Array<1, Byte, DeviceType> output_data(
      {(SIZE)comp_config.max_compressed_buffer_size});
  nvcomp_manager.compress((uint8_t *)input_data.data(), output_data.data(),
                          comp_config);
  output_data.shape(0) =
      nvcomp_manager.get_compressed_output_size(output_data.data());
  DeviceRuntime<DeviceType>::SyncQueue(0);
  log::info("LZ4 block size: " + std::to_string(chunk_size));
  log::info("LZ4 compress ratio: " +
              std::to_string((double)(input_count*sizeof(C)) / output_data.shape(0)));
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer.end();
    timer.print("LZ4 compress");
    timer.clear();
  }
  return output_data;
}

template <typename C, typename DeviceType>
Array<1, C, DeviceType>
LZ4Decompress(SubArray<1, Byte, DeviceType> &input_data) {
  Timer timer;
  if (log::level & log::TIME) timer.start();
  auto decomp_nvcomp_manager = nvcomp::create_manager(
      input_data.data(), DeviceRuntime<DeviceType>::GetQueue(0));
  size_t input_size = input_data.shape(0);
  nvcomp::DecompressionConfig decomp_config =
      decomp_nvcomp_manager->configure_decompression(input_data.data());
  Array<1, C, DeviceType> output_data({(SIZE)decomp_config.decomp_data_size});
  decomp_nvcomp_manager->decompress(output_data.data(), input_data.data(),
                                    decomp_config);
  output_data.shape(0) = decomp_config.decomp_data_size / sizeof(C);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer.end();
    timer.print("LZ4 decompress");
    timer.clear();
  }
  return output_data;
}

} // namespace mgard_x

#endif
#endif
