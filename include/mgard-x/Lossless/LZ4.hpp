#ifndef MGARD_X_LZ4_TEMPLATE_HPP
#define MGARD_X_LZ4_TEMPLATE_HPP

#ifdef MGARDX_COMPILE_CUDA

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#endif

namespace mgard_x {

template <typename DeviceType>
void LZ4Compress(Array<1, Byte, DeviceType> &data, size_t chunk_size) {
#ifdef MGARDX_COMPILE_CUDA
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  // Make a copy of the input data
  Array<1, Byte, DeviceType> input_data = data;
  Array<1, Byte, DeviceType> &output_data = data;
  nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
  nvcomp::LZ4Manager nvcomp_manager{chunk_size, dtype,
                                    DeviceRuntime<DeviceType>::GetQueue(0)};
  size_t input_count = input_data.shape(0);
  nvcomp::CompressionConfig comp_config =
      nvcomp_manager.configure_compression(input_count);
  output_data.resize({(SIZE)comp_config.max_compressed_buffer_size});
  nvcomp_manager.compress((uint8_t *)input_data.data(), output_data.data(),
                          comp_config);
  output_data.shape(0) =
      nvcomp_manager.get_compressed_output_size(output_data.data());
  DeviceRuntime<DeviceType>::SyncQueue(0);
  log::info("LZ4 block size: " + std::to_string(chunk_size));
  log::info("LZ4 compress ratio: " +
            std::to_string((double)(input_count) / output_data.shape(0)));
  if (log::level & log::TIME) {
    timer.end();
    timer.print("LZ4 compress");
    timer.clear();
  }
#else
  log::err("LZ4 for is only available on CUDA devices. Portable version is "
           "in development.");
  exit(-1);
#endif
}

template <typename DeviceType>
void LZ4Decompress(Array<1, Byte, DeviceType> &data) {
#ifdef MGARDX_COMPILE_CUDA
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  // Make a copy of the input data
  Array<1, Byte, DeviceType> input_data = data;
  Array<1, Byte, DeviceType> &output_data = data;
  auto decomp_nvcomp_manager = nvcomp::create_manager(
      input_data.data(), DeviceRuntime<DeviceType>::GetQueue(0));
  size_t input_size = input_data.shape(0);
  nvcomp::DecompressionConfig decomp_config =
      decomp_nvcomp_manager->configure_decompression(input_data.data());
  output_data.resize({(SIZE)decomp_config.decomp_data_size});
  decomp_nvcomp_manager->decompress(output_data.data(), input_data.data(),
                                    decomp_config);
  output_data.shape(0) = decomp_config.decomp_data_size;
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    timer.end();
    timer.print("LZ4 decompress");
    timer.clear();
  }
#else
  log::err("LZ4 for is only available on CUDA devices. Portable version is "
           "in development.");
  exit(-1);
#endif
}

} // namespace mgard_x

#endif
