#ifndef MGARD_X_LZ4_TEMPLATE_HPP
#define MGARD_X_LZ4_TEMPLATE_HPP

#ifdef MGARDX_COMPILE_CUDA

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#endif

namespace mgard_x {

template <typename DeviceType> class LZ4 {

public:
  LZ4() {}
  LZ4(SIZE n) { input_data = Array<1, Byte, DeviceType>({n}); }

  static size_t EstimateMemoryFootprint(SIZE n) { return n; }

  void Compress(Array<1, Byte, DeviceType> &data, size_t chunk_size,
                int queue_idx) {
#ifdef MGARDX_COMPILE_CUDA
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    // Make a copy of the input data
    input_data.resize({data.shape(0)});
    MemoryManager<DeviceType>::Copy1D(input_data.data(), data.data(),
                                      data.shape(0), queue_idx);
    // Array<1, Byte, DeviceType> input_data = data;
    Array<1, Byte, DeviceType> &output_data = data;
    nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
    nvcomp::LZ4Manager nvcomp_manager{
        chunk_size, dtype, DeviceRuntime<DeviceType>::GetQueue(queue_idx)};
    size_t input_count = input_data.shape(0);
    nvcomp::CompressionConfig comp_config =
        nvcomp_manager.configure_compression(input_count);
    output_data.resize({(SIZE)comp_config.max_compressed_buffer_size});
    nvcomp_manager.compress((uint8_t *)input_data.data(), output_data.data(),
                            comp_config);
    output_data.shape(0) =
        nvcomp_manager.get_compressed_output_size(output_data.data());
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    log::info("LZ4 block size: " + std::to_string(chunk_size));

    log::info("LZ4 compress ratio: " + std::to_string(input_count) + "/" +
              std::to_string(output_data.shape(0)) + " (" +
              std::to_string((double)(input_count) / output_data.shape(0)) +
              ")");
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

  void Decompress(Array<1, Byte, DeviceType> &data, int queue_idx) {
// DeviceRuntime<DeviceType>::SyncDevice();
#ifdef MGARDX_COMPILE_CUDA
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    // Make a copy of the input data
    input_data.resize({data.shape(0)});
    MemoryManager<DeviceType>::Copy1D(input_data.data(), data.data(),
                                      data.shape(0), queue_idx);
    // Array<1, Byte, DeviceType> input_data = data;
    Array<1, Byte, DeviceType> &output_data = data;
    auto decomp_nvcomp_manager = nvcomp::create_manager(
        input_data.data(), DeviceRuntime<DeviceType>::GetQueue(queue_idx));
    size_t input_size = input_data.shape(0);
    nvcomp::DecompressionConfig decomp_config =
        decomp_nvcomp_manager->configure_decompression(input_data.data());
    output_data.resize({(SIZE)decomp_config.decomp_data_size});
    decomp_nvcomp_manager->decompress(output_data.data(), input_data.data(),
                                      decomp_config);
    output_data.shape(0) = decomp_config.decomp_data_size;
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
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
    // DeviceRuntime<DeviceType>::SyncDevice();
  }

  // Workspace
  Array<1, Byte, DeviceType> input_data;
};

} // namespace mgard_x

#endif
