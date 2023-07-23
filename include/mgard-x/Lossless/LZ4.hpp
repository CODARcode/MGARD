#ifndef MGARD_X_LZ4_TEMPLATE_HPP
#define MGARD_X_LZ4_TEMPLATE_HPP

#ifdef MGARDX_COMPILE_CUDA

// #include "ParallelHuffman/Condense.hpp"
#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#endif

namespace mgard_x {

template <typename DeviceType> class LZ4 {

public:
  LZ4() {}
  LZ4(SIZE n, SIZE chunk_size) {
    Resize(n, chunk_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Resize(SIZE n, SIZE chunk_size, int queue_idx) {
#ifdef MGARDX_COMPILE_CUDA
    this->chunk_size = chunk_size;
    input_data.resize({n}, queue_idx);
    nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
    nvcomp::LZ4Manager nvcomp_manager{
        chunk_size, dtype, DeviceRuntime<DeviceType>::GetQueue(queue_idx)};
    size_t temp_size = nvcomp_manager.get_required_scratch_buffer_size();
    temp_data.resize({temp_size}, queue_idx);
    // size_t batch_size = (n + chunk_size - 1) / chunk_size;
    // size_t comp_temp_bytes;
    // nvcompBatchedLZ4CompressGetTempSize(
    //     batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts,
    //     &comp_temp_bytes);
    // size_t decomp_temp_bytes;
    // nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size,
    //                                       &decomp_temp_bytes);
    // temp_data.resize({std::max(comp_temp_bytes, decomp_temp_bytes)},
    // queue_idx); nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
    //     chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

    // host_uncompressed_bytes.resize(batch_size);
    // uncompressed_bytes.resize({batch_size}, queue_idx);
    // host_uncompressed_ptrs.resize(batch_size);
    // uncompressed_data_ptrs.resize({batch_size}, queue_idx);
    // compressed_bytes.resize({batch_size}, queue_idx);
    // compressed_chunck_data.resize({max_out_bytes * batch_size}, queue_idx);
    // host_compressed_ptrs.resize(batch_size);
    // compressed_data_ptrs.resize({batch_size}, queue_idx);
    // host_compressed_bytes.resize(batch_size);
    // host_compressed_write_offset.resize(batch_size);
    // compressed_write_offset.resize({batch_size}, queue_idx);
#endif
  }

  static size_t EstimateMemoryFootprint(SIZE n, SIZE chunk_size) {
    size_t size = 0;
#ifdef MGARDX_COMPILE_CUDA
    size += n;
    nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
    nvcomp::LZ4Manager nvcomp_manager{chunk_size, dtype,
                                      DeviceRuntime<DeviceType>::GetQueue(0)};
    size += nvcomp_manager.get_required_scratch_buffer_size();
    // size_t batch_size = (n + chunk_size - 1) / chunk_size;
    // size_t comp_temp_bytes;
    // nvcompBatchedLZ4CompressGetTempSize(
    //     batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts,
    //     &comp_temp_bytes);
    // size_t decomp_temp_bytes;
    // nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size,
    //                                       &decomp_temp_bytes);
    // size += std::max(comp_temp_bytes, decomp_temp_bytes);
    // size_t max_out_bytes;
    // nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
    //     chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);
    // size += batch_size * 5 * sizeof(size_t);
    // size += max_out_bytes * batch_size;
#endif
    return size;
  }

  /*
     void Compress2(Array<1, Byte, DeviceType> &data, int queue_idx) {
  #ifdef MGARDX_COMPILE_CUDA
      Timer timer;
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.start();
      }
      input_data.resize({data.shape(0)}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(input_data.data(), data.data(),
                                        data.shape(0), queue_idx);
      Array<1, Byte, DeviceType> &output_data = data;

      size_t uncompressed_total_bytes = input_data.shape(0);
      size_t batch_size =
          (uncompressed_total_bytes + chunk_size - 1) / chunk_size;

      // Input size
      // host_uncompressed_bytes.resize(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        if (i + 1 < batch_size) {
          host_uncompressed_bytes[i] = chunk_size;
        } else {
          // last chunk may be smaller
          host_uncompressed_bytes[i] =
              uncompressed_total_bytes - (chunk_size * i);
        }
      }
      // uncompressed_bytes.resize({batch_size}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(uncompressed_bytes.data(),
                                        host_uncompressed_bytes.data(),
                                        batch_size, queue_idx);

      // Input data
      // host_uncompressed_ptrs.resize(batch_size);
      SubArray input_data_subarray(input_data);
      for (size_t i = 0; i < batch_size; ++i) {
        host_uncompressed_ptrs[i] = input_data_subarray(chunk_size * i);
      }
      // uncompressed_data_ptrs.resize({batch_size}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(uncompressed_data_ptrs.data(),
                                        host_uncompressed_ptrs.data(),
  batch_size, queue_idx);

      // get the maxmimum output size for each chunk
      // size_t max_out_bytes;
      // nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size,
      // nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

      // Output size
      // compressed_bytes.resize({batch_size}, queue_idx);

      // Output data
      // compressed_chunck_data.resize({max_out_bytes * batch_size}, queue_idx);
      SubArray compressed_chunck_data_subarray(compressed_chunck_data);
      // host_compressed_ptrs.resize(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        host_compressed_ptrs[i] = compressed_chunck_data_subarray(i *
  chunk_size);
      }
      // compressed_data_ptrs.resize({batch_size}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(compressed_data_ptrs.data(),
                                        host_compressed_ptrs.data(), batch_size,
                                        queue_idx);

      // And finally, call the API to compress the data
      nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
          uncompressed_data_ptrs.data(), uncompressed_bytes.data(),
          chunk_size, // The maximum chunk size
          batch_size, temp_data.data(), temp_data.shape(0),
          compressed_data_ptrs.data(), compressed_bytes.data(),
          nvcompBatchedLZ4DefaultOpts,
          DeviceRuntime<DeviceType>::GetQueue(queue_idx));

      if (comp_res != nvcompSuccess) {
        std::cerr << "Failed compression!" << std::endl;
        assert(comp_res == nvcompSuccess);
      }

      // host_compressed_bytes.resize(batch_size);
      MemoryManager<DeviceType>::Copy1D(host_compressed_bytes.data(),
                                        compressed_bytes.data(), batch_size,
                                        queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

      // host_compressed_write_offset.resize(batch_size);
      // compressed_write_offset.resize({batch_size}, queue_idx);
      host_compressed_write_offset[0] = 0;
      for (int i = 1; i < batch_size; i++) {
        host_compressed_write_offset[i] =
            host_compressed_write_offset[i - 1] + host_compressed_bytes[i - 1];
      }
      MemoryManager<DeviceType>::Copy1D(compressed_write_offset.data(),
                                        host_compressed_write_offset.data(),
                                        batch_size, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

      size_t total_compressed_bytes =
          host_compressed_write_offset[batch_size - 1] +
          host_compressed_bytes[batch_size - 1];

      SIZE byte_offset = 0;
      SIZE compressed_size;
      advance_with_align<size_t>(byte_offset, 1);
      advance_with_align<size_t>(byte_offset, 1);
      advance_with_align<size_t>(byte_offset, batch_size);
      advance_with_align<Byte>(byte_offset, total_compressed_bytes);
      compressed_size = byte_offset;
      output_data.resize({(SIZE)(compressed_size)});
      SubArray output_data_subarray(output_data);

      byte_offset = 0;
      SerializeArray<size_t>(output_data_subarray, &uncompressed_total_bytes, 1,
                             byte_offset, queue_idx);
      SerializeArray<size_t>(output_data_subarray, &chunk_size, 1, byte_offset,
                             queue_idx);
      SerializeArray<size_t>(output_data_subarray, host_compressed_bytes.data(),
                             batch_size, byte_offset, queue_idx);
      SubArray<1, Byte, DeviceType> output_condensed_subarray(
          {compressed_size - byte_offset}, output_data_subarray(byte_offset));

      DeviceLauncher<DeviceType>::Execute(
          CondenseKernel<Byte, DeviceType>(
              compressed_chunck_data_subarray,
  SubArray(compressed_write_offset), SubArray(compressed_bytes),
  output_condensed_subarray, chunk_size), queue_idx);

      log::info("LZ4 block size: " + std::to_string(chunk_size));

      log::info(
          "LZ4 compress ratio: " + std::to_string(uncompressed_total_bytes) +
          "/" + std::to_string(compressed_size) + " (" +
          std::to_string((double)(uncompressed_total_bytes) / compressed_size) +
          ")");
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("LZ4 compress");
        timer.print_throughput("LZ4 compress", uncompressed_total_bytes);
        timer.clear();
      }
  #else
      log::err("LZ4 for is only available on CUDA devices. Portable version is "
               "in development.");
      exit(-1);
  #endif
    }

     void Decompress2(Array<1, Byte, DeviceType> &data, int queue_idx) {
  #ifdef MGARDX_COMPILE_CUDA
      Timer timer;
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.start();
      }
      input_data.resize({data.shape(0)}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(input_data.data(), data.data(),
                                        data.shape(0), queue_idx);
      Array<1, Byte, DeviceType> &output_data = data;

      SubArray compressed_subarray(input_data);
      SubArray uncompressed_subarray(output_data);
      size_t byte_offset = 0;
      size_t uncompressed_total_bytes;
      size_t chunk_size;
      size_t batch_size;

      size_t *uncompressed_total_bytes_ptr = &uncompressed_total_bytes;
      size_t *chunk_size_ptr = &chunk_size;
      DeserializeArray<size_t>(compressed_subarray,
  uncompressed_total_bytes_ptr, 1, byte_offset, false, queue_idx);
      DeserializeArray<size_t>(compressed_subarray, chunk_size_ptr, 1,
                               byte_offset, false, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      batch_size = (uncompressed_total_bytes + chunk_size - 1) / chunk_size;
      // host_compressed_bytes.resize(batch_size);
      size_t *host_compressed_bytes_ptr = host_compressed_bytes.data();
      DeserializeArray<size_t>(compressed_subarray, host_compressed_bytes_ptr,
                               batch_size, byte_offset, false, queue_idx);

      // cudaStreamSynchronize(DeviceRuntime<DeviceType>::GetQueue(queue_idx));

      // compressed_bytes.resize({batch_size}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(compressed_bytes.data(),
                                        host_compressed_bytes.data(),
  batch_size, queue_idx);

      // host_compressed_ptrs.resize(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        Byte *ptr;
        DeserializeArray<Byte>(compressed_subarray, ptr,
  host_compressed_bytes[i], byte_offset, true, queue_idx);
        host_compressed_ptrs[i] = ptr;
      }
      // compressed_data_ptrs.resize({batch_size}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(compressed_data_ptrs.data(),
                                        host_compressed_ptrs.data(), batch_size,
                                        queue_idx);

      // host_uncompressed_bytes.resize(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        if (i + 1 < batch_size) {
          host_uncompressed_bytes[i] = chunk_size;
        } else {
          // last chunk may be smaller
          host_uncompressed_bytes[i] =
              uncompressed_total_bytes - (chunk_size * i);
        }
      }
      // uncompressed_bytes.resize({batch_size}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(uncompressed_bytes.data(),
                                        host_uncompressed_bytes.data(),
                                        batch_size, queue_idx);

      // host_uncompressed_ptrs.resize(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        host_uncompressed_ptrs[i] = uncompressed_subarray(chunk_size * i);
      }
      // uncompressed_data_ptrs.resize({batch_size}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(uncompressed_data_ptrs.data(),
                                        host_uncompressed_ptrs.data(),
  batch_size, queue_idx);

      device_statuses.resize({batch_size});

      nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
          compressed_data_ptrs.data(), compressed_bytes.data(),
          uncompressed_bytes.data(), uncompressed_bytes.data(), batch_size,
          temp_data.data(), temp_data.shape(0), uncompressed_data_ptrs.data(),
          device_statuses.data(),
  DeviceRuntime<DeviceType>::GetQueue(queue_idx));

      if (decomp_res != nvcompSuccess) {
        std::cerr << "Failed compression!" << std::endl;
        assert(decomp_res == nvcompSuccess);
      }

      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("LZ4 decompress");
        timer.print_throughput("LZ4 decompress", uncompressed_total_bytes);
        timer.clear();
      }
  #else
      log::err("LZ4 for is only available on CUDA devices. Portable version is "
               "in development.");
      exit(-1);
  #endif
    }
  */
  void Compress(Array<1, Byte, DeviceType> &data, int queue_idx) {
#ifdef MGARDX_COMPILE_CUDA
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    // Make a copy of the input data
    input_data.resize({data.shape(0)}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(input_data.data(), data.data(),
                                      data.shape(0), queue_idx);
    Array<1, Byte, DeviceType> &output_data = data;
    nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
    nvcomp::LZ4Manager nvcomp_manager{
        chunk_size, dtype, DeviceRuntime<DeviceType>::GetQueue(queue_idx)};
    nvcomp_manager.set_scratch_buffer(temp_data.data());
    size_t input_count = input_data.shape(0);
    nvcomp::CompressionConfig comp_config =
        nvcomp_manager.configure_compression(input_count);
    output_data.resize({(SIZE)comp_config.max_compressed_buffer_size},
                       queue_idx);
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
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("LZ4 compress");
      timer.print_throughput("LZ4 compress", input_count);
      timer.clear();
    }
#else
    log::err("LZ4 for is only available on CUDA devices. Portable version is "
             "in development.");
    exit(-1);
#endif
  }

  void Decompress(Array<1, Byte, DeviceType> &data, int queue_idx) {
#ifdef MGARDX_COMPILE_CUDA
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    // Make a copy of the input data
    input_data.resize({data.shape(0)}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(input_data.data(), data.data(),
                                      data.shape(0), queue_idx);
    Array<1, Byte, DeviceType> &output_data = data;
    nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
    nvcomp::LZ4Manager nvcomp_manager{
        chunk_size, dtype, DeviceRuntime<DeviceType>::GetQueue(queue_idx)};
    nvcomp_manager.set_scratch_buffer(temp_data.data());
    size_t input_size = input_data.shape(0);
    nvcomp::DecompressionConfig decomp_config =
        nvcomp_manager.configure_decompression(input_data.data());
    output_data.resize({(SIZE)decomp_config.decomp_data_size}, queue_idx);
    nvcomp_manager.decompress(output_data.data(), input_data.data(),
                              decomp_config);
    output_data.shape(0) = decomp_config.decomp_data_size;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("LZ4 decompress");
      timer.print_throughput("LZ4 decompress", output_data.shape(0));
      timer.clear();
    }
#else
    log::err("LZ4 for is only available on CUDA devices. Portable version is "
             "in development.");
    exit(-1);
#endif
  }

#ifdef MGARDX_COMPILE_CUDA
  // Workspace
  size_t chunk_size;
  // size_t max_out_bytes;

  Array<1, Byte, DeviceType> input_data;
  Array<1, Byte, DeviceType> temp_data;

  // std::vector<size_t> host_uncompressed_bytes;
  // std::vector<void *> host_uncompressed_ptrs;
  // Array<1, size_t, DeviceType> uncompressed_bytes;
  // Array<1, void *, DeviceType> uncompressed_data_ptrs;

  // std::vector<void *> host_compressed_ptrs;
  // Array<1, void *, DeviceType> compressed_data_ptrs;
  // Array<1, Byte, DeviceType> compressed_chunck_data;
  // std::vector<size_t> host_compressed_bytes;
  // Array<1, size_t, DeviceType> compressed_bytes;

  // std::vector<size_t> host_compressed_write_offset;
  // Array<1, size_t, DeviceType> compressed_write_offset;

  // Array<1, nvcompStatus_t, DeviceType> device_statuses;
#endif
};

} // namespace mgard_x

#endif
