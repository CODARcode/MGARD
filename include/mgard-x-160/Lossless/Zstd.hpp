#ifndef MGARD_X_160_ZSTD_TEMPLATE_HPP
#define MGARD_X_160_ZSTD_TEMPLATE_HPP

#include <cstdio>
#include <string>
#include <zstd.h>

#include "../RuntimeX/Utilities/Exceptions.h"

namespace mgard_x_160 {

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      char mgard_x_check_msg[256];                                             \
      std::snprintf(mgard_x_check_msg, sizeof(mgard_x_check_msg),              \
                    "CHECK(" #cond ") failed: " __VA_ARGS__);                  \
      throw mgard_x_160::ProcessingException(std::string(__FILE__) + ":" +         \
                                         std::to_string(__LINE__) + " " +      \
                                         mgard_x_check_msg);                   \
    }                                                                          \
  } while (0)

#define CHECK_ZSTD(fn, ...)                                                    \
  do {                                                                         \
    size_t const err = (fn);                                                   \
    CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err));                   \
  } while (0)

template <typename DeviceType> class Zstd {
public:
  Zstd() {}
  Zstd(SIZE n, int compressionLevel) : compressionLevel(compressionLevel) {
    Resize(n, compressionLevel, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Resize(SIZE buffer_size, int compressionLevel, int queue_idx) {
    if (this->buffer_size < buffer_size) {
      Release(queue_idx);
      this->compressionLevel = compressionLevel;
      size_t const estimated_out_size = ZSTD_compressBound(buffer_size);
      MemoryManager<DeviceType>::MallocHost(
          out_data, estimated_out_size + sizeof(size_t), queue_idx);
      MemoryManager<DeviceType>::MallocHost(in_data, buffer_size, queue_idx);
      this->buffer_size = buffer_size;
    }
  }

  void Release(int queue_idx) {
    if (out_data != nullptr) {
      MemoryManager<DeviceType>::FreeHost(out_data, queue_idx);
      out_data = nullptr;
    }
    if (in_data != nullptr) {
      MemoryManager<DeviceType>::FreeHost(in_data, queue_idx);
      in_data = nullptr;
    }
  }
  ~Zstd() {
    Release(0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  static size_t EstimateMemoryFootprint(SIZE n) {
    size_t size = 0;
    return size;
  }

  void Compress(Array<1, Byte, DeviceType> &data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    Array<1, Byte, DeviceType> &input_data = data;
    Array<1, Byte, DeviceType> &output_data = data;
    size_t input_count = input_data.shape(0);
    Resize(input_count, compressionLevel, queue_idx);
    size_t const estimated_out_size = ZSTD_compressBound(input_count);
    MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count,
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    size_t const actual_out_size =
        ZSTD_compress(out_data + sizeof(size_t), estimated_out_size, in_data,
                      input_count, compressionLevel);
    CHECK_ZSTD(actual_out_size);

    *(size_t *)out_data = (size_t)input_count;
    output_data.resize({(SIZE)(actual_out_size + sizeof(size_t))});
    MemoryManager<DeviceType>::Copy1D(output_data.data(), out_data,
                                      actual_out_size + sizeof(size_t),
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    log::dbg("Zstd compression level: " + std::to_string(compressionLevel));
    log::dbg("Zstd compress ratio: " +
             std::to_string((double)(input_count) /
                            (actual_out_size + sizeof(size_t))));
    if (log::level & log::TIME) {
      timer.end();
      timer.print("Zstd compress", input_count);
      timer.clear();
    }
  }

  void Decompress(Array<1, Byte, DeviceType> &data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME)
      timer.start();
    Array<1, Byte, DeviceType> &input_data = data;
    Array<1, Byte, DeviceType> &output_data = data;
    size_t input_count = input_data.shape(0);

    Resize(input_count, compressionLevel, queue_idx);
    MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count,
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // Written as a size_t by Compress; reading it into a narrower type would
    // truncate streams of 4 GiB or more.
    size_t actual_out_count = *reinterpret_cast<const size_t *>(in_data);

    // FIX: Only resize out_data if needed, do not touch in_data which contains
    // the compressed data we just copied. The original code called Resize()
    // which would free and reallocate BOTH in_data and out_data, destroying
    // the compressed data when actual_out_count > buffer_size.
    if (this->buffer_size < actual_out_count) {
      size_t const estimated_out_size = ZSTD_compressBound(actual_out_count);
      if (out_data != nullptr) {
        MemoryManager<DeviceType>::FreeHost(out_data, queue_idx);
      }
      MemoryManager<DeviceType>::MallocHost(
          out_data, estimated_out_size + sizeof(size_t), queue_idx);
    }
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // A corrupt or truncated stream must fail here: an unchecked failure
    // leaves out_data as garbage that the next stage misreads (MGARD 1.6.0
    // crashed in the Huffman decoder this way).
    size_t const decompressed_size =
        ZSTD_decompress(out_data, actual_out_count, in_data + sizeof(size_t),
                        input_count - sizeof(size_t));
    CHECK_ZSTD(decompressed_size);
    CHECK(decompressed_size == actual_out_count,
          "Zstd decompressed %zu bytes, header says %zu", decompressed_size,
          actual_out_count);

    output_data.resize({(SIZE)actual_out_count});
    MemoryManager<DeviceType>::Copy1D(output_data.data(), out_data,
                                      actual_out_count, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (log::level & log::TIME) {
      timer.end();
      timer.print("Zstd decompress", actual_out_count);
      timer.clear();
    }
  }

  int compressionLevel;
  SIZE buffer_size = 0;
  Byte *in_data = nullptr;
  Byte *out_data = nullptr;
};

} // namespace mgard_x_160

#endif
