#ifndef MGARD_X_ZSTD_TEMPLATE_HPP
#define MGARD_X_ZSTD_TEMPLATE_HPP

#include <zstd.h>

namespace mgard_x {

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d CHECK(%s) failed: ", __FILE__, __LINE__, #cond);  \
      fprintf(stderr, "" __VA_ARGS__);                                         \
      fprintf(stderr, "\n");                                                   \
      exit(1);                                                                 \
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
    log::info("Zstd compression level: " + std::to_string(compressionLevel));
    log::info("Zstd compress ratio: " +
              std::to_string((double)(input_count) /
                             (actual_out_size + sizeof(size_t))));
    if (log::level & log::TIME) {
      timer.end();
      timer.print("Zstd compress");
      timer.print_throughput("Zstd compress", input_count);
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

    uint32_t actual_out_count = 0;
    actual_out_count = *reinterpret_cast<const size_t *>(in_data);
    Resize(actual_out_count, compressionLevel, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    ZSTD_decompress(out_data, actual_out_count, in_data + sizeof(size_t),
                    input_count - sizeof(size_t));

    output_data.resize({(SIZE)actual_out_count});
    MemoryManager<DeviceType>::Copy1D(output_data.data(), out_data,
                                      actual_out_count, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (log::level & log::TIME) {
      timer.end();
      timer.print("Zstd decompress");
      timer.print_throughput("Zstd decompress", actual_out_count);
      timer.clear();
    }
  }

  int compressionLevel;
  SIZE buffer_size = 0;
  Byte *in_data = nullptr;
  Byte *out_data = nullptr;
};

} // namespace mgard_x

#endif