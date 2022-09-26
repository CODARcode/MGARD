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

template <typename DeviceType>
void ZstdCompress(Array<1, Byte, DeviceType> &data, int compressionLevel) {
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  Array<1, Byte, DeviceType> &input_data = data;
  Array<1, Byte, DeviceType> &output_data = data;
  size_t input_count = input_data.shape(0);

  size_t const estimated_out_size = ZSTD_compressBound(input_count);
  Byte *out_data = nullptr;
  MemoryManager<DeviceType>::MallocHost(out_data,
                                        estimated_out_size + sizeof(size_t), 0);
  Byte *in_data = nullptr;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  size_t const actual_out_size =
      ZSTD_compress(out_data + sizeof(size_t), estimated_out_size, in_data,
                    input_count, compressionLevel);
  CHECK_ZSTD(actual_out_size);

  *(size_t *)out_data = (size_t)input_count;

  output_data.resize({(SIZE)(actual_out_size + sizeof(size_t))});
  MemoryManager<DeviceType>::Copy1D(output_data.data(), out_data,
                                    actual_out_size + sizeof(size_t), 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);

  log::info("Zstd compression level: " + std::to_string(compressionLevel));
  log::info("Zstd compress ratio: " +
            std::to_string((double)(input_count) /
                           (actual_out_size + sizeof(size_t))));
  if (log::level & log::TIME) {
    timer.end();
    timer.print("Zstd compress");
    timer.clear();
  }
}

template <typename DeviceType>
void ZstdDecompress(Array<1, Byte, DeviceType> &data) {
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  Array<1, Byte, DeviceType> &input_data = data;
  Array<1, Byte, DeviceType> &output_data = data;
  size_t input_count = input_data.shape(0);
  Byte *in_data = nullptr;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  uint32_t actual_out_count = 0;
  actual_out_count = *reinterpret_cast<const size_t *>(in_data);
  Byte *out_data = nullptr;
  MemoryManager<DeviceType>::MallocHost(out_data, actual_out_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  ZSTD_decompress(out_data, actual_out_count, in_data + sizeof(size_t),
                  input_count - sizeof(size_t));

  output_data.resize({(SIZE)actual_out_count});
  MemoryManager<DeviceType>::Copy1D(output_data.data(), out_data,
                                    actual_out_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);
  if (log::level & log::TIME) {
    timer.end();
    timer.print("Zstd decompress");
    timer.clear();
  }
}

} // namespace mgard_x

#endif