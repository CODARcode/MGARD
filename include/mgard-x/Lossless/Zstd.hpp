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

template <typename C, typename DeviceType>
Array<1, Byte, DeviceType> ZstdCompress(SubArray<1, C, DeviceType> &input_data,
                                        int compressionLevel) {
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  size_t input_count = input_data.shape(0);

  size_t const estimated_out_size = ZSTD_compressBound(input_count * sizeof(C));
  uint8_t *out_data = NULL; //(uint8_t *)malloc(cBuffSize);
  MemoryManager<DeviceType>::MallocHost(out_data,
                                        estimated_out_size + sizeof(size_t), 0);

  // assert(cBuff);

  C *in_data = NULL;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  size_t const actual_out_size =
      ZSTD_compress(out_data + sizeof(size_t), estimated_out_size, in_data,
                    input_count * sizeof(C), compressionLevel);
  CHECK_ZSTD(actual_out_size);

  *(size_t *)out_data = (size_t)input_count;

  Array<1, Byte, DeviceType> output_data(
      {(SIZE)(actual_out_size + sizeof(size_t))});
  output_data.load(out_data);

  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);
  log::info("Zstd compression level: " + std::to_string(compressionLevel));
  log::info("Zstd compress ratio: " +
            std::to_string((double)(input_count * sizeof(C)) /
                           (actual_out_size + sizeof(size_t))));
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer.end();
    timer.print("Zstd compress");
    timer.clear();
  }
  return output_data;

  // std::copy(cBuff, cBuff + cSize, back_inserter(out_data));

  // free(cBuff);
}

template <typename C, typename DeviceType>
Array<1, C, DeviceType>
ZstdDecompress(SubArray<1, Byte, DeviceType> &input_data) {
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  size_t input_count = input_data.shape(0);
  Byte *in_data = NULL;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  uint32_t actual_out_count = 0;
  actual_out_count = *reinterpret_cast<const size_t *>(in_data);
  // *oriData = (uint8_t*)malloc(outSize);
  C *out_data = NULL;
  MemoryManager<DeviceType>::MallocHost(out_data, actual_out_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  ZSTD_decompress(out_data, actual_out_count, in_data + sizeof(size_t),
                  input_count - sizeof(size_t));

  Array<1, C, DeviceType> output_data({(SIZE)actual_out_count});
  output_data.load(out_data);

  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncDevice();
    timer.end();
    timer.print("Zstd decompress");
    timer.clear();
  }
  return output_data;
}

} // namespace mgard_x

#endif