#ifndef MGARD_X_CPU_LOSSLESS_TEMPLATE_HPP
#define MGARD_X_CPU_LOSSLESS_TEMPLATE_HPP

#include <zstd.h>

/*! CHECK
 * Check that the condition holds. If it doesn't print a message and die.
 */
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d CHECK(%s) failed: ", __FILE__, __LINE__, #cond);  \
      fprintf(stderr, "" __VA_ARGS__);                                         \
      fprintf(stderr, "\n");                                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
#define CHECK_ZSTD(fn, ...)                                                    \
  do {                                                                         \
    size_t const err = (fn);                                                   \
    CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err));                   \
  } while (0)

namespace mgard {
void huffman_encoding(long int *quantized_data, const std::size_t n,
                      unsigned char **out_data_hit, size_t *out_data_hit_size,
                      unsigned char **out_data_miss, size_t *out_data_miss_size,
                      unsigned char **out_tree, size_t *out_tree_size);
void huffman_decoding(long int *quantized_data,
                      const std::size_t quantized_data_size,
                      unsigned char *out_data_hit, size_t out_data_hit_size,
                      unsigned char *out_data_miss, size_t out_data_miss_size,
                      unsigned char *out_tree, size_t out_tree_size);
} // namespace mgard

namespace mgard_x {

template <typename DeviceType>
unsigned char *compress_memory_huffman(long int *const src,
                                       const std::size_t srcLen,
                                       std::size_t &outsize) {
  unsigned char *out_data_hit = 0;
  size_t out_data_hit_size;
  unsigned char *out_data_miss = 0;
  size_t out_data_miss_size;
  unsigned char *out_tree = 0;
  size_t out_tree_size;
  ::mgard::huffman_encoding(src, srcLen, &out_data_hit, &out_data_hit_size,
                            &out_data_miss, &out_data_miss_size, &out_tree,
                            &out_tree_size);

  const size_t total_size =
      out_data_hit_size / 8 + 4 + out_data_miss_size + out_tree_size;
  unsigned char *payload = (unsigned char *)malloc(total_size);
  unsigned char *bufp = payload;

  if (out_tree_size) {
    std::memcpy(bufp, out_tree, out_tree_size);
    bufp += out_tree_size;
  }

  std::memcpy(bufp, out_data_hit, out_data_hit_size / 8 + 4);
  bufp += out_data_hit_size / 8 + 4;

  if (out_data_miss_size) {
    std::memcpy(bufp, out_data_miss, out_data_miss_size);
    bufp += out_data_miss_size;
  }

  free(out_tree);
  free(out_data_hit);
  free(out_data_miss);

  const size_t cBuffSize = ZSTD_compressBound(total_size);
  unsigned char *const zstd_buffer = new unsigned char[cBuffSize];

  const std::size_t cSize =
      ZSTD_compress(zstd_buffer, cBuffSize, payload, total_size, 1);
  CHECK_ZSTD(cSize);

  free(payload);
  payload = 0;

  const std::size_t bufferLen = 3 * sizeof(size_t) + cSize;
  unsigned char *const buffer = new unsigned char[bufferLen];
  outsize = bufferLen;

  bufp = buffer;
  *(size_t *)bufp = out_tree_size;
  bufp += sizeof(size_t);

  *(size_t *)bufp = out_data_hit_size;
  bufp += sizeof(size_t);

  *(size_t *)bufp = out_data_miss_size;
  bufp += sizeof(size_t);

  {
    unsigned char const *const p = zstd_buffer;
    std::copy(p, p + cSize, bufp);
  }

  {
    unsigned char *buf = buffer;
    out_tree_size = *(size_t *)buf;
    buf += sizeof(size_t);

    out_data_hit_size = *(size_t *)buf;
    buf += sizeof(size_t);

    out_data_miss_size = *(size_t *)buf;
    buf += sizeof(size_t);
  }

  return buffer;
}

template <typename DeviceType>
void decompress_memory_huffman(unsigned char *const src,
                               const std::size_t srcLen, long int *const dst,
                               const std::size_t dstLen) {
  unsigned char *out_data_hit = 0;
  size_t out_data_hit_size;
  unsigned char *out_data_miss = 0;
  size_t out_data_miss_size;
  unsigned char *out_tree = 0;
  size_t out_tree_size;

  unsigned char *buf = src;

  out_tree_size = *(size_t *)buf;
  buf += sizeof(size_t);

  out_data_hit_size = *(size_t *)buf;
  buf += sizeof(size_t);

  out_data_miss_size = *(size_t *)buf;
  buf += sizeof(size_t);

  size_t total_huffman_size =
      out_tree_size + out_data_hit_size / 8 + 4 + out_data_miss_size;
  unsigned char *huffman_encoding_p =
      (unsigned char *)malloc(total_huffman_size);

  size_t const dSize = ZSTD_decompress(huffman_encoding_p, total_huffman_size,
                                       buf, srcLen - 3 * sizeof(size_t));
  CHECK_ZSTD(dSize);

  /* When zstd knows the content size, it will error if it doesn't match. */
  CHECK(total_huffman_size == dSize,
        "Impossible because zstd will check this condition!");

  out_tree = huffman_encoding_p;
  out_data_hit = huffman_encoding_p + out_tree_size;
  out_data_miss =
      huffman_encoding_p + out_tree_size + out_data_hit_size / 8 + 4;

  mgard::huffman_decoding(dst, dstLen, out_data_hit, out_data_hit_size,
                          out_data_miss, out_data_miss_size, out_tree,
                          out_tree_size);

  free(huffman_encoding_p);
}

template <typename C, typename DeviceType>
Array<1, Byte, DeviceType> CPUCompress(SubArray<1, C, DeviceType> &input_data) {

  // PrintSubarray("CPUCompress input", input_data);

  size_t input_count = input_data.getShape(0);

  C *in_data = NULL;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  std::vector<long int> qv(input_count);
  for (size_t i = 0; i < input_count; i++) {
    qv[i] = (long int)in_data[i];
  }

  std::size_t actual_out_size;
  unsigned char *lossless_data = compress_memory_huffman<DeviceType>(
      qv.data(), qv.size(), actual_out_size);

  uint8_t *out_data = NULL;
  MemoryManager<DeviceType>::MallocHost(out_data,
                                        actual_out_size + sizeof(size_t), 0);

  *(size_t *)out_data = (size_t)input_count;
  std::memcpy(out_data + sizeof(size_t), lossless_data, actual_out_size);

  Array<1, Byte, DeviceType> output_data(
      {(SIZE)(actual_out_size + sizeof(size_t))});
  output_data.load(out_data);

  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);
  delete[] lossless_data;

  // PrintSubarray("CPUCompress output", SubArray(output_data));

  return output_data;
}

template <typename C, typename DeviceType>
Array<1, C, DeviceType>
CPUDecompress(SubArray<1, Byte, DeviceType> &input_data) {

  // PrintSubarray("CPUDecompress input", input_data);
  size_t input_count = input_data.getShape(0);
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

  long int *qv = new long int[actual_out_count];
  size_t out_size = actual_out_count * sizeof(long int);
  decompress_memory_huffman<DeviceType>(
      in_data + sizeof(size_t), input_count - sizeof(size_t), qv, out_size);

  for (size_t i = 0; i < actual_out_count; i++) {
    out_data[i] = (C)qv[i];
  }

  Array<1, C, DeviceType> output_data({(SIZE)actual_out_count});
  output_data.load(out_data);

  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);

  // PrintSubarray("CPUDecompress output", SubArray(output_data));

  return output_data;
}

} // namespace mgard_x

#endif