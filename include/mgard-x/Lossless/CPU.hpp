#ifndef MGARD_X_CPU_LOSSLESS_TEMPLATE_HPP
#define MGARD_X_CPU_LOSSLESS_TEMPLATE_HPP

#include "proto/mgard.pb.h"
#include <lossless.hpp>

namespace mgard_x {

template <typename C> mgard::pb::Header setup_header() {
  mgard::pb::Header header;
  mgard::pb::Quantization &q = *header.mutable_quantization();
  if (std::is_same<C, std::int8_t>::value) {
    q.set_type(mgard::pb::Quantization::INT8_T);
  } else if (std::is_same<C, std::int16_t>::value) {
    q.set_type(mgard::pb::Quantization::INT16_T);
  } else if (std::is_same<C, std::int32_t>::value) {
    q.set_type(mgard::pb::Quantization::INT32_T);
  } else if (std::is_same<C, std::int64_t>::value) {
    q.set_type(mgard::pb::Quantization::INT64_T);
  }
  mgard::pb::Encoding &e = *header.mutable_encoding();
  // MGARD-X requires Zstd, so we always use CPU_HUFFMAN_ZSTD
  e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
  e.set_serialization(mgard::pb::Encoding::RFMH);
  return header;
}

template <typename C, typename DeviceType>
Array<1, Byte, DeviceType> CPUCompress(SubArray<1, C, DeviceType> &input_data) {

  size_t input_count = input_data.getShape(0);

  C *in_data = NULL;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  mgard::MemoryBuffer<unsigned char> lossless_data_buffer =
      mgard::compress(setup_header<C>(), in_data, input_count * sizeof(C));
  unsigned char *lossless_data = lossless_data_buffer.data.get();
  std::size_t actual_out_size = lossless_data_buffer.size;

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
  return output_data;
}

template <typename C, typename DeviceType>
Array<1, C, DeviceType>
CPUDecompress(SubArray<1, Byte, DeviceType> &input_data) {

  size_t input_count = input_data.getShape(0);
  Byte *in_data = NULL;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  uint32_t actual_out_count = 0;
  actual_out_count = *reinterpret_cast<const size_t *>(in_data);
  C *out_data = NULL;
  MemoryManager<DeviceType>::MallocHost(out_data, actual_out_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  mgard::decompress(setup_header<C>(), in_data + sizeof(size_t),
                    input_count - sizeof(size_t), out_data,
                    actual_out_count * sizeof(C));

  Array<1, C, DeviceType> output_data({(SIZE)actual_out_count});
  output_data.load(out_data);

  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);

  return output_data;
}

} // namespace mgard_x

#endif