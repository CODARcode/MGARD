#ifndef MGARD_X_CPU_LOSSLESS_TEMPLATE_HPP
#define MGARD_X_CPU_LOSSLESS_TEMPLATE_HPP

#include "../../compressors.hpp"

namespace mgard_x {


template <typename C, typename DeviceType>
Array<1, Byte, DeviceType> 
CPUCompress(SubArray<1, C, DeviceType> &input_data) {

  // PrintSubarray("CPUCompress input", input_data);

  size_t input_count = input_data.getShape(0);

  C * in_data = NULL;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);


  std::vector<long int> qv(input_count);
  for (size_t i = 0; i < input_count; i++) {
    qv[i] = (long int)in_data[i];
  }

  std::size_t actual_out_size;
  unsigned char * lossless_data = mgard::compress_memory_huffman(qv, actual_out_size);

  uint8_t *out_data = NULL;
  MemoryManager<DeviceType>::MallocHost(out_data, actual_out_size + sizeof(size_t), 0);


  *(size_t*)out_data = (size_t)input_count;
  std::memcpy(out_data + sizeof(size_t), lossless_data, actual_out_size);

  Array<1, Byte, DeviceType> output_data({(SIZE)(actual_out_size + sizeof(size_t))});
  output_data.loadData(out_data);

  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);
  delete [] lossless_data;

  // PrintSubarray("CPUCompress output", SubArray(output_data));

  return output_data;
}


template <typename C, typename DeviceType>
Array<1, C, DeviceType>
CPUDecompress(SubArray<1, Byte, DeviceType> &input_data) {

  // PrintSubarray("CPUDecompress input", input_data);
  size_t input_count = input_data.getShape(0);
  Byte * in_data = NULL;
  MemoryManager<DeviceType>::MallocHost(in_data, input_count, 0);
  MemoryManager<DeviceType>::Copy1D(in_data, input_data.data(), input_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  uint32_t actual_out_count = 0;
  actual_out_count = *reinterpret_cast<const size_t*>(in_data);
  // *oriData = (uint8_t*)malloc(outSize);
  C * out_data = NULL;
  MemoryManager<DeviceType>::MallocHost(out_data, actual_out_count, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  long int * qv = new long int[actual_out_count];
  size_t out_size = actual_out_count * sizeof(long int);
  mgard::decompress_memory_huffman(in_data + sizeof(size_t), input_count - sizeof(size_t),
                             qv, out_size);

  for (size_t i = 0; i < actual_out_count; i++) {
    out_data[i] = (C)qv[i];
  }

  Array<1, C, DeviceType> output_data({(SIZE)actual_out_count});
  output_data.loadData(out_data);

  MemoryManager<DeviceType>::FreeHost(out_data);
  MemoryManager<DeviceType>::FreeHost(in_data);

  // PrintSubarray("CPUDecompress output", SubArray(output_data));


  return output_data;
}

}

#endif