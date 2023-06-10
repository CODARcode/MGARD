/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SERIALIZER_HH
#define MGARD_X_SERIALIZER_HH

namespace mgard_x {
template <typename T>
MGARDX_CONT_EXEC void align_byte_offset(SIZE &byte_offset) {
  if (byte_offset % sizeof(T) != 0) {
    byte_offset = ((byte_offset - 1) / sizeof(T) + 1) * sizeof(T);
  }
}

template <typename T>
MGARDX_CONT_EXEC void advance_with_align(SIZE &byte_offset, SIZE count) {
  align_byte_offset<T>(byte_offset);
  byte_offset += count * sizeof(T);
}

template <typename T, typename DeviceType>
void SerializeArray(SubArray<1, Byte, DeviceType> &array, T *data_ptr,
                    SIZE count, SIZE &byte_offset, int queue_idx) {
  using Mem = MemoryManager<DeviceType>;
  align_byte_offset<T>(byte_offset);
  Mem::Copy1D(array(byte_offset), (Byte *)data_ptr, count * sizeof(T),
              queue_idx);
  byte_offset += count * sizeof(T);
}

template <typename T, typename DeviceType>
void DeserializeArray(SubArray<1, Byte, DeviceType> &array, T *&data_ptr,
                      SIZE count, SIZE &byte_offset, bool zero_copy,
                      int queue_idx) {
  using Mem = MemoryManager<DeviceType>;
  align_byte_offset<T>(byte_offset);
  if (zero_copy) {
    data_ptr = (T *)array(byte_offset);
  } else {
    Mem::Copy1D((Byte *)data_ptr, array(byte_offset), count * sizeof(T),
                queue_idx);
  }
  byte_offset += count * sizeof(T);
}

template <typename T, typename DeviceType>
void Serialize(Byte *serialize_ptr, T *data_ptr, SIZE count, SIZE &byte_offset,
               int queue_idx) {
  using Mem = MemoryManager<DeviceType>;
  // align_byte_offset<T>(byte_offset);
  Mem::Copy1D(serialize_ptr + byte_offset, (Byte *)data_ptr, count * sizeof(T),
              queue_idx);
  byte_offset += count * sizeof(T);
}

template <typename T, typename DeviceType>
void Deserialize(Byte *serialize_ptr, T *&data_ptr, SIZE count,
                 SIZE &byte_offset, bool zero_copy, int queue_idx) {
  using Mem = MemoryManager<DeviceType>;
  // align_byte_offset<T>(byte_offset);
  if (zero_copy) {
    data_ptr = (T *)(serialize_ptr + byte_offset);
  } else {
    Mem::Copy1D((Byte *)data_ptr, serialize_ptr + byte_offset,
                count * sizeof(T), queue_idx);
  }
  byte_offset += count * sizeof(T);
}

} // namespace mgard_x

#endif