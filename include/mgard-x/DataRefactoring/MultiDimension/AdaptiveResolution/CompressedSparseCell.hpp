/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COMPRESSED_SPARSE_CELL
#define MGARD_X_COMPRESSED_SPARSE_CELL

#include "Cell.hpp"

namespace mgard_x {

template <typename T, typename DeviceType>
class CompressedSparseCell {
public:
  CompressedSparseCell() {}
  CompressedSparseCell(std::vector<Cell<T, DeviceType>> cell_list){
    using Mem = MemoryManager<DeviceType>;

    num_cell = cell_list.size();

    for (int i = 0; i < 8; i++) {
      value_array[i] = Array<1, T, DeviceType>({num_cell});
      value[i] = SubArray(value_array[i]);
      for (SIZE j = 0; j < num_cell; j++) {
        Mem::Copy1D(value[i](j), &(cell_list[j].value[i]), 1, 0);
      }
    }

    for (int i = 0; i < 3; i++) {
      index_array[i] = Array<1, SIZE, DeviceType>({num_cell});
      index[i] = SubArray(index_array[i]);
      for (SIZE j = 0; j < num_cell; j++) {
        Mem::Copy1D(index[i](j), &(cell_list[j].index[i]), 1, 0);
      }
    }

    for (int i = 0; i < 3; i++) {
      size_array[i] = Array<1, SIZE, DeviceType>({num_cell});
      size[i] = SubArray(size_array[i]);
      for (SIZE j = 0; j < num_cell; j++) {
        Mem::Copy1D(size[i](j), &(cell_list[j].size[i]), 1, 0);
      }
    }
  }

  CompressedSparseCell(SIZE num_cell){
    this->num_cell = num_cell;
    using Mem = MemoryManager<DeviceType>;

    for (int i = 0; i < 8; i++) {
      value_array[i] = Array<1, T, DeviceType>({num_cell});
      value[i] = SubArray(value_array[i]);
    }

    for (int i = 0; i < 3; i++) {
      index_array[i] = Array<1, SIZE, DeviceType>({num_cell});
      index[i] = SubArray(index_array[i]);
    }

    for (int i = 0; i < 3; i++) {
      size_array[i] = Array<1, SIZE, DeviceType>({num_cell});
      size[i] = SubArray(size_array[i]);
    }
  }

  CompressedSparseCell(const CompressedSparseCell<T, DeviceType> &csc) {
    num_cell = csc.num_cell;
    for (int i = 0; i < 8; i++) {
      value_array[i] = csc.value_array[i];
      value[i] = SubArray(value_array[i]);
    }
    for (int i = 0; i < 3; i++) {
      index_array[i] = csc.index_array[i];
      index[i] = SubArray(index_array[i]);
    }
    for (int i = 0; i < 3; i++) {
      size_array[i] = csc.size_array[i];
      size[i] = SubArray(size_array[i]);
    }
  }

  ~CompressedSparseCell() {
  }

  void Print() {
    using Mem = MemoryManager<DeviceType>;
    T ** v = new T*[8];
    for (int i = 0; i < 8; i++) {
      v[i] = new T[num_cell];
      Mem::Copy1D(v[i], value[i].data(), num_cell, 0);
    }
    DeviceRuntime<DeviceType>::SyncQueue(0);

    for (int i = 0; i < num_cell; i++) {
      printf("cell[%d]: ", i);
      for (int j = 0; j < 8; j++) printf("%f ", v[j][i]);
      printf("\n");
    }

    for (int i = 0; i < 8; i++) {
      delete[] v[i];
    }
    delete v;

    // for (int i = 0; i < 8; i++) PrintSubarray("value[" + std::to_string(i) + "]", value[i]);
    // for (int i = 0; i < 3; i++) PrintSubarray("index[" + std::to_string(i) + "]", index[i]);
    // for (int i = 0; i < 3; i++) PrintSubarray("size[" + std::to_string(i) + "]", size[i]);
  }
  
  SIZE num_cell = 0;
  Array<1, T, DeviceType> value_array[8];
  Array<1, SIZE, DeviceType> index_array[3];
  Array<1, SIZE, DeviceType> size_array[3];

  SubArray<1, T, DeviceType> value[8];
  SubArray<1, SIZE, DeviceType> index[3];
  SubArray<1, SIZE, DeviceType> size[3];

};

}

#endif