/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_MEMORY_MANAGEMENT_HPP
#define MGARD_CUDA_MEMORY_MANAGEMENT_HPP

#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <sstream>   // std::stringstream
#include <stdexcept> // std::runtime_error
#include <string>
#include <utility> // std::pair
#include <vector>

#include "MemoryManagement.h"

namespace mgard_cuda {

template <typename SubArrayType> 
void PrintSubarray(std::string name, SubArrayType subArray) {
  Handle<1, float> tmp_handle;

  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray.shape[0];
  if (SubArrayType::NumDims >= 2) ncol = subArray.shape[1];
  if (SubArrayType::NumDims >= 3) nrow = subArray.shape[2];

  using T = typename SubArrayType::DataType;

  std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * " << nfib << ") sizeof(T) = "  <<sizeof(T) << std::endl;

  T *v = new T[nrow * ncol * nfib];
  cudaMemcpy3DAsyncHelper(tmp_handle, v, nfib * sizeof(T), nfib * sizeof(T),
                          ncol, subArray.data(), subArray.lddv1 * sizeof(T), nfib * sizeof(T), subArray.lddv2,
                          nfib * sizeof(T), ncol, nrow, D2H, 0);
  tmp_handle.sync(0);
  
  
  for (int i = 0; i < nrow; i++) {
    printf("[i = %d]\n", i);
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        if (std::is_same<T, std::uint8_t>::value) {
          std::cout << std::setw(8) << (unsigned int)v[nfib * ncol * i + nfib * j + k] << ", ";
        } else {
          std::cout << std::setw(8) << std::setprecision(6) << std::fixed
                  << v[nfib * ncol * i + nfib * j + k] << ", ";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  delete [] v;
}

}

#endif