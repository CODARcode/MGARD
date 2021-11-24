/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_MEMORY_MANAGEMENT_HPP
#define MGARD_X_MEMORY_MANAGEMENT_HPP

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

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

namespace mgard_x {

template <typename SubArrayType> 
void PrintSubarray(std::string name, SubArrayType subArray) {
  // Handle<1, float> tmp_handle;

  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray.getShape(0);
  if (SubArrayType::NumDims >= 2) ncol = subArray.getShape(1);
  if (SubArrayType::NumDims >= 3) nrow = subArray.getShape(2);

  using T = typename SubArrayType::DataType;

  std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * " << nfib << ") sizeof(T) = "  <<sizeof(T) << std::endl;

  T *v = new T[nrow * ncol * nfib];
  // cudaMemcpy3DAsyncHelper(tmp_handle, v, nfib * sizeof(T), nfib * sizeof(T),
  //                         ncol, subArray.data(), subArray.lddv1 * sizeof(T), nfib * sizeof(T), subArray.lddv2,
  //                         nfib * sizeof(T), ncol, nrow, D2H, 0);
  MemoryManager<CUDA>::CopyND(v, nfib, subArray.data(), subArray.getLddv1(),
                              nfib, ncol * nrow, 0);
  DeviceRuntime<CUDA>::SyncQueue(0);
  // tmp_handle.sync(0);
  
  
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

template <typename SubArrayType> 
void CompareSubarray(std::string name, SubArrayType subArray1, SubArrayType subArray2) {
  // Handle<1, float> tmp_handle;

  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray1.getShape(0);
  if (SubArrayType::NumDims >= 2) ncol = subArray1.getShape(1);
  if (SubArrayType::NumDims >= 3) nrow = subArray1.getShape(2);

  if (subArray1.getShape(0) != subArray2.getShape(0) ||
      subArray1.getShape(1) != subArray2.getShape(1) ||
      subArray1.getShape(2) != subArray2.getShape(2)) {
    std::cout << log::log_err <<"CompareSubarray: shape mismatch!\n";
    exit(-1);
  }

  using T = typename SubArrayType::DataType;

  std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * " << nfib << ") sizeof(T) = "  <<sizeof(T) << std::endl;

  T *v1 = new T[nrow * ncol * nfib];
  T *v2= new T[nrow * ncol * nfib];
  MemoryManager<CUDA>::CopyND(v1, nfib, subArray1.data(), subArray1.getLddv1(),
                              nfib, ncol * nrow, 0);
  MemoryManager<CUDA>::CopyND(v2, nfib, subArray2.data(), subArray2.getLddv1(),
                              nfib, ncol * nrow, 0);

  // Handle<1, float> tmp_handle;
  // cudaMemcpy3DAsyncHelper(tmp_handle, v1, nfib * sizeof(T), nfib * sizeof(T),
  //                         ncol, subArray1.data(), subArray1.lddv1 * sizeof(T), nfib * sizeof(T), subArray1.lddv2,
  //                         nfib * sizeof(T), ncol, nrow, D2H, 0);

  // cudaMemcpy3DAsyncHelper(tmp_handle, v2, nfib * sizeof(T), nfib * sizeof(T),
  //                         ncol, subArray2.data(), subArray2.lddv1 * sizeof(T), nfib * sizeof(T), subArray2.lddv2,
  //                         nfib * sizeof(T), ncol, nrow, D2H, 0);
  // gpuErrchk(cudaDeviceSynchronize());
  DeviceRuntime<CUDA>::SyncQueue(0);

  for (int i = 0; i < nrow; i++) {
    printf("[i = %d]\n", i);
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        T a = v1[nfib * ncol * i + nfib * j + k];
        T b = v2[nfib * ncol * i + nfib * j + k];
        if (fabs(a-b) > 1e-5) {
          std::cout << ANSI_RED;
        } else {
          std::cout << ANSI_GREEN;
        }
        if (std::is_same<T, std::uint8_t>::value) {
          std::cout << std::setw(8) << (unsigned int)v2[nfib * ncol * i + nfib * j + k] << ", ";
        } else {
          std::cout << std::setw(8) << std::setprecision(6) << std::fixed
                  << v2[nfib * ncol * i + nfib * j + k] << ", ";
        }
        std::cout << ANSI_RESET;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  delete [] v1;
  delete [] v2;
}


}

#endif