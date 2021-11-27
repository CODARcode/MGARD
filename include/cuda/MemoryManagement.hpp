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

// #include "MemoryManagement.h"

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


// print 3D CPU
template <typename T>
void verify_matrix(SIZE nrow, SIZE ncol, SIZE nfib, T *v, SIZE ldv1, SIZE ldv2,
                   std::string file_prefix, bool store, bool verify) {
  std::string filename = file_prefix + ".dat";
  if (store) {
    std::ofstream myfile;
    myfile.open(filename, std::ios::out | std::ios::binary);
    if (!myfile) {
      printf("Error: cannot write file\n");
      return;
    }
    myfile.write((char *)v, nrow * ncol * nfib * sizeof(T));
    myfile.close();
    if (!myfile.good()) {
      printf("Error occurred at write time!\n");
      return;
    }
  }
  if (verify) {
    std::fstream fin;
    fin.open(filename, std::ios::in | std::ios::binary);
    if (!fin) {
      printf("Error: cannot read file\n");
      return;
    }
    T *v2 = new T[nrow * ncol * nfib];
    fin.read((char *)v2, nrow * ncol * nfib * sizeof(T));
    fin.close();
    if (!fin.good()) {
      printf("Error occurred at reading time!\n");
      return;
    }

    bool mismatch = false;
    for (int i = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
        for (int k = 0; k < nfib; k++) {
          if (v[get_idx(ldv1, ldv2, i, j, k)] !=
              v2[get_idx(nfib, ncol, i, j, k)]) {
            std::cout << filename << ": ";
            printf("Mismatch[%d %d %d] %f - %f\n", i, j, k,
                   v[get_idx(ldv1, ldv2, i, j, k)],
                   v2[get_idx(nfib, ncol, i, j, k)]);
            mismatch = true;
          }
        }
      }
    }

    delete v2;
    if (mismatch)
      exit(-1);
  }
}

// print 3D GPU
template <typename T>
void verify_matrix_cuda(SIZE nrow, SIZE ncol, SIZE nfib, T *dv, SIZE lddv1,
                        SIZE lddv2, SIZE sizex, std::string file_prefix,
                        bool store, bool verify) {
  // std::cout << std::setw(10);
  // std::cout << std::setprecision(2) << std::fixed;
  if (store || verify) {
    // Handle<3, float> *tmp_handle = new Handle<3, float>();
    int queue_idx = 0;

    T *v = new T[nrow * ncol * nfib];
    // cudaMemcpy3DAsyncHelper(*tmp_handle, v, nfib * sizeof(T), nfib * sizeof(T),
    //                         ncol, dv, lddv1 * sizeof(T), sizex * sizeof(T),
    //                         lddv2, nfib * sizeof(T), ncol, nrow, D2H,
    //                         queue_idx);
    MemoryManager<CUDA>::CopyND(v, nfib, dv, lddv1,
                              nfib, ncol * nrow, 0);
    DeviceRuntime<CUDA>::SyncQueue(0);
    // tmp_handle->sync(queue_idx);
    verify_matrix(nrow, ncol, nfib, v, nfib, ncol, file_prefix, store, verify);
    delete[] v;
    // delete tmp_handle;
  }
}

}

#endif