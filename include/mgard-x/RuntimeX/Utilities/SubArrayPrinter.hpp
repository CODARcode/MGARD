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
  using DeviceType = typename SubArrayType::DevType;

  std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * " << nfib << ") sizeof(T) = "  <<sizeof(T) << std::endl;
  // std::cout << name << "\n";

  T *v = new T[nrow * ncol * nfib];
  // cudaMemcpy3DAsyncHelper(tmp_handle, v, nfib * sizeof(T), nfib * sizeof(T),
  //                         ncol, subArray.data(), subArray.lddv1 * sizeof(T), nfib * sizeof(T), subArray.lddv2,
  //                         nfib * sizeof(T), ncol, nrow, D2H, 0);
  for (SIZE i = 0; i < nrow; i++) {
    MemoryManager<DeviceType>::CopyND(v + ncol * nfib * i, nfib, subArray.data() + subArray.getLddv1() * subArray.getLddv2() * i, subArray.getLddv1(),
                                nfib, ncol, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
  // tmp_handle.sync(0);
  
  
  for (int i = 0; i < nrow; i++) {
    printf("[i = %d]\n", i);
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        // std::cout << "[ " << j << ", " << k <<" ]: ";
        if (std::is_same<T, std::uint8_t>::value) {
          std::cout << std::setw(8) << (unsigned int)v[nfib * ncol * i + nfib * j + k] << " ";
        } else {
          std::cout << std::setw(8) << std::setprecision(6) << std::fixed
                  << v[nfib * ncol * i + nfib * j + k] << " ";
        }
        // std::cout << "\n";
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
  using DeviceType = typename SubArrayType::DevType;
  std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * " << nfib << ") sizeof(T) = "  <<sizeof(T) << std::endl;

  T *v1 = new T[nrow * ncol * nfib];
  T *v2= new T[nrow * ncol * nfib];

  for (SIZE i = 0; i < nrow; i++) {
    MemoryManager<DeviceType>::CopyND(v1 + ncol * nfib * i, nfib, subArray1.data() + subArray1.getLddv1() * subArray1.getLddv2() * i, subArray1.getLddv1(),
                                nfib, ncol, 0);
    MemoryManager<DeviceType>::CopyND(v2 + ncol * nfib * i, nfib, subArray2.data() + subArray2.getLddv1() * subArray2.getLddv2() * i, subArray2.getLddv1(),
                                nfib, ncol, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);

  T max_error = 0;
  bool pass = true;
  for (int i = 0; i < nrow; i++) {
    printf("[i = %d]\n", i);
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        T a = v1[nfib * ncol * i + nfib * j + k];
        T b = v2[nfib * ncol * i + nfib * j + k];
        max_error = std::max(max_error, fabs(a-b));
        if (fabs(a-b) > 1e-5 * fabs(a)) {
          std::cout << ANSI_RED;
          pass = false;
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

  printf("Check: %d, max error: %f\n", pass, max_error);
  delete [] v1;
  delete [] v2;
}


template <typename SubArrayType1, typename SubArrayType2> 
void CompareSubarray(std::string name, SubArrayType1 subArray1, SubArrayType2 subArray2, bool print, double error_thresold) {
  // Handle<1, float> tmp_handle;

  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray1.getShape(0);
  if (SubArrayType1::NumDims >= 2) ncol = subArray1.getShape(1);
  if (SubArrayType1::NumDims >= 3) nrow = subArray1.getShape(2);

  if (subArray1.getShape(0) != subArray2.shape[0] ||
      (SubArrayType1::NumDims >= 2 && subArray1.getShape(1) != subArray2.shape[1]) ||
      (SubArrayType1::NumDims >= 3 && subArray1.getShape(2) != subArray2.shape[2])) {
    std::cout << log::log_err <<"CompareSubarray: shape mismatch!\n";
    exit(-1);
  }

  using T = typename SubArrayType1::DataType;
  using DeviceType = typename SubArrayType1::DevType;
  // std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * " << nfib << ") sizeof(T) = "  <<sizeof(T) << std::endl;
  std::cout << name << "\n";

  T *v1 = new T[nrow * ncol * nfib];
  T *v2= new T[nrow * ncol * nfib];
  for (SIZE i = 0; i < nrow; i++) {
    MemoryManager<DeviceType>::CopyND(v1 + ncol * nfib * i, nfib, subArray1.data() + subArray1.getLddv1() * subArray1.getLddv2() * i, subArray1.getLddv1(),
                                nfib, ncol, 0);
    MemoryManager<DeviceType>::CopyND(v2 + ncol * nfib * i, nfib, subArray2.data() + subArray2.lddv1 * subArray2.lddv2 * i, subArray2.lddv1,
                                nfib, ncol, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);

  T max_error = 0;
  bool pass = true;
  for (int i = 0; i < nrow; i++) {
    if (print) printf("[i = %d]\n", i);
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        T a = v1[nfib * ncol * i + nfib * j + k];
        T b = v2[nfib * ncol * i + nfib * j + k];
        max_error = std::max(max_error, fabs(a-b));
        if (fabs(a-b) > error_thresold * fabs(a)) {
          if (print) std::cout << ANSI_RED;
          pass = false;
        } else {
          if (print) std::cout << ANSI_GREEN;
        }
        if (std::is_same<T, std::uint8_t>::value) {
          if (print) std::cout << std::setw(8) << (unsigned int)v2[nfib * ncol * i + nfib * j + k] << ", ";
        } else {
          if (print) std::cout << std::setw(8) << std::setprecision(6) << std::fixed
                  << v2[nfib * ncol * i + nfib * j + k] << ", ";
        }
        if (print) std::cout << ANSI_RESET;
      }
      if (print) std::cout << std::endl;
    }
    if (print) std::cout << std::endl;
  }
  if (print) std::cout << std::endl;

  printf("Check: %d, max error: %f\n", pass, max_error);
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

    delete [] v2;
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
    // MemoryManager<CUDA>::CopyND(v, nfib, dv, lddv1,
    //                           nfib, ncol * nrow, 0);
    // DeviceRuntime<CUDA>::SyncQueue(0);
    // tmp_handle->sync(queue_idx);
    verify_matrix(nrow, ncol, nfib, v, nfib, ncol, file_prefix, store, verify);
    delete[] v;
    // delete tmp_handle;
  }
}



template <typename T, typename DeviceType>
void CompareSubArrays(SubArray<1, T, DeviceType> array1, SubArray<1, T, DeviceType> array2) {
  SIZE n = array1.shape[0];
  using Mem = MemoryManager<DeviceType>;
  T * q1 = new T[n];
  T * q2 = new T[n];
  Mem::Copy1D(q1, array1.data(), n, 0);
  Mem::Copy1D(q2, array2.data(), n, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  bool pass = true;
  for (int i = 0; i < n; i++) {
    if (q1[i] != q2[i] ) {
      pass = false;
      std::cout << "diff at " << i << "(" << q1[i] << ", " << q2[i] << ")\n";
    }
  }
  printf("pass: %d\n", pass);
  delete [] q1;
  delete [] q2;
}

template <typename T, typename DeviceType>
void DumpSubArray(std::string name, SubArray<1, T, DeviceType> array) {
  SIZE n = array.getShape(0);
  using Mem = MemoryManager<DeviceType>;
  T * q = new T[n];
  Mem::Copy1D(q, array.data(), n, 0);
  std::fstream myfile;
  myfile.open(name, std::ios::out | std::ios::binary);
  if (!myfile) {
    printf("Error: cannot open file\n");
    return;
  }
  myfile.write((char *)q, n * sizeof(T));
  myfile.close();
  if (!myfile.good()) {
    printf("Error occurred at write time!\n");
    return;
  }
  delete [] q;
}

template <typename T, typename DeviceType>
void LoadSubArray(std::string name, SubArray<1, T, DeviceType> array) {
  SIZE n = array.getShape(0);
  using Mem = MemoryManager<DeviceType>;
  T * q = new T[n];
  std::fstream myfile;
  myfile.open(name, std::ios::in | std::ios::binary);
  if (!myfile) {
    printf("Error: cannot open file\n");
    return;
  }
  myfile.read((char *)q, n * sizeof(T));
  myfile.close();
  if (!myfile.good()) {
    printf("Error occurred at read time!\n");
    return;
  }
  Mem::Copy1D(array.data(), q, n, 0);
  delete [] q;
}

}

#endif