/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
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

  DIM D = SubArrayType::NumDims;

  nfib = subArray.shape(D - 1);
  if (SubArrayType::NumDims >= 2)
    ncol = subArray.shape(D - 2);
  if (SubArrayType::NumDims >= 3)
    nrow = subArray.shape(D - 3);

  using T = typename SubArrayType::DataType;
  using DeviceType = typename SubArrayType::DevType;

  std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * "
            << nfib << ") sizeof(T) = " << sizeof(T) << std::endl;
  // std::cout << name << "\n";

  T *v = new T[nrow * ncol * nfib];
  // cudaMemcpy3DAsyncHelper(tmp_handle, v, nfib * sizeof(T), nfib * sizeof(T),
  //                         ncol, subArray.data(), subArray.lddv1 * sizeof(T),
  //                         nfib * sizeof(T), subArray.lddv2, nfib * sizeof(T),
  //                         ncol, nrow, D2H, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  for (SIZE i = 0; i < nrow; i++) {
    MemoryManager<DeviceType>::CopyND(
        v + ncol * nfib * i, nfib,
        subArray.data() + subArray.lddv1() * subArray.lddv2() * i,
        subArray.lddv1(), nfib, ncol, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
  // tmp_handle.sync(0);

  for (int i = 0; i < nrow; i++) {
    printf("[i = %d]\n", i);
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        // std::cout << "[ " << j << ", " << k <<" ]: ";
        if (std::is_same<T, std::uint8_t>::value) {
          std::cout << std::setw(8)
                    << (unsigned int)v[nfib * ncol * i + nfib * j + k] << " ";
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
  delete[] v;
}

template <typename SubArrayType>
void CompareSubarray(std::string name, SubArrayType subArray1,
                     SubArrayType subArray2) {
  // Handle<1, float> tmp_handle;

  DIM D = SubArrayType::NumDims;

  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray1.shape(D - 1);
  if (SubArrayType::NumDims >= 2)
    ncol = subArray1.shape(D - 2);
  if (SubArrayType::NumDims >= 3)
    nrow = subArray1.shape(D - 3);

  if (subArray1.shape(D - 1) != subArray2.shape(D - 1) ||
      subArray1.shape(D - 2) != subArray2.shape(D - 2) ||
      subArray1.shape(D - 3) != subArray2.shape(D - 3)) {
    std::cout << log::log_err << "CompareSubarray: shape mismatch!\n";
    exit(-1);
  }

  using T = typename SubArrayType::DataType;
  using DeviceType = typename SubArrayType::DevType;
  std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * "
            << nfib << ") sizeof(T) = " << sizeof(T) << std::endl;

  T *v1 = new T[nrow * ncol * nfib];
  T *v2 = new T[nrow * ncol * nfib];

  for (SIZE i = 0; i < nrow; i++) {
    MemoryManager<DeviceType>::CopyND(
        v1 + ncol * nfib * i, nfib,
        subArray1.data() + subArray1.lddv1() * subArray1.lddv2() * i,
        subArray1.lddv1(), nfib, ncol, 0);
    MemoryManager<DeviceType>::CopyND(
        v2 + ncol * nfib * i, nfib,
        subArray2.data() + subArray2.lddv1() * subArray2.lddv2() * i,
        subArray2.lddv1(), nfib, ncol, 0);
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
        max_error = std::max(max_error, fabs(a - b));
        if (fabs(a - b) > 1e-5 * fabs(a)) {
          std::cout << ANSI_RED;
          pass = false;
        } else {
          std::cout << ANSI_GREEN;
        }
        if (std::is_same<T, std::uint8_t>::value) {
          std::cout << std::setw(8)
                    << (unsigned int)v2[nfib * ncol * i + nfib * j + k] << ", ";
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
  delete[] v1;
  delete[] v2;
}

template <typename SubArrayType1, typename SubArrayType2>
void CompareSubarray(std::string name, SubArrayType1 subArray1,
                     SubArrayType2 subArray2, bool print,
                     double error_thresold) {
  // Handle<1, float> tmp_handle;

  DIM D = SubArrayType1::NumDims;

  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray1.shape(D - 1);
  if (SubArrayType1::NumDims >= 2)
    ncol = subArray1.shape(D - 2);
  if (SubArrayType1::NumDims >= 3)
    nrow = subArray1.shape(D - 3);

  if (subArray1.shape(D - 1) != subArray2.shape[0] ||
      (SubArrayType1::NumDims >= 2 &&
       subArray1.shape(D - 2) != subArray2.shape[1]) ||
      (SubArrayType1::NumDims >= 3 &&
       subArray1.shape(D - 3) != subArray2.shape[2])) {
    std::cout << log::log_err << "CompareSubarray: shape mismatch!\n";
    exit(-1);
  }

  using T = typename SubArrayType1::DataType;
  using DeviceType = typename SubArrayType1::DevType;
  // std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * "
  // << nfib << ") sizeof(T) = "  <<sizeof(T) << std::endl;
  std::cout << name << "\n";

  T *v1 = new T[nrow * ncol * nfib];
  T *v2 = new T[nrow * ncol * nfib];
  for (SIZE i = 0; i < nrow; i++) {
    MemoryManager<DeviceType>::CopyND(
        v1 + ncol * nfib * i, nfib,
        subArray1.data() + subArray1.lddv1() * subArray1.lddv2() * i,
        subArray1.lddv1(), nfib, ncol, 0);
    MemoryManager<DeviceType>::CopyND(v2 + ncol * nfib * i, nfib,
                                      subArray2.data() +
                                          subArray2.lddv1 * subArray2.lddv2 * i,
                                      subArray2.lddv1, nfib, ncol, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);

  T max_error = 0;
  bool pass = true;
  for (int i = 0; i < nrow; i++) {
    if (print)
      printf("[i = %d]\n", i);
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        T a = v1[nfib * ncol * i + nfib * j + k];
        T b = v2[nfib * ncol * i + nfib * j + k];
        max_error = std::max(max_error, fabs(a - b));
        if (fabs(a - b) > error_thresold * fabs(a)) {
          if (print)
            std::cout << ANSI_RED;
          pass = false;
        } else {
          if (print)
            std::cout << ANSI_GREEN;
        }
        if (std::is_same<T, std::uint8_t>::value) {
          if (print)
            std::cout << std::setw(8)
                      << (unsigned int)v2[nfib * ncol * i + nfib * j + k]
                      << ", ";
        } else {
          if (print)
            std::cout << std::setw(8) << std::setprecision(6) << std::fixed
                      << v2[nfib * ncol * i + nfib * j + k] << ", ";
        }
        if (print)
          std::cout << ANSI_RESET;
      }
      if (print)
        std::cout << std::endl;
    }
    if (print)
      std::cout << std::endl;
  }
  if (print)
    std::cout << std::endl;

  printf("Check: %d, max error: %f\n", pass, max_error);
  delete[] v1;
  delete[] v2;
}

template <typename SubArrayType>
void CompareSubarray4D(SubArrayType subArray1, SubArrayType subArray2) {
  if (SubArrayType::NumDims != 4) {
    std::cout << log::log_err
              << "CompareSubarray4D expects 4D subarray type.\n";
    exit(-1);
  }

  DIM D = SubArrayType::NumDims;

  if (subArray1.shape(D - 4) != subArray2.shape(D - 4)) {
    std::cout << log::log_err << "CompareSubarray4D mismatch 4D size.\n";
    exit(-1);
  }

  using T = typename SubArrayType::DataType;
  SIZE idx[4] = {0, 0, 0, 0};
  for (SIZE i = 0; i < subArray1.shape(0); i++) {
    idx[3] = i;
    SubArrayType temp1 = subArray1;
    SubArrayType temp2 = subArray2;
    // Adding offset to the 4th dim. (slowest)
    temp1.offset_dim(0, i);
    temp2.offset_dim(0, i);
    // Make 3D slice on the other three dims
    CompareSubarray("4D = " + std::to_string(i), temp1.Slice3D(1, 2, 3),
                    temp2.Slice3D(1, 2, 3));
  }
}

template <typename SubArrayType>
void PrintSubarray4D(std::string name, SubArrayType subArray1) {
  if (SubArrayType::NumDims != 4) {
    std::cout << log::log_err << "PrintSubarray4D expects 4D subarray type.\n";
    exit(-1);
  }

  DIM D = SubArrayType::NumDims;

  std::cout << name << "\n";
  using T = typename SubArrayType::DataType;
  SIZE idx[4] = {0, 0, 0, 0};
  for (SIZE i = 0; i < subArray1.shape(D - 4); i++) {
    idx[3] = i;
    SubArrayType temp1 = subArray1;
    temp1.offset_dim(3, i);
    PrintSubarray("i = " + std::to_string(i), temp1.Slice3D(0, 1, 2));
  }
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
    printf("Mismatch result: %d\n", mismatch);

    delete[] v2;
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
    // cudaMemcpy3DAsyncHelper(*tmp_handle, v, nfib * sizeof(T), nfib *
    // sizeof(T),
    //                         ncol, dv, lddv1 * sizeof(T), sizex * sizeof(T),
    //                         lddv2, nfib * sizeof(T), ncol, nrow, D2H,
    //                         queue_idx);
    // MemoryManager<DeviceType>::CopyND(v, nfib, dv, lddv1, nfib, ncol * nrow,
    // 0); DeviceRuntime<CUDA>::SyncQueue(0); tmp_handle->sync(queue_idx);
    verify_matrix(nrow, ncol, nfib, v, nfib, ncol, file_prefix, store, verify);
    delete[] v;
    // delete tmp_handle;
  }
}

template <typename T, typename DeviceType>
void CompareSubArrays(SubArray<1, T, DeviceType> array1,
                      SubArray<1, T, DeviceType> array2) {
  SIZE n = array1.shape[0];
  using Mem = MemoryManager<DeviceType>;
  T *q1 = new T[n];
  T *q2 = new T[n];
  Mem::Copy1D(q1, array1.data(), n, 0);
  Mem::Copy1D(q2, array2.data(), n, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  bool pass = true;
  for (int i = 0; i < n; i++) {
    if (q1[i] != q2[i]) {
      pass = false;
      std::cout << "diff at " << i << "(" << q1[i] << ", " << q2[i] << ")\n";
    }
  }
  printf("pass: %d\n", pass);
  delete[] q1;
  delete[] q2;
}

template <DIM D, typename T, typename DeviceType>
void DumpSubArray(std::string name, SubArray<D, T, DeviceType> subArray) {
  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray.shape(D - 1);
  if (D >= 2)
    ncol = subArray.shape(D - 2);
  if (D >= 3)
    nrow = subArray.shape(D - 3);

  T *v = new T[nrow * ncol * nfib];
  DeviceRuntime<DeviceType>::SyncQueue(0);
  MemoryManager<DeviceType>::CopyND(v, nfib, subArray.data(),
                                    subArray.ld(D - 1), nfib, ncol * nrow, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  std::fstream myfile;
  myfile.open(name, std::ios::out | std::ios::binary);
  if (!myfile) {
    printf("Error: cannot open file\n");
    return;
  }
  myfile.write((char *)v, nrow * ncol * nfib * sizeof(T));
  myfile.close();
  if (!myfile.good()) {
    printf("Error occurred at write time!\n");
    return;
  }
  delete[] v;
}

template <DIM D, typename T, typename DeviceType>
void LoadSubArray(std::string name, SubArray<D, T, DeviceType> subArray) {

  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray.shape(D - 1);
  if (D >= 2)
    ncol = subArray.shape(D - 2);
  if (D >= 3)
    nrow = subArray.shape(D - 3);

  T *v = new T[nrow * ncol * nfib];

  std::fstream myfile;
  myfile.open(name, std::ios::in | std::ios::binary);
  if (!myfile) {
    printf("Error: cannot open file\n");
    return;
  }
  myfile.read((char *)v, nrow * ncol * nfib * sizeof(T));
  myfile.close();
  if (!myfile.good()) {
    printf("Error occurred at read time!\n");
    return;
  }
  MemoryManager<DeviceType>::CopyND(subArray.data(), subArray.ld(D - 1), v,
                                    nfib, nfib, ncol * nrow, 0);
  delete[] v;
}

template <DIM D, typename T, typename DeviceType>
void VerifySubArray(std::string name, SubArray<D, T, DeviceType> subArray,
                    bool dump, bool verify) {
  if (dump) {
    printf("dump %s\n", name.c_str());
    DumpSubArray(name, subArray);
  }
  if (verify) {
    SIZE nrow = 1;
    SIZE ncol = 1;
    SIZE nfib = 1;

    nfib = subArray.shape(D - 1);
    if (D >= 2)
      ncol = subArray.shape(D - 2);
    if (D >= 3)
      nrow = subArray.shape(D - 3);

    T *v1 = new T[nrow * ncol * nfib];
    T *v2 = new T[nrow * ncol * nfib];

    std::fstream myfile;
    myfile.open(name, std::ios::in | std::ios::binary);
    if (!myfile) {
      printf("Error: cannot open file\n");
      return;
    }
    myfile.read((char *)v1, nrow * ncol * nfib * sizeof(T));
    myfile.close();
    if (!myfile.good()) {
      printf("Error occurred at read time!\n");
      return;
    }

    MemoryManager<DeviceType>::CopyND(v2, nfib, subArray.data(),
                                      subArray.ld(D - 1), nfib, ncol * nrow, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);

    printf("verify %s\n", name.c_str());
    bool pass = true;
    for (int i = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
        for (int k = 0; k < nfib; k++) {
          if (v1[i * ncol * nfib + j * nfib + k] !=
              v2[i * ncol * nfib + j * nfib + k]) {
            pass = false;
            printf("diff at [%d][%d][%d]: ", i, j, k);
            std::cout << v1[i * ncol * nfib + j * nfib + k] << ", "
                      << v2[i * ncol * nfib + j * nfib + k] << "\n";
          }
        }
      }
    }
    printf("pass: %d\n", pass);
    delete[] v1;
    delete[] v2;
  }
}

} // namespace mgard_x

#endif
