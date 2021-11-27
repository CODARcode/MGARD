/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <vector>


 
#include "cuda/CommonInternal.h"

#include "cuda/CompressionWorkflow.h"

// #include "cuda/MemoryManagement.h"

#include "cuda/DataRefactoring.h"
#include "cuda/LinearQuantization.h"
#include "cuda/LinearQuantization.hpp"
#include "cuda/LosslessCompression.h"

#include "cuda/DeviceAdapters/DeviceAdapter.h"

#include "cuda/ParallelHuffman/Huffman.hpp"
#include "cuda/Lossless/LZ4.hpp"
#include "cuda/Utilities/CheckEndianess.hpp"

#define BLOCK_SIZE 64

using namespace std::chrono;

namespace mgard_x {

template <typename T>
struct linf_norm : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T x, T y) { return max(abs(x), abs(y)); }
};

template <typename T>
struct l2_norm : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return x*x; }
};

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType> compress(Handle<D, T> &handle, Array<D, T, DeviceType> &in_array,
                                 enum error_bound_type type, T tol, T s) {

  DeviceRuntime<DeviceType>::SelectDevice(handle.dev_id);
  Timer timer_total, timer_each;
  for (DIM i = 0; i < D; i++) {
    if (handle.shape[i] != in_array.getShape()[i]) {
      std::cout << log::log_err
                << "The shape of input array does not match the shape "
                   "initilized in handle!\n";
      std::vector<SIZE> empty_shape;
      empty_shape.push_back(1);
      Array<1, unsigned char, DeviceType> empty(empty_shape);
      return empty;
    }
  }

  SubArray in_subarray(in_array);
  SIZE total_elems = handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;

  if (handle.timing) timer_total.start();
  T norm = (T)1.0;

  if (type == error_bound_type::REL) {
    if (handle.timing) timer_each.start();
    thrust::device_vector<T> v_vec(handle.dofs[0][0] * handle.dofs[1][0] *
                                   handle.linearized_depth);

    Array<1, T, DeviceType> temp_array(
          {(SIZE)(handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth)}, 
          false);
    MemoryManager<DeviceType>().CopyND(
                      temp_array.get_dv(), handle.dofs[0][0],
                      in_array.get_dv(), in_array.get_ldvs_h()[0], 
                      handle.dofs[0][0], (SIZE)(handle.dofs[1][0] * handle.linearized_depth),
                      0);


    MemoryManager<DeviceType>().CopyND(
                      thrust::raw_pointer_cast(v_vec.data()), handle.dofs[0][0],
                      in_array.get_dv(), in_array.get_ldvs_h()[0], 
                      handle.dofs[0][0], (SIZE)(handle.dofs[1][0] * handle.linearized_depth),
                      0);

    SubArray temp_subarray(temp_array);
    Array<1, T, DeviceType> norm_array({1});
    SubArray norm_subarray(norm_array);

    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (s == std::numeric_limits<T>::infinity()) {
      // norm = thrust::reduce(v_vec.begin(), v_vec.end(), (T)0, linf_norm<T>());
      DeviceCollective<DeviceType>::AbsMax(total_elems, temp_subarray, norm_subarray, 0);
      DeviceRuntime<CUDA>::SyncQueue(0);
      norm = norm_array.getDataHost()[0];
    } else {
      // thrust::transform(v_vec.begin(), v_vec.end(), v_vec.begin(), l2_norm<T>());
      // norm = thrust::reduce(v_vec.begin(), v_vec.end(), (T)0);
      // printf("thrust norm: %f\n", norm);
      // norm = std::sqrt(norm);
      DeviceCollective<DeviceType>::SquareSum(total_elems, temp_subarray, norm_subarray, 0);
      DeviceRuntime<CUDA>::SyncQueue(0);
      norm = norm_array.getDataHost()[0];
      norm = std::sqrt(norm);
    }
    if (handle.timing) {
      timer_each.end();
      timer_each.print("Calculating norm");
      timer_each.clear();
    }
  }
  
  // Decomposition
  if (handle.timing) timer_each.start();
  decompose<D, T, DeviceType>(handle, in_subarray, handle.l_target, 0);
  DeviceRuntime<DeviceType>::SyncDevice();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Decomposition");
    timer_each.clear();
  }

  // Quantization
  bool prep_huffman = handle.lossless == lossless_type::GPU_Huffman ||
                      handle.lossless == lossless_type::GPU_Huffman_LZ4;

  if (handle.timing) timer_each.start();

  Array<D, QUANTIZED_INT, DeviceType> dqv_array(handle.shape_org, false);

  LENGTH estimate_outlier_count = (double)total_elems * 1;
  LENGTH zero = 0, outlier_count;
  Array<1, LENGTH, DeviceType> outlier_count_array({1});
  Array<1, LENGTH, DeviceType> outlier_idx_array({(SIZE)estimate_outlier_count});
  Array<1, QUANTIZED_INT, DeviceType> outliers_array({(SIZE)estimate_outlier_count});
  MemoryManager<DeviceType>::Copy1D(outlier_count_array.get_dv(), &zero, 1, 0);
  MemoryManager<DeviceType>::Memset1D(outlier_idx_array.get_dv(), estimate_outlier_count, 0);
  MemoryManager<DeviceType>::Memset1D(outliers_array.get_dv(), estimate_outlier_count, 0);

  Metadata m;
  m.ptype = processor_type::GPU_CUDA;
  m.ebtype = type;
  if (type == error_bound_type::REL) {
    m.norm = norm;
  }
  m.tol = tol;
  if (s == std::numeric_limits<T>::infinity()) {
    m.ntype = norm_type::L_Inf;
  } else {
    m.ntype = norm_type::L_2;
    m.s = s;
  }
  m.l_target = handle.l_target;
  m.ltype = handle.lossless;
  m.huff_dict_size = handle.huff_dict_size;
  m.huff_block_size = handle.huff_block_size;

  m.dtype = std::is_same<T, double>::value ? data_type::Double : data_type::Float;
  m.etype = CheckEndianess();
  m.dstype = handle.dstype;
  m.total_dims = D;
  m.shape = new uint64_t[D];
  for (int d = 0; d < D; d++) {
    m.shape[d] = (uint64_t)handle.dofs[D - 1 - d][0];
  }
  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    m.cltype = coordinate_location::Embedded;
    for (int d = 0; d < D; d++) {
      m.coords.push_back((Byte *)handle.coords_h[D - 1 - d]);
    }
  }


  // cudaMemGetInfo(&free, &total); printf("Mem: %f/%f\n",
  // (double)(total-free)/1e9, (double)total/1e9);

  T *quantizers = new T[m.l_target + 1];
  size_t dof = 1;
  for (int d = 0; d < D; d++) dof *= handle.dofs[d][0];
  calc_quantizers<D, T>(dof, quantizers, m, false);
  Array<1, T, DeviceType> quantizers_array({m.l_target + 1});
  quantizers_array.loadData(quantizers);
  SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);
  delete [] quantizers;

  LevelwiseLinearQuantizeND<D, T, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(handle.ranges), handle.l_target, quantizers_subarray,
          SubArray<2, T, DeviceType>(handle.volumes_array), 
          m, SubArray<D, T, DeviceType>(in_array),
          SubArray<D, QUANTIZED_INT, DeviceType>(dqv_array), prep_huffman,
          SubArray<1, SIZE, DeviceType>(handle.shapes[0], true),
          SubArray<1, LENGTH, DeviceType>(outlier_count_array), SubArray<1, LENGTH, DeviceType>(outlier_idx_array),
          SubArray<1, QUANTIZED_INT, DeviceType>(outliers_array),
          0);
  MemoryManager<DeviceType>::Copy1D(&outlier_count, outlier_count_array.get_dv(), 1, 0);

  DeviceRuntime<DeviceType>::SyncDevice();

  if (handle.timing) {
    timer_each.end();
    timer_each.print("Quantization");
    timer_each.clear();
  }

  // Huffman compression
  if (handle.timing) timer_each.start();

  uint64_t *hufdata;
  size_t hufdata_size;
  Array<1, Byte, DeviceType> huffman_array;
  Array<1, Byte, DeviceType> lz4_array;
  SubArray<1, Byte, DeviceType> lossless_compressed_subarray;

  // Cast to QUANTIZED_UNSIGNED_INT
  SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType> qv({total_elems},
                                                  (QUANTIZED_UNSIGNED_INT*)dqv_array.get_dv());
  huffman_array =
  HuffmanCompress<QUANTIZED_UNSIGNED_INT, uint64_t, DeviceType>(
      qv, handle.huff_block_size, handle.huff_dict_size);
  lossless_compressed_subarray = SubArray(huffman_array);

  handle.sync_all();

  if (handle.timing) {
    timer_each.end();
    timer_each.print("Huffman Compress");
    timer_each.clear();
  }

  // LZ4 compression
  if (handle.lossless == lossless_type::GPU_Huffman_LZ4) {
    if (handle.timing) timer_each.start();
    // lz4_compress(handle, hufdata, hufdata_size / sizeof(uint64_t), lz4_hufdata,
    //              lz4_hufdata_size, handle.lz4_block_size, 0);
    lz4_array = 
    LZ4Compress(lossless_compressed_subarray, handle.lz4_block_size);
    lossless_compressed_subarray = SubArray(lz4_array);

    if (handle.timing) {
      timer_each.end();
      timer_each.print("LZ4 Compress");
      timer_each.clear();
    }
  }

  if (handle.timing) {
    timer_total.end();
    timer_total.print("Overall Compress");
    std::cout << log::log_time << "Compression Throughput: " <<
      (double)(total_elems*sizeof(T))/timer_total.get()/1e9 << " GB/s)\n";
    timer_total.clear();
  }

  // Output serilization
  if (handle.timing) timer_each.start();

  uint32_t metadata_size;

  SERIALIZED_TYPE *serizalied_meta = m.Serialize(metadata_size);
  delete[] m.shape;

  SIZE outsize = 0;
  SIZE byte_offset = 0;
  advance_with_align<SERIALIZED_TYPE>(byte_offset, metadata_size);
  advance_with_align<LENGTH>(byte_offset, 1);
  advance_with_align<LENGTH>(byte_offset, outlier_count);
  advance_with_align<QUANTIZED_INT>(byte_offset, outlier_count);
  advance_with_align<SIZE>(byte_offset, 1);
  align_byte_offset<uint64_t>(byte_offset);
  advance_with_align<Byte>(byte_offset, lossless_compressed_subarray.getShape(0));

  outsize = byte_offset;
  DeviceRuntime<DeviceType>::SyncDevice();

  Array<1, unsigned char, DeviceType> compressed_array({outsize});
  SubArray compressed_subarray(compressed_array);

  SERIALIZED_TYPE *buffer = compressed_array.get_dv();
  void *buffer_p = (void *)buffer;
  byte_offset = 0;

  SerializeArray<SERIALIZED_TYPE>(compressed_subarray, serizalied_meta, metadata_size, byte_offset);
  SerializeArray<LENGTH>(compressed_subarray, outlier_count_array.get_dv(), 1, byte_offset);
  SerializeArray<LENGTH>(compressed_subarray, outlier_idx_array.get_dv(), outlier_count, byte_offset);
  SerializeArray<QUANTIZED_INT>(compressed_subarray, outliers_array.get_dv(), outlier_count, byte_offset);
  SIZE lossless_size = lossless_compressed_subarray.getShape(0);
  SerializeArray<SIZE>(compressed_subarray, &lossless_size, 1, byte_offset);

  align_byte_offset<uint64_t>(byte_offset);
  SerializeArray<Byte>(compressed_subarray, lossless_compressed_subarray.data(), 
                       lossless_compressed_subarray.getShape(0), byte_offset);

  handle.sync_all();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Serilization");
    timer_each.clear();
  }

  delete serizalied_meta;
  return compressed_array;  
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> decompress(Handle<D, T> &handle,
                       Array<1, unsigned char, DeviceType> &compressed_array) {
  DeviceRuntime<DeviceType>::SelectDevice(handle.dev_id);
  Timer timer_total, timer_each;

  SIZE total_elems = handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;

  SubArray compressed_subarray(compressed_array);
  SIZE byte_offset = 0;

  Metadata m;
  uint32_t metadata_size;
  SERIALIZED_TYPE * metadata_size_prt = (SERIALIZED_TYPE*)&metadata_size;
  byte_offset = m.metadata_size_offset();
  DeserializeArray<SERIALIZED_TYPE>(compressed_subarray, metadata_size_prt, sizeof(uint32_t), byte_offset, false);
  SERIALIZED_TYPE * serizalied_meta = (SERIALIZED_TYPE *)std::malloc(metadata_size);
  byte_offset = 0;
  DeserializeArray<SERIALIZED_TYPE>(compressed_subarray, serizalied_meta, metadata_size, byte_offset, false);

  m.Deserialize(serizalied_meta, metadata_size);

  if (m.etype != CheckEndianess()) {
    std::cout << log::log_err << "This data was compressed on a machine with different endianess!\n";
    exit(-1);
  }

  if (strcmp(m.magic_word, MAGIC_WORD) != 0) {
    std::cout << log::log_err << "This data was not compressed with MGARD or corrupted!\n";
    exit(-1);
  }

  if (m.ptype != processor_type::GPU_CUDA) {
    std::cout << log::log_err << "This data was not compressed with GPU, please use CPU to decompress!\n";
    exit(-1);
  }

  if (handle.timing) timer_each.start();
  void *lz4_hufmeta;
  size_t lz4_hufmeta_size;
  void *lz4_hufdata;
  size_t lz4_hufdata_size;

  uint8_t *hufmeta;
  Byte *hufdata;
  size_t hufmeta_size;
  SIZE hufdata_size;
  size_t outsize;

  LENGTH outlier_count;
  LENGTH * outlier_count_ptr = &outlier_count;
  LENGTH * outlier_idx;
  QUANTIZED_INT * outliers;
  SIZE lossless_size;
  SIZE * lossless_size_ptr = &lossless_size;
  Byte * lossless_data;
  
  DeserializeArray<LENGTH>(compressed_subarray, outlier_count_ptr, 1, byte_offset, false);
  DeserializeArray<LENGTH>(compressed_subarray, outlier_idx, outlier_count, byte_offset, true);
  DeserializeArray<QUANTIZED_INT>(compressed_subarray, outliers, outlier_count, byte_offset, true);
  DeserializeArray<SIZE>(compressed_subarray, lossless_size_ptr, 1, byte_offset, false);
  align_byte_offset<uint64_t>(byte_offset);
  DeserializeArray<Byte>(compressed_subarray, lossless_data, lossless_size, byte_offset, true);
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Deserilization");
    timer_each.clear();
  }

  DeviceRuntime<DeviceType>::SyncDevice();

  SubArray<1, LENGTH, DeviceType> outlier_idx_subarray({(SIZE)outlier_count}, outlier_idx);
  SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray({(SIZE)outlier_count}, outliers);
  SubArray<1, Byte, DeviceType> lossless_compressed_subarray({(SIZE) lossless_size}, lossless_data);
  

  Array<1, Byte, DeviceType> lz4_array;
  Array<1, Byte, DeviceType> huffman_array;

  if (handle.timing) timer_total.start();
  if (m.ltype == lossless_type::GPU_Huffman_LZ4) {
    if (handle.timing) timer_each.start();
    // uint64_t *lz4_decompressed_hufdata;
    // size_t lz4_decompressed_hufdata_size;
    // lz4_decompress(handle, (void *)hufdata, hufdata_size,
    //                lz4_decompressed_hufdata, lz4_decompressed_hufdata_size, 0);

    lz4_array = 
    LZ4Decompress<Byte, DeviceType>(lossless_compressed_subarray);
    lossless_compressed_subarray =  SubArray(lz4_array);
    DeviceRuntime<DeviceType>::SyncDevice();

    if (handle.timing) {
      timer_each.end();
      timer_each.print("LZ4 Decompress");
      timer_each.clear();
    }
  }

  QUANTIZED_UNSIGNED_INT * unsigned_dqv;
  if (handle.timing) timer_each.start();

  
  Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> primary =
  HuffmanDecompress<QUANTIZED_UNSIGNED_INT, uint64_t, DeviceType>(lossless_compressed_subarray);
  DeviceRuntime<DeviceType>::SyncDevice();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Huffman Decompress");
    timer_each.clear();
  }

  if (handle.timing) timer_each.start();

  std::vector<SIZE> decompressed_shape(D);
  for (int i = 0; i < D; i++)
    decompressed_shape[i] = handle.shape[i];
  std::reverse(decompressed_shape.begin(), decompressed_shape.end());
  Array<D, T, DeviceType> decompressed_data(decompressed_shape);
  SubArray<D, T, DeviceType> decompressed_subarray(decompressed_data);

  bool prep_huffman = m.ltype == lossless_type::GPU_Huffman ||
                      m.ltype == lossless_type::GPU_Huffman_LZ4;

  Array<D, QUANTIZED_INT, DeviceType> dqv_array(handle.shape_org, false);
  MemoryManager<DeviceType>::Copy1D(dqv_array.get_dv(), (QUANTIZED_INT*)primary.get_dv(), total_elems, 0);

  T *quantizers = new T[m.l_target + 1];
  size_t dof = 1;
  for (int d = 0; d < D; d++) dof *= handle.dofs[d][0];
  calc_quantizers<D, T>(dof, quantizers, m, false);
  Array<1, T, DeviceType> quantizers_array({m.l_target + 1});
  quantizers_array.loadData(quantizers);
  SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);
  delete [] quantizers;

  LevelwiseLinearDequantizeND<D, T, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(handle.ranges), handle.l_target, 
            quantizers_subarray,
            SubArray<2, T, DeviceType>(handle.volumes_array), 
            m, decompressed_subarray,
            SubArray<D, QUANTIZED_INT, DeviceType>(dqv_array), prep_huffman,
            SubArray<1, SIZE, DeviceType>(handle.shapes[0], true),
            outlier_count, outlier_idx_subarray, outliers_subarray, 0);

  DeviceRuntime<DeviceType>::SyncDevice();

  handle.sync_all();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Dequantization");
    timer_each.clear();
  }

  if (handle.timing) timer_each.start();
  recompose<D, T, DeviceType>(handle, decompressed_subarray, m.l_target, 0);
  handle.sync_all();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Recomposition");
    timer_each.clear();
  }

  handle.sync_all();
  if (handle.timing) {
    timer_total.end();
    timer_total.print("Overall Decompression");
    std::cout << log::log_time << "Decompression Throughput: " << 
      (double)(total_elems*sizeof(T))/timer_total.get()/1e9 << " GB/s)\n";
    timer_total.clear();
  }

  return decompressed_data;
}

#define KERNELS(D, T)                                                          \
  template Array<1, unsigned char, CUDA> compress<D, T, CUDA>(                             \
      Handle<D, T> & handle, Array<D, T, CUDA> & in_array,                           \
      enum error_bound_type type, T tol, T s);                                 \
  template Array<D, T, CUDA> decompress<D, T, CUDA>(                                       \
      Handle<D, T> & handle, Array<1, unsigned char, CUDA> & compressed_array);      

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS

} // namespace mgard_x
