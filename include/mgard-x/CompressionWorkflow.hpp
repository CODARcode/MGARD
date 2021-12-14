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
#include <vector>


#include "Types.h"

#include "Handle.hpp"
#include "RuntimeX/RuntimeX.h"
#include "CompressionWorkflow.h"
#include "DataRefactoring/DataRefactoring.h"
#include "Quantization/LinearQuantization.hpp"
#include "Lossless/ParallelHuffman/Huffman.hpp"

#ifdef MGARDX_COMPILE_CUDA
#include "Lossless/LZ4.hpp"
#include "Lossless/Cascaded.hpp"
#endif

#include "Lossless/Zstd.hpp"
#include "Utilities/CheckEndianess.h"

// for debugging
// #include "../cuda/CommonInternal.h"
// #include "../cuda/DataRefactoring.h"
// #include "../cuda/SubArray.h"

#define BLOCK_SIZE 64

using namespace std::chrono;

namespace mgard_x {

static bool debug_print = true;

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType> compress(Handle<D, T, DeviceType> &handle, Array<D, T, DeviceType> &in_array,
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

    Array<1, T, DeviceType> temp_array;
    SubArray<1, T, DeviceType> temp_subarray;
    Array<1, T, DeviceType> norm_array({1});
    SubArray<1, T, DeviceType> norm_subarray(norm_array);
    if (MemoryManager<DeviceType>::ReduceMemoryFootprint) { // zero copy
      temp_subarray = SubArray<1, T, DeviceType>({total_elems}, in_array.get_dv());
    } else { // need to linearized
      temp_array = Array<1, T, DeviceType>(
            {(SIZE)(handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth)}, 
            false);
      MemoryManager<DeviceType>().CopyND(
                        temp_array.get_dv(), handle.dofs[0][0],
                        in_array.get_dv(), in_array.get_ldvs_h()[0], 
                        handle.dofs[0][0], (SIZE)(handle.dofs[1][0] * handle.linearized_depth),
                        0);
      temp_subarray = SubArray<1, T, DeviceType>(temp_array);
    }
    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (s == std::numeric_limits<T>::infinity()) {
      DeviceCollective<DeviceType>::AbsMax(total_elems, temp_subarray, norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = norm_array.getDataHost()[0];
    } else {
      DeviceCollective<DeviceType>::SquareSum(total_elems, temp_subarray, norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = norm_array.getDataHost()[0];
      norm = std::sqrt(norm);
    }
    if (handle.timing) {
      timer_each.end();
      timer_each.print("Calculating norm");
      timer_each.clear();
      if (s == std::numeric_limits<T>::infinity()) {
        std::cout << log::log_info << "L_inf norm: " << norm << std::endl;
      } else {
        std::cout << log::log_info << "L_2 norm: " << norm << std::endl;
      }
    }
  }
  
  // Decomposition
  if (handle.timing) timer_each.start();
  decompose<D, T, DeviceType>(handle, in_subarray, handle.l_target, 0);
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Decomposition");
    timer_each.clear();
  }

  // gpuErrchk(cudaDeviceSynchronize());

  // Quantization
  bool prep_huffman = true; // always do Huffman

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

  if (std::is_same<DeviceType, Serial>::value) {
    m.ptype = processor_type::X_Serial;
  } else if (std::is_same<DeviceType, CUDA>::value) {
    m.ptype = processor_type::X_CUDA;
  } else if (std::is_same<DeviceType, HIP>::value) {
    m.ptype = processor_type::X_HIP;
  }

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

  SubArray<1, LENGTH, DeviceType> outlier_idx_subarray(outlier_idx_array);
  SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray(outliers_array);

  LevelwiseLinearQuantizeND<D, T, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(handle.ranges), handle.l_target, quantizers_subarray,
          SubArray<2, T, DeviceType>(handle.volumes_array), 
          m, SubArray<D, T, DeviceType>(in_array),
          SubArray<D, QUANTIZED_INT, DeviceType>(dqv_array), prep_huffman,
          SubArray<1, SIZE, DeviceType>(handle.shapes[0], true),
          SubArray<1, LENGTH, DeviceType>(outlier_count_array), outlier_idx_subarray,
          outliers_subarray,
          0);
  MemoryManager<DeviceType>::Copy1D(&outlier_count, outlier_count_array.get_dv(), 1, 0);
  DeviceRuntime<DeviceType>::SyncDevice();
  m.huff_outlier_count = outlier_count;
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Quantization");
    timer_each.clear();
    std::cout << log::log_info << "Outlier ratio: " << 
    outlier_count << "/" << total_elems << " (" <<
    (double)100*outlier_count/total_elems << "%)\n"; 
  }
  if (debug_print) {
     // PrintSubarray("decompresed", SubArray<D, T, DeviceType>(in_array));
    // PrintSubarray("quantized parimary", SubArray<D, QUANTIZED_INT, DeviceType>(dqv_array));
    // std::cout << "outlier_count: " << outlier_count << std::endl;
    // PrintSubarray("quantized outliers_array", SubArray<1, QUANTIZED_INT, DeviceType>(outliers_array));
    // PrintSubarray("quantized outlier_idx_array", SubArray<1, LENGTH, DeviceType>(outlier_idx_array));
  }

  // Huffman compression
  if (handle.timing) timer_each.start();

  Array<1, Byte, DeviceType> huffman_array;
  Array<1, Byte, DeviceType> lossless_compressed_array;
  SubArray<1, Byte, DeviceType> lossless_compressed_subarray;

  // Cast to QUANTIZED_UNSIGNED_INT
  SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType> qv({total_elems},
                                                  (QUANTIZED_UNSIGNED_INT*)dqv_array.get_dv());
  huffman_array =
  HuffmanCompress<QUANTIZED_UNSIGNED_INT, uint64_t, DeviceType>(
      qv, handle.huff_block_size, handle.huff_dict_size, outlier_count, outlier_idx_subarray, outliers_subarray);
  lossless_compressed_subarray = SubArray(huffman_array);
  
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Huffman Compress");
    std::cout << log::log_info << "Huffman block size: " << 
      handle.huff_block_size << "\n"; 
    std::cout << log::log_info << "Huffman dictionary size: " << 
      handle.huff_dict_size << "\n"; 
    std::cout << log::log_info << "Huffman compress ratio: " << 
      total_elems*sizeof(QUANTIZED_UNSIGNED_INT) << "/" <<
      lossless_compressed_subarray.getShape(0) << " (" <<
      (double)total_elems*sizeof(QUANTIZED_UNSIGNED_INT) / lossless_compressed_subarray.getShape(0) << ")\n"; 
    timer_each.clear();
  }

  // LZ4 compression
  if (handle.lossless == lossless_type::Huffman_LZ4) {
#ifdef MGARDX_COMPILE_CUDA
    if (handle.timing) timer_each.start();
    SIZE lz4_before_size = lossless_compressed_subarray.getShape(0);
    lossless_compressed_array = 
    LZ4Compress(lossless_compressed_subarray, handle.lz4_block_size);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    SIZE lz4_after_size = lossless_compressed_subarray.getShape(0);
    if (handle.timing) {
      timer_each.end();
      timer_each.print("LZ4 Compress");
      std::cout << log::log_info << "LZ4 block size: " << 
      handle.lz4_block_size << "\n"; 
      std::cout << log::log_info << "LZ4 compress ratio: " << 
      (double)lz4_before_size / lz4_after_size << "\n"; 
      timer_each.clear();
    }
#else
    std::cout << log::log_warn << "LZ4 only available in CUDA. Switched to Zstd.\n";
    handle.lossless = lossless_type::Huffman_Zstd;
    m.ltype = handle.lossless; // update matadata
#endif
  } 

  if (handle.lossless == lossless_type::Huffman_Zstd) {
    if (handle.timing) timer_each.start();
    SIZE zstd_before_size = lossless_compressed_subarray.getShape(0);
    lossless_compressed_array = 
    ZstdCompress(lossless_compressed_subarray, handle.zstd_compress_level);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    SIZE zstd_after_size = lossless_compressed_subarray.getShape(0);
    if (handle.timing) {
      timer_each.end();
      timer_each.print("Zstd Compress");
      std::cout << log::log_info << "Zstd compression level: " << 
      handle.zstd_compress_level << "\n"; 
      std::cout << log::log_info << "Zstd compress ratio: " << 
      (double)zstd_before_size / zstd_after_size << "\n"; 
      timer_each.clear();
    }
  }

  if (handle.timing) {
    timer_total.end();
    timer_total.print("Overall Compress");
    std::cout << log::log_time << "Compression Throughput: " <<
      (double)(total_elems*sizeof(T))/timer_total.get()/1e9 << " GB/s\n";
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
  // advance_with_align<LENGTH>(byte_offset, 1);
  // advance_with_align<LENGTH>(byte_offset, outlier_count);
  // advance_with_align<QUANTIZED_INT>(byte_offset, outlier_count);
  advance_with_align<SIZE>(byte_offset, 1);
  align_byte_offset<uint64_t>(byte_offset); // for zero copy when deserialize 
  advance_with_align<Byte>(byte_offset, lossless_compressed_subarray.getShape(0));

  outsize = byte_offset;
  DeviceRuntime<DeviceType>::SyncDevice();

  Array<1, unsigned char, DeviceType> compressed_array({outsize});
  SubArray compressed_subarray(compressed_array);

  SERIALIZED_TYPE *buffer = compressed_array.get_dv();
  void *buffer_p = (void *)buffer;
  byte_offset = 0;

  SerializeArray<SERIALIZED_TYPE>(compressed_subarray, serizalied_meta, metadata_size, byte_offset);
  // SerializeArray<LENGTH>(compressed_subarray, outlier_count_array.get_dv(), 1, byte_offset);
  // SerializeArray<LENGTH>(compressed_subarray, outlier_idx_array.get_dv(), outlier_count, byte_offset);
  // SerializeArray<QUANTIZED_INT>(compressed_subarray, outliers_array.get_dv(), outlier_count, byte_offset);
  SIZE lossless_size = lossless_compressed_subarray.getShape(0);
  SerializeArray<SIZE>(compressed_subarray, &lossless_size, 1, byte_offset);

  align_byte_offset<uint64_t>(byte_offset);
  SerializeArray<Byte>(compressed_subarray, lossless_compressed_subarray.data(), 
                       lossless_compressed_subarray.getShape(0), byte_offset);
  // handle.sync_all();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Serilization");
    timer_each.clear();
  }

  if (debug_print) {
    // PrintSubarray("Final compressed_subarray", compressed_subarray);
  }

  free(serizalied_meta);
  return compressed_array;  
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> decompress(Handle<D, T, DeviceType> &handle,
                       Array<1, unsigned char, DeviceType> &compressed_array) {
  DeviceRuntime<DeviceType>::SelectDevice(handle.dev_id);
  Timer timer_total, timer_each;

  SIZE total_elems = handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;

  SubArray compressed_subarray(compressed_array);

  if (debug_print) {
    // PrintSubarray("Before decompression", compressed_subarray);
  }
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

  if (m.ptype == processor_type::GPU_CUDA) {
    std::cout << log::log_err << "This data was compressed with legacy CUDA compressor!\n";
    exit(-1);
  }

  if (m.ptype != processor_type::X_Serial &&
      m.ptype != processor_type::X_CUDA && 
      m.ptype != processor_type::X_HIP) {
    std::cout << log::log_err << "This data was not compressed with legacy MGARD-CUDA compressor or MGARD-X compressor!\n";
    exit(-1);
  }

  if (handle.timing) timer_each.start();

  size_t outsize;

  LENGTH outlier_count;
  LENGTH * outlier_count_ptr = &outlier_count;
  LENGTH * outlier_idx;
  QUANTIZED_INT * outliers;
  SIZE lossless_size;
  SIZE * lossless_size_ptr = &lossless_size;
  Byte * lossless_data;
  
  // DeserializeArray<LENGTH>(compressed_subarray, outlier_count_ptr, 1, byte_offset, false);
  // DeserializeArray<LENGTH>(compressed_subarray, outlier_idx, outlier_count, byte_offset, true);
  // DeserializeArray<QUANTIZED_INT>(compressed_subarray, outliers, outlier_count, byte_offset, true);
  DeserializeArray<SIZE>(compressed_subarray, lossless_size_ptr, 1, byte_offset, false);
  align_byte_offset<uint64_t>(byte_offset);
  DeserializeArray<Byte>(compressed_subarray, lossless_data, lossless_size, byte_offset, true);
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Deserilization");
    timer_each.clear();
  }

  DeviceRuntime<DeviceType>::SyncDevice();

  // SubArray<1, LENGTH, DeviceType> outlier_idx_subarray({(SIZE)outlier_count}, outlier_idx);
  // SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray({(SIZE)outlier_count}, outliers);
  SubArray<1, LENGTH, DeviceType> outlier_idx_subarray;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray;
  SubArray<1, Byte, DeviceType> lossless_compressed_subarray({(SIZE) lossless_size}, lossless_data);
  

  Array<1, Byte, DeviceType> lossless_compressed_array;
  Array<1, Byte, DeviceType> huffman_array;

  if (handle.timing) timer_total.start();

  if (m.ltype == lossless_type::Huffman_LZ4) {
#ifdef MGARDX_COMPILE_CUDA
    if (handle.timing) timer_each.start();
    lossless_compressed_array = LZ4Decompress<Byte, DeviceType>(lossless_compressed_subarray);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    DeviceRuntime<DeviceType>::SyncDevice();
    if (handle.timing) {
      timer_each.end();
      timer_each.print("LZ4 Decompress");
      timer_each.clear();
    }
#else
  std::cout << log::log_warn << "LZ4 only available in CUDA. Switched to Zstd.\n";
  handle.lossless = lossless_type::Huffman_Zstd;
  m.ltype = handle.lossless; // update matadata
#endif
  }


  if (m.ltype == lossless_type::Huffman_Zstd) {
    if (handle.timing) timer_each.start();
    lossless_compressed_array = ZstdDecompress<Byte, DeviceType>(lossless_compressed_subarray);
    lossless_compressed_subarray =  SubArray(lossless_compressed_array);
    DeviceRuntime<DeviceType>::SyncDevice();
    if (handle.timing) {
      timer_each.end();
      timer_each.print("Zstd Decompress");
      timer_each.clear();
    }
  }

  QUANTIZED_UNSIGNED_INT * unsigned_dqv;
  
  if (debug_print) {
    // PrintSubarray("Huffman lossless_compressed_subarray", lossless_compressed_subarray);
  }



  // SubArray<1, LENGTH, DeviceType> outlier_idx_subarray2({(SIZE)outlier_count}, outlier_idx);
  // SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray2({(SIZE)outlier_count}, outliers);

  if (handle.timing) timer_each.start();
  Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> primary =
  HuffmanDecompress<QUANTIZED_UNSIGNED_INT, uint64_t, DeviceType>(lossless_compressed_subarray,
                                                                  outlier_count,
                                                                  outlier_idx_subarray,
                                                                  outliers_subarray);
  DeviceRuntime<DeviceType>::SyncDevice();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Huffman Decompress");
    timer_each.clear();
  }

  // if (debug_print) {
  //   PrintSubarray("Quantized primary", SubArray(primary));
  // }


  if (handle.timing) timer_each.start();

  std::vector<SIZE> decompressed_shape(D);
  for (int i = 0; i < D; i++)
    decompressed_shape[i] = handle.shape[i];
  std::reverse(decompressed_shape.begin(), decompressed_shape.end());
  Array<D, T, DeviceType> decompressed_data(decompressed_shape);
  SubArray<D, T, DeviceType> decompressed_subarray(decompressed_data);

  bool prep_huffman = true;

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

  // handle.sync_all();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Dequantization");
    timer_each.clear();
  }


  // if (debug_print) {
  //   PrintSubarray2("Dequanzed primary", SubArray(primary));
  // }

  if (handle.timing) timer_each.start();
  recompose<D, T, DeviceType>(handle, decompressed_subarray, m.l_target, 0);
  // handle.sync_all();
  if (handle.timing) {
    timer_each.end();
    timer_each.print("Recomposition");
    timer_each.clear();
  }

  // handle.sync_all();
  if (handle.timing) {
    timer_total.end();
    timer_total.print("Overall Decompression");
    std::cout << log::log_time << "Decompression Throughput: " << 
      (double)(total_elems*sizeof(T))/timer_total.get()/1e9 << " GB/s\n";
    timer_total.clear();
  }

  // if (debug_print) {
  //   PrintSubarray2("decompressed_subarray", SubArray<D, T, DeviceType>(decompressed_subarray));
  // }


  return decompressed_data;
}

} // namespace mgard_x
