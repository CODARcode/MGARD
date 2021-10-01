#include "cuda/CommonInternal.h"
#include "cuda/Functor.h"
#include "cuda/AutoTuner.h"
#include "cuda/Task.h"
#include "cuda/DeviceAdapters/DeviceAdapterCuda.h"
#include "cuda/MDR/BitplaneEncoder/BitplaneEncoder.hpp"
#include <mma.h>
#include <iostream>
#include <fstream>
#include <cuda_profiler_api.h>
using namespace nvcuda;

#include <chrono>
using namespace std::chrono;
template <typename T>
MGARDm_CONT_EXEC void
print_bits(T v, int num_bits, bool reverse = false) {
  for (int j = 0; j < num_bits; j++) {
    if (!reverse) printf("%u", (v >> sizeof(T)*8-1-j) & 1u);
    else printf("%u", (v >> j) & 1u);
  }
}


template <typename T_data, typename T_bitplane, mgard_cuda::OPTION NumGroupsPerWarpPerIter, mgard_cuda::SIZE NumWarpsPerTB, mgard_cuda::OPTION BinaryType, mgard_cuda::OPTION DataEncodingAlgorithm, mgard_cuda::OPTION ErrorCollectingAlgorithm, mgard_cuda::OPTION DataDecodingAlgorithm>
void test(mgard_cuda::LENGTH n, 
          mgard_cuda::SIZE encoding_num_bitplanes, 
          mgard_cuda::SIZE decoding_num_bitplanes,
          int warmup, int repeat){
  
  using T_sfp = typename std::conditional<std::is_same<T_data, double>::value, int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
  using HandleType = mgard_cuda::Handle<3, float>;
  using T_error = float;
  HandleType handle({5, 5, 5});

  double total_data = (n*encoding_num_bitplanes)/8/1e9;
  printf("T_data: %d, T_bitplane: %d, NumGroups: %u, NumWarps: %u, BinaryType: %d, Encoding: %d, Error: %d, Decoding: %d, n: %u, Total data: %.2e GB.\n", 
          sizeof(T_data)*8, sizeof(T_bitplane)*8, NumGroupsPerWarpPerIter, NumWarpsPerTB, BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm, n, total_data);

  std::ofstream ofs;
  std::string filename = "pref_"+
                         std::to_string(sizeof(T_data)*8) + "_" +
                         std::to_string(sizeof(T_bitplane)*8) + "_" +
                         std::to_string(n) + "_" +
                         std::to_string(encoding_num_bitplanes) + "_" +
                         std::to_string(decoding_num_bitplanes) + "_" +
                         std::to_string(BinaryType) + "_" +
                         std::to_string(DataEncodingAlgorithm) + "_" +
                         std::to_string(ErrorCollectingAlgorithm) + "_" +
                         std::to_string(DataDecodingAlgorithm) + ".csv";

  std::string path = __FILE__;
  path.erase (path.end()-std::string("test_encoding_warp.cu").length(), path.end());
  ofs.open (path + "/encoding_perf_results/" + filename, std::ofstream::out );
  if(!ofs) std::cout<<"Writing to file failed"<<std::endl;

  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;

  T_data * v = new T_data[n];
  T_data * v2 = NULL;
  for (int i = 0; i < n; i++) {
    // v[i] = rand() - RAND_MAX / 2;// % 10000000;
    v[i] = rand() % 100;
  }




  mgard_cuda::Array<1, T_data> v_array({(mgard_cuda::SIZE)n});
  v_array.loadData(v);
  mgard_cuda::SubArray<1, T_data> v_subarray(v_array);
  // mgard_cuda::PrintSubarray("v_subarray", v_subarray);

  mgard_cuda::Array<1, T_error> level_errors_array({encoding_num_bitplanes+1});
  mgard_cuda::SubArray<1, T_error> level_errors(level_errors_array);

  mgard_cuda::Array<1, bool> signs_array({(mgard_cuda::SIZE)n});
  mgard_cuda::SubArray<1, bool> signs_subarray(signs_array);

  // mgard_cuda::Array<1, T_data> v2_array({(mgard_cuda::SIZE)n});
  // mgard_cuda::SubArray<1, T_data> v2_subarray(v2_array);


  mgard_cuda::Array<1, T_data> result_array({1});
  mgard_cuda::SubArray<1, T_data> result(result_array);
  mgard_cuda::DeviceReduce<HandleType, T_data, mgard_cuda::CUDA> deviceReduce(handle);
  deviceReduce.AbsMax(v_subarray.shape[0], v_subarray, result, 0);
  handle.sync_all();

  mgard_cuda::SIZE starting_bitplane = 0;
  T_data level_max_error = *(result_array.getDataHost());
  int exp = 0;
  frexp(level_max_error, &exp);

  std::vector<uint32_t> stream_sizes;
  std::vector<double> level_sq_err; 
  // CPU
  { 
    total_data = (n*encoding_num_bitplanes)/8/1e9;
    auto encoder = mgard_cuda::MDR::GroupedBPEncoder<3, T_data, T_bitplane>(handle);
    t1 = high_resolution_clock::now();
    auto streams = encoder.encode(v, n, exp, encoding_num_bitplanes, stream_sizes, level_sq_err);
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "CPU Encoding time: " << time_span.count() <<" s (" << total_data/time_span.count() << "GB/s)\n";
    ofs << total_data/time_span.count() << ",";

    std::vector<uint8_t const *> const_streams;
    for (int i = 0; i < streams.size(); i++) const_streams.push_back(streams[i]);

    total_data = (n*decoding_num_bitplanes)/8/1e9;
    t1 = high_resolution_clock::now();
    auto level_decoded_data = encoder.progressive_decode(const_streams, n, exp, starting_bitplane, decoding_num_bitplanes, 0);
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "CPU Decoding time: " << time_span.count() <<" s (" << total_data/time_span.count() << "GB/s)\n";
    ofs << total_data/time_span.count() << ",";

    for (int i = 0; i < streams.size(); i++) delete [] streams[i];
    delete [] level_decoded_data;
  }

  //GPU-Warp
  {
    mgard_cuda::MDR::GroupedWarpEncoder<HandleType, T_data, T_bitplane, T_error, 
                                        NumGroupsPerWarpPerIter, NumWarpsPerTB, BinaryType, 
                                        DataEncodingAlgorithm, ErrorCollectingAlgorithm, 
                                        mgard_cuda::CUDA>encoder(handle);

    mgard_cuda::MDR::GroupedWarpDecoder<HandleType, T_data, T_bitplane, 
                                        NumGroupsPerWarpPerIter, NumWarpsPerTB, BinaryType, 
                                        DataDecodingAlgorithm, 
                                        mgard_cuda::CUDA> decoder(handle);

    mgard_cuda::Array<2, T_error> level_errors_work_array({encoding_num_bitplanes+1, MGARDm_NUM_SMs});
    mgard_cuda::SubArray<2, T_error> level_errors_work(level_errors_work_array);
    
    mgard_cuda::Array<2, T_bitplane> encoded_bitplanes_array({encoding_num_bitplanes, encoder.MaxBitplaneLength(n)});
    mgard_cuda::SubArray<2, T_bitplane> encoded_bitplanes_subarray(encoded_bitplanes_array);

    total_data = (n*encoding_num_bitplanes)/8/1e9;

    for (int r = 0; r < warmup; r++) {
      encoder.Execute(n, encoding_num_bitplanes, exp, 
                        v_subarray, encoded_bitplanes_subarray, level_errors, level_errors_work, 0);
    }

    handle.sync(0);
    t1 = high_resolution_clock::now();
    for (int r = 0; r < repeat; r++) {
      encoder.Execute(n, encoding_num_bitplanes, exp, 
                      v_subarray, encoded_bitplanes_subarray, level_errors, level_errors_work, 0);
    }
    handle.sync(0);
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1) / repeat;
    std::cout << "GPU Encoding time: " << time_span.count() <<" s (" << total_data/time_span.count() << "GB/s)\n";
    ofs << total_data/time_span.count() << ",";
    // mgard_cuda::PrintSubarray("encoded_bitplanes_subarray", encoded_bitplanes_subarray);

    total_data = (n*decoding_num_bitplanes)/8/1e9;
    
    
    for (int r = 0; r < warmup; r++) {
      decoder.Execute(n, starting_bitplane, decoding_num_bitplanes, exp, 
                      encoded_bitplanes_subarray, signs_subarray, v_subarray, 0);
    }

    handle.sync(0);
    t1 = high_resolution_clock::now();
    for (int r = 0; r < repeat; r++) {
      decoder.Execute(n, starting_bitplane, decoding_num_bitplanes, exp, 
                      encoded_bitplanes_subarray, signs_subarray, v_subarray, 0);
    }
    handle.sync(0);
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1) / repeat;
    std::cout << "GPU Decoding time: " << time_span.count() <<" s (" << total_data/time_span.count() << "GB/s)\n";
    ofs << total_data/time_span.count() << "\n";
  }

  // v2 = v2_array.getDataHost();  

  //  printf("Original data:\n");
  // for (int i = 0; i < n; i++) {
  //   printf("[%d]%f:\t", i, v[i]);
  //   // print_bits(v[i], sizeof(T_data)*8, false);
  //   printf("\n");
  //   if ((i+1) % (sizeof(T_bitplane) * 8) == 0) {
  //     printf("\n");
  //   }
  // }
  // printf("\n");
  // printf("Decoded data:\n");
  // for (int i = 0; i < n; i++) {
  //   printf("[%d]%f:\t", i, v2[i]);
  //   // print_bits(v2[i], sizeof(T_data)*8, false);
  //   printf("\n");
  //   if (i && i % (sizeof(T_bitplane) * 8) == 0) {
  //     printf("\n");
  //   }
  // }
  // printf("\n");

  bool check_correctness = false;
  if (check_correctness) {
    printf("Check: ");
    bool pass = true;

    double * errors = new double[encoding_num_bitplanes];
    for (int i = 0; i < encoding_num_bitplanes+1; i++) errors[i] = 0.0;

    if (BinaryType == NEGABINARY) exp += 2;
    for (int i = 0; i < n; i++) {
      T_data cur_data = v[i];
      bool sign = cur_data < 0 ? 1 : 0;
      T_data shifted_data = ldexp(cur_data, (int)encoding_num_bitplanes - (int)exp);
      T_fp fp_data;
      if (BinaryType == BINARY) {
        fp_data = (T_fp) fabs(shifted_data);
      } else if (BinaryType == NEGABINARY) {
        fp_data = mgard_cuda::binary2negabinary((T_sfp)shifted_data);
      }
      T_fp mask = ~((1u << sizeof(T_fp)*8 - decoding_num_bitplanes) - 1);
      // print_bits(mask, sizeof(T_fp)*8, false);
      // printf("\n");
      fp_data = fp_data & mask;
      T_data cur_data2;
      if (BinaryType == BINARY) {
        cur_data2 = ldexp((T_data)fp_data, - encoding_num_bitplanes + exp);
        cur_data2 = sign == 1 ? -cur_data2 : cur_data2;
      } else if (BinaryType == NEGABINARY) {
        cur_data2 = ldexp((T_data)mgard_cuda::negabinary2binary(fp_data), - encoding_num_bitplanes + exp);
        cur_data2 = decoding_num_bitplanes % 2 != 0 ? -cur_data2 : cur_data2;
      }

      if (fabs(cur_data2-v2[i]) > 1e-6) {
        // printf("%f %f %f\n", v[i], cur_data2, v2[i]);
        pass = false;
      }

      {
        T_fp fp_data = (T_fp) fabs(shifted_data);
        T_sfp fps_data = (T_sfp) shifted_data;
        T_fp ngb_data = mgard_cuda::binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(shifted_data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = shifted_data - fps_data;
        }

        // printf("fp: %u error: \n", fp_data);
        for(int bitplane_idx = 0; bitplane_idx < encoding_num_bitplanes; bitplane_idx++){
          uint64_t mask = (1 << bitplane_idx) - 1;
          T_error diff = 0;
          if (BinaryType == BINARY) {
            diff = (T_error) (fp_data & mask) + mantissa;
          } else if (BinaryType == NEGABINARY) {
            diff = (T_error) mgard_cuda::negabinary2binary(ngb_data & mask) + mantissa;
          }
          errors[encoding_num_bitplanes-bitplane_idx] += diff * diff;
          // printf("%f ", diff * diff);
        }
        errors[0] += shifted_data * shifted_data;
        // printf("%f \n", shifted_data * shifted_data);
      }
    }

    if (pass) printf("\e[32mpass\e[0m\n");
    else printf("\e[31mno pass\e[0m\n");

    // mgard_cuda::PrintSubarray("level_errors", level_errors);
    // printf("errors: ");
    // for (int i = 0; i < encoding_num_bitplanes+1; i++) {
    //   errors[i] = ldexp(errors[i], 2*(- (int)encoding_num_bitplanes + exp));
    //   printf("%f ", errors[i]);
    // }
    // printf("\n");
  }

  
  ofs.close();
  delete [] v;
  // mgard_cuda::cudaFreeHostHelper(v2);

}


template <mgard_cuda::OPTION BinaryType,
          mgard_cuda::OPTION DataEncodingAlgorithm,
          mgard_cuda::OPTION ErrorCollectingAlgorithm,
          mgard_cuda::OPTION DataDecodingAlgorithm>
void test_method() {
  typedef unsigned long long int uint64_t;

  int warmup = 0;
  int repeat = 1;

  for(mgard_cuda::LENGTH N = 1024; N <= 512*1024*1024 ; N *= 2) 
  // mgard_cuda::SIZE N = 200;
  // mgard_cuda::SIZE N = 512*1024*1024;
  { 

    // for debug
    // test<float, uint32_t, 1, 1,
    //       BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm>
    //       (N, 32, 10, warmup, repeat);
          
    // test<float, uint32_t, 32, 2,
    //       BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm>
    //       (N, 32, 32, warmup, repeat);
    // test<float, uint32_t, 16, 4,
    //       BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm>
    //       (N, 32, 32, warmup, repeat);
    // test<float, uint32_t, 8, 8,
    //       BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm>
    //       (N, 32, 32, warmup, repeat);

    test<float, uint32_t, 4, 16,
          BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm>
          (N, 32, 32, warmup, repeat);
    // test<float, uint32_t, 2, 32,
    //       BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm>
    //       (N, 32, 32, warmup, repeat);

    // test<uint32_t, uint64_t, METHOD>(N, 30, 10);

    // test<uint64_t, uint8_t, METHOD>(N, 50, 10);
    // test<uint64_t, uint16_t, METHOD>(N, 50, 10);
    // test<uint64_t, uint32_t, METHOD>(N, 50, 10);
    // test<uint64_t, uint64_t, METHOD>(N, 50, 10);

    // for (mgard_cuda::SIZE num_bitplanes = 1; num_bitplanes <= 32; num_bitplanes++) {
    //   test<float, uint32_t, 4, 16,
    //       BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm>
    //       (N, num_bitplanes, num_bitplanes, warmup, repeat);
    // }
  }
}

int main() {
  // test_method<BINARY, Warp_Bit_Transpose_Serial_All, Error_Collecting_Serial_All, Warp_Bit_Transpose_Serial_All>();
  // test_method<BINARY, Warp_Bit_Transpose_Parallel_B_Serial_b, Warp_Error_Collecting_Serial_All, Warp_Bit_Transpose_Parallel_B_Serial_b>();
  test_method<BINARY, Warp_Bit_Transpose_Parallel_B_Serial_b, Warp_Error_Collecting_Parallel_Bitplanes_Serial_Error, Warp_Bit_Transpose_Parallel_B_Serial_b>();
  // test_method<BINARY, Warp_Bit_Transpose_Parallel_B_Serial_b, Warp_Error_Collecting_Serial_Bitplanes_Atomic_Error, Warp_Bit_Transpose_Parallel_B_Serial_b>();
  // test_method<BINARY, Warp_Bit_Transpose_Parallel_B_Serial_b, Warp_Error_Collecting_Serial_Bitplanes_Reduce_Error, Warp_Bit_Transpose_Parallel_B_Serial_b>();
  // test_method<BINARY, Warp_Bit_Transpose_Serial_B_Atomic_b, Error_Collecting_Serial_All, Warp_Bit_Transpose_Serial_B_Atomic_b>();
  // test_method<BINARY, Warp_Bit_Transpose_Serial_B_Reduce_b, Error_Collecting_Serial_All, Warp_Bit_Transpose_Serial_B_Reduce_b>();
  // test_method<BINARY, Warp_Bit_Transpose_Serial_B_Ballot_b, Error_Collecting_Serial_All, Warp_Bit_Transpose_Serial_B_Ballot_b>();

  // test_method<NEGABINARY, Warp_Bit_Transpose_Serial_All, Error_Collecting_Disable, Warp_Bit_Transpose_Serial_All>();
  // test_method<NEGABINARY, Warp_Bit_Transpose_Parallel_B_Serial_b, Error_Collecting_Disable, Warp_Bit_Transpose_Parallel_B_Serial_b>();
  // test_method<NEGABINARY, Warp_Bit_Transpose_Serial_B_Atomic_b, Error_Collecting_Disable, Warp_Bit_Transpose_Serial_B_Atomic_b>();
  // test_method<NEGABINARY, Warp_Bit_Transpose_Serial_B_Reduce_b, Error_Collecting_Disable, Warp_Bit_Transpose_Serial_B_Reduce_b>();
  // test_method<NEGABINARY, Warp_Bit_Transpose_Serial_B_Ballot_b, Error_Collecting_Disable, Warp_Bit_Transpose_Serial_B_Ballot_b>();


  return 0;
}