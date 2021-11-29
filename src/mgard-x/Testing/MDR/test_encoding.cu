#include "cuda/CommonInternal.h"
#include "cuda/Functor.h"
#include "cuda/AutoTuners/AutoTuner.h"
#include "cuda/Task.h"
#include "cuda/DeviceAdapters/DeviceAdapterCuda.h"
#include "cuda/MDR/BitplaneEncoder/GroupedBPEncoderGPU.hpp"
#include <mma.h>
#include <iostream>
#include <fstream>
using namespace nvcuda;

#include <chrono>
using namespace std::chrono;
template <typename T>
MGARDX_CONT_EXEC void
print_bits(T v, int num_bits, bool reverse = false) {
  for (int j = 0; j < num_bits; j++) {
    if (!reverse) printf("%u", (v >> sizeof(T)*8-1-j) & 1u);
    else printf("%u", (v >> j) & 1u);
  }
}


template <typename T_data, typename T_bitplane, mgard_x::OPTION BinaryType, mgard_x::OPTION DataEncodingAlgorithm, mgard_x::OPTION ErrorCollectingAlgorithm, mgard_x::OPTION DataDecodingAlgorithm>
void test(mgard_x::SIZE n, 
          mgard_x::SIZE num_batches_per_TB, 
          mgard_x::SIZE encoding_num_bitplanes, 
          mgard_x::SIZE decoding_num_bitplanes){
  
  bool check_correctness = true;

  double total_data = (n*encoding_num_bitplanes)/8/1e9;
  printf("T_data: %d, T_bitplane: %d, BinaryType: %d, DataEncodingAlgorithm: %d, ErrorCollectingAlgorithm: %d, DataDecodingAlgorithm: %d, Total data: %.2e GB.\n", 
          sizeof(T_data)*8, sizeof(T_bitplane)*8, BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm, total_data);

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
  path.erase (path.end()-std::string("test_encoding.cu").length(), path.end());
  ofs.open (path + "/encoding_pref_results/" + filename, std::ofstream::out | std::ofstream::app);
  if(!ofs) std::cout<<"Writing to file failed"<<std::endl;

  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;

  // printf("Generating data\n");
  T_data * v = new T_data[n];
  for (int i = 0; i < n; i++) {
    v[i] = rand();// % 10000000;
    for (int j = 0; j < sizeof(T_data)*8; j++) {
      // T_data bit = rand() % 2;
      // v[i] += bit << j;
    }
  }

  // printf("Done Generating\n");

 


  // T_bitplane * bitplane = new T_bitplane[num_blocks*sizeof(T_data)*8];

  // mgard_x::Handle<1, float> handle;

  

  // mgard_x::Array<1, T_bitplane> bitplane_array({(mgard_x::SIZE)num_blocks*(int)sizeof(T_data)*8});
  // mgard_x::SubArray<1, T_bitplane> bitplane_subarray(bitplane_array);


  

  // printf("Starting encoding\n");
  using HandleType = mgard_x::Handle<1, float>;
  using T_error = double;
  HandleType handle({5});

  

  const mgard_x::SIZE num_elems_per_TB = sizeof(T_bitplane) * 8 * num_batches_per_TB;
  const mgard_x::SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
  mgard_x::SIZE num_TB = (n-1)/num_elems_per_TB+1;
  mgard_x::SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_TB;
  printf("n: %u, num_batches_per_TB: %u, num_elems_per_TB: %u, num_TB: %u\n", n, num_batches_per_TB, num_elems_per_TB, num_TB);


  mgard_x::MDR::GroupedEncoder<T_data, T_bitplane, T_error, BINARY, DataEncodingAlgorithm, ErrorCollectingAlgorithm, mgard_x::CUDA> encoder;

  mgard_x::Array<1, T_data, mgard_x::CUDA> v_array({n});
  v_array.loadData(v);
  mgard_x::SubArray<1, T_data, mgard_x::CUDA> v_subarray(v_array);
  // mgard_x::PrintSubarray("v_subarray", v_subarray);

  mgard_x::Array<2, T_error, mgard_x::CUDA> level_errors_work_array({encoding_num_bitplanes+1, num_TB});
  mgard_x::SubArray<2, T_error, mgard_x::CUDA> level_errors_work(level_errors_work_array);
  mgard_x::Array<1, T_error, mgard_x::CUDA> level_errors_array({encoding_num_bitplanes+1});
  mgard_x::SubArray<1, T_error, mgard_x::CUDA> level_errors(level_errors_array);

  mgard_x::Array<2, T_bitplane, mgard_x::CUDA> encoded_bitplanes_array({encoding_num_bitplanes, encoder.MaxBitplaneLength(n)});
  mgard_x::SubArray<2, T_bitplane, mgard_x::CUDA> encoded_bitplanes_subarray(encoded_bitplanes_array);

  mgard_x::Array<1, T_data, mgard_x::CUDA> result_array({1});
  mgard_x::SubArray<1, T_data, mgard_x::CUDA> result(result_array);
  mgard_x::DeviceCollective<mgard_x::CUDA> deviceReduce;
  deviceReduce.AbsMax(v_subarray.getShape(0), v_subarray, result, 0);
  mgard_x::DeviceRuntime<mgard_x::CUDA>().SyncQueue(0);
  T_data level_max_error = *(result_array.getDataHost());
  int exp = 0;
  frexp(level_max_error, &exp);


  mgard_x::DeviceRuntime<mgard_x::CUDA>().SyncQueue(0);
  t1 = high_resolution_clock::now();

  encoder.Execute(n, num_batches_per_TB, encoding_num_bitplanes, exp, 
                  v_subarray, encoded_bitplanes_subarray, level_errors, level_errors_work, 0);

  // mgard_x::EncodingTest<mgard_x::Handle<1, float>, T_data, T_bitplane, METHOD, mgard_x::CUDA>(handle).Execute(n, v_subarray, bitplane_subarray, encoding_num_bitplanes, 0);
  mgard_x::DeviceRuntime<mgard_x::CUDA>().SyncQueue(0);
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);

  // mgard_x::PrintSubarray("encoded_bitplanes_subarray", encoded_bitplanes_subarray);
  
  // printf("Done encoding\n");
  
  std::cout << "Encoding time: " << time_span.count() <<" s (" << total_data/time_span.count() << "GB/s)\n";
  ofs << total_data/time_span.count() << ",";

  T_bitplane * bitplanes = encoded_bitplanes_array.getDataHost();
  // bool pass = true;
  // if (check_correctness){
  //   printf("Encoding: ");
  //   for (int i = 0; i < n; i++) {
  //     for (int j = 0; j < encoding_num_bitplanes; j++) {
  //       uint8_t bit1 = (v[i] >> sizeof(T_data)*8-1-j) & 1u;
  //       uint8_t bit2 = (bitplanes[j+i/(sizeof(T_bitplane)*8)*sizeof(T_data)*8] >> (sizeof(T_bitplane)*8-1-i%(sizeof(T_bitplane)*8)))& 1u;
  //       if (bit1 != bit2) {
  //         pass = false;
  //         // printf("\e[31m%u\e[0m", bit1);
  //       } else {
  //         // printf("\e[32m%u\e[0m", bit1);
  //       }
  //     }
  //     // printf("\n");
  //   }

  //   if (pass) printf("\e[32mpass\e[0m\n");
  //   else printf("\e[31mno pass\e[0m\n");
  // }

  

  // printf("Bitplane: %u\n", bitplane_max_length_total * encoding_num_bitplanes);
  // for (int i = 0; i < bitplane_max_length_total * encoding_num_bitplanes; i++) {
  //   printf("[%d]%llu:\t", i, bitplanes[i]);
  //   print_bits(bitplanes[i], sizeof(T_bitplane)*8, false);
  //   printf("\n");
  //   if ((i+1) % (sizeof(T_data) * 8) == 0) {
  //     printf("\n");
  //   }
  // }
  // printf("\n");

  total_data = (n*decoding_num_bitplanes)/8/1e9;

  mgard_x::SIZE starting_bitplane = 0;
  mgard_x::Array<1, bool, mgard_x::CUDA> signs_array({n});
  mgard_x::SubArray<1, bool, mgard_x::CUDA> signs_subarray(signs_array);

  mgard_x::Array<1, T_data, mgard_x::CUDA> v2_array({n});
  mgard_x::SubArray<1, T_data, mgard_x::CUDA> v2_subarray(v2_array);

  mgard_x::DeviceRuntime<mgard_x::CUDA>().SyncQueue(0);
  t1 = high_resolution_clock::now();

  mgard_x::MDR::GroupedDecoder<T_data, T_bitplane, BinaryType, 
                  DataDecodingAlgorithm, mgard_x::CUDA>().
                  Execute(n, num_batches_per_TB, starting_bitplane, decoding_num_bitplanes, exp, 
                  encoded_bitplanes_subarray, signs_subarray, v2_subarray, 0);


  // mgard_x::DecodingTest<mgard_x::Handle<1, float>, T_bitplane, T_data, METHOD, mgard_x::CUDA>(handle).Execute(n, bitplane_subarray, v2_subarray, decoding_num_bitplanes, 0);
  mgard_x::DeviceRuntime<mgard_x::CUDA>().SyncQueue(0);
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "Decoding time: " << time_span.count() <<" s (" << total_data/time_span.count() << "GB/s)\n";
  ofs << total_data/time_span.count() << "\n";
  T_data * v2 = v2_array.getDataHost();


  using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;

  

  // printf("Original data:\n");
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



  // bool pass;
  // if (check_correctness) {
  //   printf("Decoding: ");
  //   pass = true;
  //   for (int i = 0; i < n; i++) {
  //     for (int j = 0; j < decoding_num_bitplanes; j++) {
  //       uint8_t bit1 = (v[i] >> sizeof(T_data)*8-1-j) & 1u;
  //       uint8_t bit2 = (v2[i] >> decoding_num_bitplanes-1-j) & 1u;
  //       if (bit1 != bit2) {
  //         pass = false;
  //       }
  //     }
  //   }

  //   if (pass) printf("\e[32mpass\e[0m\n");
  //   else printf("\e[31mno pass\e[0m\n");
  //   ofs.close();
  // }
}


template <mgard_x::OPTION BinaryType,
          mgard_x::OPTION DataEncodingAlgorithm,
          mgard_x::OPTION ErrorCollectingAlgorithm,
          mgard_x::OPTION DataDecodingAlgorithm>
void test_method() {
  typedef unsigned long long int uint64_t;

  // for(mgard_x::SIZE N = 1; N <= 16*1024*1024 ; N *= 2) 
  mgard_x::SIZE N = 8*1024*1024;
  for(mgard_x::SIZE num_batches_per_TB = 1; num_batches_per_TB <= 1; num_batches_per_TB *= 2)
  { 

    // test<uint32_t, uint8_t, METHOD>(N, 30, 10);
    // test<uint32_t, uint16_t, METHOD>(N, 30, 10);
    test<float, uint32_t, BinaryType, DataEncodingAlgorithm, ErrorCollectingAlgorithm, DataDecodingAlgorithm>(N, num_batches_per_TB, 32, 32);
    // test<uint32_t, uint64_t, METHOD>(N, 30, 10);

    // test<uint64_t, uint8_t, METHOD>(N, 50, 10);
    // test<uint64_t, uint16_t, METHOD>(N, 50, 10);
    // test<uint64_t, uint32_t, METHOD>(N, 50, 10);
    // test<uint64_t, uint64_t, METHOD>(N, 50, 10);
  }
}

int main() {
  test_method<BINARY, Bit_Transpose_Serial_All, Error_Collecting_Disable, Bit_Transpose_Serial_All>();
  // test_method<BINARY, Bit_Transpose_Parallel_B_Serial_b, Error_Collecting_Disable, Bit_Transpose_Parallel_B_Serial_b>();
  // test_method<BINARY, Bit_Transpose_Parallel_B_Atomic_b, Error_Collecting_Disable, Bit_Transpose_Parallel_B_Atomic_b>();
  // test_method<BINARY, Bit_Transpose_Parallel_B_Reduce_b, Error_Collecting_Disable, Bit_Transpose_Parallel_B_Reduce_b>();
  // test_method<BINARY, Bit_Transpose_Parallel_B_Ballot_b, Error_Collecting_Disable, Bit_Transpose_Parallel_B_Ballot_b>();
  // test_method<1>();
  // test_method<2>();
  // test_method<0>();
  // test_method<4>();

  return 0;
}