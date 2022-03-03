#define MGARDX_COMPILE_CUDA

#include <bitset>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

#include "mgard-x/Hierarchy.h"
#include "mgard-x/RuntimeX/RuntimeX.h"

// #include "utils.hpp"
#include "mgard-x/MDR/Refactor/Refactor.hpp"

using namespace std;

template <class T, class Refactor>
void evaluate(const vector<T> &data, const vector<mgard_x::SIZE> &dims,
              int target_level, int num_bitplanes, Refactor &refactor) {
  printf("evaluate\n");
  struct timespec start, end;
  int err = 0;
  cout << "Start refactoring" << endl;
  err = clock_gettime(CLOCK_REALTIME, &start);
  refactor.refactor(data.data(), dims, target_level, num_bitplanes);
  err = clock_gettime(CLOCK_REALTIME, &end);
  cout << "Refactor time: "
       << (double)(end.tv_sec - start.tv_sec) +
              (double)(end.tv_nsec - start.tv_nsec) / (double)1000000000
       << "s" << endl;
}

// template <class T, class Decomposer, class Interleaver, class Encoder,
//           class Compressor, class ErrorCollector, class Writer>
// void test(string filename, const vector<mgard_x::SIZE> &dims, int target_level,
//           int num_bitplanes, Decomposer decomposer, Interleaver interleaver,
//           Encoder encoder, Compressor compressor, ErrorCollector collector,
//           Writer writer) {
//   auto refactor =
//       MDR::ComposedRefactor<T, Decomposer, Interleaver, Encoder,
//                                      Compressor, ErrorCollector, Writer>(
//           decomposer, interleaver, encoder, compressor, collector, writer);
//   size_t num_elements = 1;

//   FILE *pFile;
//   pFile = fopen(filename.c_str(), "rb");
//   for (int d = 0; d < dims.size(); d++)
//     num_elements *= dims[d];
//   vector<T> data(
//       num_elements); // MGARD::readfile<T>(filename.c_str(), num_elements);
//   fread(data.data(), 1, num_elements * sizeof(T), pFile);
//   fclose(pFile);
//   evaluate(data, dims, target_level, num_bitplanes, refactor);
// }

template <mgard_x::DIM D, class T_data, class T_bitplane,
          class Decomposer, class Interleaver, class Encoder, class Compressor,
          class ErrorCollector, class Writer>
void test2(string filename, const vector<mgard_x::SIZE> &dims, int target_level,
           int num_bitplanes, mgard_x::Hierarchy<D, T_data, mgard_x::CUDA> &hierarchy, Decomposer decomposer,
           Interleaver interleaver, Encoder encoder, Compressor compressor,
           ErrorCollector collector, Writer writer) {
  printf("test2\n");

  auto refactor =
      mgard_x::MDR::ComposedRefactor<D, T_data, T_bitplane,
                                     Decomposer, Interleaver, Encoder,
                                     Compressor, ErrorCollector, Writer>(
          hierarchy, decomposer, interleaver, encoder, compressor, collector,
          writer);
  size_t num_elements = 1;

  printf("loading file\n");
  FILE *pFile;
  pFile = fopen(filename.c_str(), "rb");
  for (int d = 0; d < dims.size(); d++)
    num_elements *= dims[d];
  vector<T_data> data(
      num_elements); // MGARD::readfile<T>(filename.c_str(), num_elements);
  fread(data.data(), 1, num_elements * sizeof(T_data), pFile);
  fclose(pFile);
  printf("done loading file\n");
  evaluate(data, dims, target_level, num_bitplanes, refactor);
}

int main(int argc, char **argv) {

  int argv_id = 1;
  string filename = string(argv[argv_id++]);
  int target_level = atoi(argv[argv_id++]);
  int num_bitplanes = atoi(argv[argv_id++]);
  if (num_bitplanes % 2 == 1) {
    num_bitplanes += 1;
    std::cout << "Change to " << num_bitplanes + 1
              << " bitplanes for simplicity of negabinary encoding"
              << std::endl;
  }
  int num_dims = atoi(argv[argv_id++]);
  vector<mgard_x::SIZE> dims(num_dims, 0);
  for (int i = 0; i < num_dims; i++) {
    dims[i] = atoi(argv[argv_id++]);
  }

  string metadata_file = "refactored_data/metadata.bin";
  vector<string> files;
  for (int i = 0; i <= target_level; i++) {
    string filename = "refactored_data/level_" + to_string(i) + ".bin";
    files.push_back(filename);
  }
  using T = float;
  using T_stream = uint32_t;
  using T_error = double;
  if (num_bitplanes > 32) {
    num_bitplanes = 32;
    std::cout << "Only less than 32 bitplanes are supported for "
                 "single-precision floating point"
              << std::endl;
  }
  const mgard_x::DIM D = 3;
  printf("dims: %u %u %u\n", dims[2], dims[1], dims[0]);

  mgard_x::Hierarchy<D, T, mgard_x::CUDA> hierarchy(dims, 0, target_level);

  // if (false) {
    auto decomposer = mgard_x::MDR::MGARDOrthoganalDecomposer<D, T>(hierarchy);

    auto interleaver = mgard_x::MDR::DirectInterleaver<D, T>(hierarchy);
    // auto interleaver = mgard_x::MDR::SFCInterleaver<T>();
    // auto interleaver = mgard_x::MDR::BlockedInterleaver<T>();

    auto encoder = mgard_x::MDR::GroupedBPEncoder<T, T_stream, T_error>();
    // auto encoder = mgard_x::MDR::GroupedBPEncoderGPU<D, T, T_stream>(handle);
    // auto encoder = mgard_x::MDR::NegaBinaryBPEncoder<D, T, T_stream>(handle);
    // auto encoder = mgard_x::MDR::PerBitBPEncoder<D, T, T_stream>(handle);
    // auto encoder = mgard_x::MDR::PerBitBPEncoderGPU<D, T, T_stream>(handle);
    // auto encoder = mgard_x::MDR::GroupedBPEncoderGPU<D, T, T_stream>(handle);

    auto compressor = mgard_x::MDR::DefaultLevelCompressor<T_stream>();
    // auto compressor = mgard_x::MDR::AdaptiveLevelCompressor(32);
    // auto compressor = mgard_x::MDR::NullLevelCompressor();

    auto collector = mgard_x::MDR::SquaredErrorCollector<T>();

    auto writer = mgard_x::MDR::ConcatLevelFileWriter(metadata_file, files);
    // auto writer = mgard_x::MDR::HPSSFileWriter(metadata_file, files, 2048,
    // 512 * 1024 * 1024);

    test2<D, T, T_stream>(
        filename, dims, target_level, num_bitplanes, hierarchy, decomposer,
        interleaver, encoder, compressor, collector, writer);

    // test2<T>(filename, dims, target_level, num_bitplanes, decomposer,
    //         interleaver, encoder, compressor, collector, writer);
  // }

  // if (true) {
  //   std::vector<mgard_x::Array<1, bool, mgard_x::CUDA>> level_signs;

    // auto decomposer = mgard_x::MDR::MGARDOrthoganalDecomposer<D, T>(hierarchy);
    // auto interleaver = mgard_x::MDR::DirectInterleaver<D, T>(hierarchy);
    // auto encoder = mgard_x::MDR::GroupedBPEncoder<T, T_stream, T_error>();
  //   auto encoder =
  //       mgard_m::MDR::GroupedWarpBPEncoder<D, T, T_stream, T_error>();
  //   auto compressor =
  //       mgard_m::MDR::DefaultLevelCompressor<D, T_stream>();
  //   auto collector = mgard_x::MDR::SquaredErrorCollector<T>();
  //   auto writer = mgard_x::MDR::ConcatLevelFileWriter(metadata_file, files);
  //   test2<D, T, T_stream>(
  //       filename, dims, target_level, num_bitplanes, hierarchy, decomposer,
  //       interleaver, encoder, compressor, collector, writer);
  // }

  return 0;
}

#undef MGARDX_COMPILE_CUDA