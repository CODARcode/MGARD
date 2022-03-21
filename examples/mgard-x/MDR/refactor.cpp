#include <bitset>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

#include "mgard/mdr.hpp"

using namespace std;

template <class T, class Refactor>
void evaluate(const vector<T> &data, const vector<uint32_t> &dims,
              int target_level, int num_bitplanes, Refactor refactor) {
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

template <class T, class Decomposer, class Interleaver, class Encoder,
          class Compressor, class ErrorCollector, class Writer>
void test(string filename, const vector<uint32_t> &dims, int target_level,
          int num_bitplanes, Decomposer decomposer, Interleaver interleaver,
          Encoder encoder, Compressor compressor, ErrorCollector collector,
          Writer writer) {
  auto refactor = MDR::ComposedRefactor<T, Decomposer, Interleaver, Encoder,
                                        Compressor, ErrorCollector, Writer>(
      decomposer, interleaver, encoder, compressor, collector, writer);
  size_t num_elements = 0;
  auto data = MGARD::readfile<T>(filename.c_str(), num_elements);
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
  vector<uint32_t> dims(num_dims, 0);
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
  if (num_bitplanes > 32) {
    num_bitplanes = 32;
    std::cout << "Only less than 32 bitplanes are supported for "
                 "single-precision floating point"
              << std::endl;
  }
  auto decomposer = MDR::MGARDOrthoganalDecomposer<T>();
  // auto decomposer = MDR::MGARDHierarchicalDecomposer<T>();
  auto interleaver = MDR::DirectInterleaver<T>();
  // auto interleaver = MDR::SFCInterleaver<T>();
  // auto interleaver = MDR::BlockedInterleaver<T>();
  auto encoder = MDR::GroupedBPEncoder<T, T_stream>();
  // auto encoder = MDR::NegaBinaryBPEncoder<T, T_stream>();
  // auto encoder = MDR::PerBitBPEncoder<T, T_stream>();
  // auto compressor = MDR::DefaultLevelCompressor();
  auto compressor = MDR::AdaptiveLevelCompressor(32);
  // auto compressor = MDR::NullLevelCompressor();
  auto collector = MDR::SquaredErrorCollector<T>();
  auto writer = MDR::ConcatLevelFileWriter(metadata_file, files);
  // auto writer = MDR::HPSSFileWriter(metadata_file, files, 2048, 512 * 1024 *
  // 1024);

  test<T>(filename, dims, target_level, num_bitplanes, decomposer, interleaver,
          encoder, compressor, collector, writer);
  return 0;
}