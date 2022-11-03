#include "mgard/mdr_x_lowlevel.hpp"
#include "mgard/mgard-x/Utilities/ErrorCalculator.h"
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

template <class T_data>
void print_statistics(const T_data *data_ori, const T_data *data_dec,
                      size_t data_size) {
  double max_val = data_ori[0];
  double min_val = data_ori[0];
  double max_abs = fabs(data_ori[0]);
  for (int i = 0; i < data_size; i++) {
    if (data_ori[i] > max_val)
      max_val = data_ori[i];
    if (data_ori[i] < min_val)
      min_val = data_ori[i];
    if (fabs(data_ori[i]) > max_abs)
      max_abs = fabs(data_ori[i]);
  }
  double max_err = 0;
  int pos = 0;
  double mse = 0;
  for (int i = 0; i < data_size; i++) {
    double err = data_ori[i] - data_dec[i];
    mse += err * err;
    if (fabs(err) > max_err) {
      pos = i;
      max_err = fabs(err);
    }
  }
  mse /= data_size;
  double psnr = 20 * log10((max_val - min_val) / sqrt(mse));
  cout << "Max value = " << max_val << ", min value = " << min_val << endl;
  cout << "Max error = " << max_err << ", pos = " << pos << endl;
  cout << "MSE = " << mse << ", PSNR = " << psnr << endl;
  cout << "L2 error = "
       << mgard_x::L_2_error({(mgard_x::SIZE)data_size}, data_ori, data_dec,
                             mgard_x::error_bound_type::ABS, 0)
       << endl;
  cout << "L_inf error = "
       << mgard_x::L_inf_error(data_size, data_ori, data_dec,
                               mgard_x::error_bound_type::ABS)
       << endl;
}

template <mgard_x::DIM D, class T_data, typename DeviceType>
void test(string filename, int num_bitplanes,
          mgard_x::Hierarchy<D, T_data, DeviceType> &hierarchy,
          std::string metadata_file, std::vector<std::string> files,
          const vector<double> &tolerance, double s) {
    
  size_t num_elements = 1;

  printf("loading file\n");
  FILE *pFile;
  pFile = fopen(filename.c_str(), "rb");
  for (int d = 0; d < D; d++)
    num_elements *= hierarchy.level_shape(hierarchy.l_target(), d);
  vector<T_data> data(num_elements);
  fread(data.data(), 1, num_elements * sizeof(T_data), pFile);
  fclose(pFile);
  printf("done loading file\n");
  mgard_x::Array<D, T_data, DeviceType> input_array(hierarchy.level_shape(hierarchy.l_target()));
  input_array.load(data.data());
  mgard_x::log::level |= mgard_x::log::TIME;
  // mgard_x::Timer timer;
  // timer.start();

  mgard_x::Config config;
  mgard_x::MDR::MDRData<DeviceType> mdr_data;
  mgard_x::MDR::MDRMetaData mdr_metadata;
  {
    auto refactor =
        mgard_x::MDR::ComposedRefactor<D, T_data, DeviceType>(
            hierarchy, config, metadata_file, files);
    refactor.Refactor(input_array, mdr_metadata, mdr_data, 0);
  }

  {
    auto reconstructor = mgard_x::MDR::ComposedReconstructor<
        D, T_data, DeviceType>(hierarchy, config, metadata_file, files);

    

    // reconstructor.load_metadata();

    mdr_metadata.InitializeForReconstruction();
    mgard_x::Array<D, T_data, DeviceType> reconstructed_data(hierarchy.level_shape(hierarchy.l_target()));
    reconstructed_data.memset(0);
    for (int i = 0; i < tolerance.size(); i++) {
      mgard_x::log::level |= mgard_x::log::TIME;
      // mgard_x::Timer timer;
      // timer.start();
      reconstructor.GenerateRequest(mdr_metadata, tolerance[i], s);
      mdr_metadata.PrintStatus();
      mdr_metadata.LoadBitplans();
      // reconstructor.progressive_reconstruct(tolerance[i], s, reconstructed_data);
      reconstructor.ProgressiveReconstruct(mdr_metadata, mdr_data, reconstructed_data, 0);
      // timer.end();
      // timer.print("Reconstruct");
      auto dims = reconstructor.get_dimensions();
      size_t size = 1;
      for (int d = 0; d < D; d++) {
        size *= hierarchy.level_shape(hierarchy.l_target(), d);
      }
      print_statistics(data.data(), reconstructed_data.hostCopy(), size);
    }
  }

  // timer.end();
  // timer.print("Refactor");
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

  int num_tolerance = atoi(argv[argv_id++]);
  vector<double> tolerance(num_tolerance, 0);
  for (int i = 0; i < num_tolerance; i++) {
    tolerance[i] = atof(argv[argv_id++]);
  }
  double s = atof(argv[argv_id++]);



  string metadata_file = "refactored_data/metadata.bin";
  vector<string> files;
  for (int i = 0; i <= target_level; i++) {
    string filename = "refactored_data/level_" + to_string(i) + ".bin";
    files.push_back(filename);
  }
  using T = float;
  using T_stream = uint32_t;
  using T_error = double;
  using DeviceType = mgard_x::CUDA;
  if (num_bitplanes > 32) {
    num_bitplanes = 32;
    std::cout << "Only less than 32 bitplanes are supported for "
                 "single-precision floating point"
              << std::endl;
  }
  const mgard_x::DIM D = 3;
  printf("dims: %lu %lu %lu\n", dims[0], dims[1], dims[2]);

  mgard_x::log::level |= mgard_x::log::INFO;
  mgard_x::Config config;
  config.max_larget_level = target_level;
  mgard_x::Hierarchy<D, T, DeviceType> hierarchy(dims, config);

  // if (false) {
  auto decomposer =
      mgard_x::MDR::MGARDOrthoganalDecomposer<D, T, DeviceType>(hierarchy);

  auto interleaver =
      mgard_x::MDR::DirectInterleaver<D, T, DeviceType>(hierarchy);
  // auto interleaver = mgard_x::MDR::SFCInterleaver<T>();
  // auto interleaver = mgard_x::MDR::BlockedInterleaver<T>();

  auto encoder = mgard_x::MDR::GroupedBPEncoder<D, T, T_stream, T_error,
  DeviceType>(hierarchy);
  // auto encoder =
  //     mgard_x::MDR::GroupedWarpBPEncoder<D, T, T_stream, T_error, DeviceType>(
  //         hierarchy);

  auto compressor =
      mgard_x::MDR::DefaultLevelCompressor<T_stream, DeviceType>(hierarchy.total_num_elems(), 8192, 20480, 1.0);
  // auto compressor = mgard_x::MDR::AdaptiveLevelCompressor(32);
  // auto compressor = mgard_x::MDR::NullLevelCompressor();

  // auto collector = mgard_x::MDR::SquaredErrorCollector<T>();
  auto collector = mgard_x::MDR::MaxErrorCollector<T>();

  auto writer = mgard_x::MDR::ConcatLevelFileWriter(metadata_file, files);
  // auto writer = mgard_x::MDR::HPSSFileWriter(metadata_file, files, 2048,
  // 512 * 1024 * 1024);

  test<D, T, DeviceType>(
      filename, num_bitplanes, hierarchy, 
      metadata_file, files, tolerance, s);

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