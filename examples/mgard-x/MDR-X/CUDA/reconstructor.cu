#include "mgard/mdr_x.hpp"
#include "mgard/mgard-x/Utilities/ErrorCalculator.h"
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

template <typename Type>
std::vector<Type> readfile(const char *file, size_t &num) {
  std::ifstream fin(file, std::ios::binary);
  if (!fin) {
    std::cout << " Error, Couldn't find the file"
              << "\n";
    return std::vector<Type>();
  }
  fin.seekg(0, std::ios::end);
  const size_t num_elements = fin.tellg() / sizeof(Type);
  fin.seekg(0, std::ios::beg);
  auto data = std::vector<Type>(num_elements);
  fin.read(reinterpret_cast<char *>(&data[0]), num_elements * sizeof(Type));
  fin.close();
  num = num_elements;
  return data;
}

template <class T>
void print_statistics(const T *data_ori, const T *data_dec, size_t data_size) {
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

template <class T, class Reconstructor>
void evaluate(const vector<T> &data, const vector<double> &tolerance,
              Reconstructor reconstructor) {
  struct timespec start, end;
  for (int i = 0; i < tolerance.size(); i++) {
    cout << "Start reconstruction" << endl;
    clock_gettime(CLOCK_REALTIME, &start);
    auto reconstructed_data =
        reconstructor.progressive_reconstruct(tolerance[i]);
    clock_gettime(CLOCK_REALTIME, &end);
    cout << "Reconstruct time: "
         << (double)(end.tv_sec - start.tv_sec) +
                (double)(end.tv_nsec - start.tv_nsec) / (double)1000000000
         << "s" << endl;
    auto dims = reconstructor.get_dimensions();
    size_t size = 1;
    for (int i = 0; i < dims.size(); i++) {
      size *= dims[i];
    }
    print_statistics(data.data(), reconstructed_data, size);
  }
}

template <mgard_x::DIM D, class T, class T_stream, typename DeviceType,
          class Decomposer, class Interleaver, class Encoder, class Compressor,
          class ErrorEstimator, class SizeInterpreter, class Retriever>
void test(string filename, const vector<double> &tolerance,
          mgard_x::Hierarchy<D, T, DeviceType> &hierarchy,
          Decomposer decomposer, Interleaver interleaver, Encoder encoder,
          Compressor compressor, ErrorEstimator estimator,
          SizeInterpreter interpreter, Retriever retriever) {
  auto reconstructor = mgard_x::MDR::ComposedReconstructor<
      D, T, T_stream, Decomposer, Interleaver, Encoder, Compressor,
      SizeInterpreter, ErrorEstimator, Retriever, DeviceType>(
      hierarchy, decomposer, interleaver, encoder, compressor, interpreter,
      retriever);
  cout << "loading metadata" << endl;
  reconstructor.load_metadata();

  size_t num_elements = 0;
  auto data = readfile<T>(filename.c_str(), num_elements);
  evaluate(data, tolerance, reconstructor);
}

int main(int argc, char **argv) {

  int argv_id = 1;
  string filename = string(argv[argv_id++]);
  int error_mode = atoi(argv[argv_id++]);
  int num_tolerance = atoi(argv[argv_id++]);
  vector<double> tolerance(num_tolerance, 0);
  for (int i = 0; i < num_tolerance; i++) {
    tolerance[i] = atof(argv[argv_id++]);
  }
  double s = atof(argv[argv_id++]);

  string metadata_file = "refactored_data/metadata.bin";
  int num_levels = 0;
  int num_dims = 0;
  vector<mgard_x::SIZE> dims;
  {
    // metadata interpreter, otherwise information needs to be provided
    size_t num_bytes = 0;
    auto metadata = readfile<uint8_t>(metadata_file.c_str(), num_bytes);
    assert(num_bytes > num_dims * sizeof(mgard_x::SIZE) + 2);
    num_dims = metadata[0];
    mgard_x::SIZE *dim = (mgard_x::SIZE *)&(metadata[1]);
    printf("dim: ");
    for (int i = 0; i < num_dims; i++) {
      dims.push_back(dim[i]);
      printf("%u ", dim[i]);
    }
    printf("\n");
    num_levels = metadata[num_dims * sizeof(mgard_x::SIZE) + 1];
    cout << "number of dimension = " << num_dims
         << ", number of levels = " << num_levels << endl;
  }
  vector<string> files;
  for (int i = 0; i < num_levels; i++) {
    string filename = "refactored_data/level_" + to_string(i) + ".bin";
    files.push_back(filename);
  }

  using T = float;
  using T_stream = uint32_t;
  using T_error = double;
  using DeviceType = mgard_x::CUDA;

  const mgard_x::DIM D = 3;
  mgard_x::Hierarchy<D, T, DeviceType> hierarchy(dims, 0, num_levels - 1);

  auto decomposer =
      mgard_x::MDR::MGARDOrthoganalDecomposer<D, T, DeviceType>(hierarchy);
  auto interleaver =
      mgard_x::MDR::DirectInterleaver<D, T, DeviceType>(hierarchy);
  // auto encoder = mgard_x::MDR::GroupedBPEncoder<T, T_stream, T_error,
  // DeviceType>();
  auto encoder =
      mgard_x::MDR::GroupedWarpBPEncoder<T, T_stream, T_error, DeviceType>();
  auto compressor =
      mgard_x::MDR::DefaultLevelCompressor<T_stream, DeviceType>();
  auto retriever = mgard_x::MDR::ConcatLevelFileRetriever(metadata_file, files);

  switch (error_mode) {
  case 1: {
    auto estimator =
        mgard_x::MDR::SNormErrorEstimator<T>(num_dims, num_levels - 1, s);
    // auto interpreter =
    // mgard_x::MDR::SignExcludeGreedyBasedSizeInterpreter<mgard_x::MDR::SNormErrorEstimator<T>>(estimator);
    // auto interpreter =
    // mgard_x::MDR::NegaBinaryGreedyBasedSizeInterpreter<mgard_x::MDR::SNormErrorEstimator<T>>(estimator);

    auto interpreter = mgard_x::MDR::RoundRobinSizeInterpreter<
        mgard_x::MDR::SNormErrorEstimator<T>>(estimator);
    // auto interpreter =
    // mgard_x::MDR::InorderSizeInterpreter<mgard_x::MDR::SNormErrorEstimator<T>>(estimator);
    // auto estimator = mgard_x::MDR::L2ErrorEstimator_HB<T>(num_dims,
    // num_levels - 1); auto interpreter =
    // mgard_x::MDR::SignExcludeGreedyBasedSizeInterpreter<mgard_x::MDR::L2ErrorEstimator_HB<T>>(estimator);
    test<D, T, T_stream, DeviceType>(filename, tolerance, hierarchy, decomposer,
                                     interleaver, encoder, compressor,
                                     estimator, interpreter, retriever);
    break;
  }
  default: {
    auto estimator = mgard_x::MDR::MaxErrorEstimatorOB<T>(num_dims);
    auto interpreter = mgard_x::MDR::SignExcludeGreedyBasedSizeInterpreter<
        mgard_x::MDR::MaxErrorEstimatorOB<T>>(estimator);
    // auto interpreter =
    // MDR::RoundRobinSizeInterpreter<MDR::MaxErrorEstimatorOB<T>>(estimator);
    // auto interpreter =
    // MDR::InorderSizeInterpreter<MDR::MaxErrorEstimatorOB<T>>(estimator);
    // auto estimator = MDR::MaxErrorEstimatorHB<T>();
    // auto interpreter =
    // MDR::SignExcludeGreedyBasedSizeInterpreter<MDR::MaxErrorEstimatorHB<T>>(estimator);
    test<D, T, T_stream, DeviceType>(filename, tolerance, hierarchy, decomposer,
                                     interleaver, encoder, compressor,
                                     estimator, interpreter, retriever);
  }
  }

  return 0;
}