/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <chrono>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compress.hpp"
// #include "compress_cuda.hpp"
#include "mgard-x/Utilities/ErrorCalculator.h"

using namespace std::chrono;

void print_usage_message(std::string error) {
  if (error.compare("") != 0) {
    std::cout << mgard_x::log::log_err << error << std::endl;
  }
  printf("Options\n\
\t -z: compress data\n\
\t\t -i <path to data file to be compressed>\n\
\t\t -c <path to compressed file>\n\
\t\t -t <s|d>: data type (s: single; d:double)\n\
\t\t -n <ndim>: total number of dimensions\n\
\t\t\t [dim1]: slowest dimention\n\
\t\t\t [dim2]: 2nd slowest dimention\n\
\t\t\t  ...\n\
\t\t\t [dimN]: fastest dimention\n\
\t\t -m <abs|rel>: error bound mode (abs: abolute; rel: relative)\n\
\t\t -e <error>: error bound\n\
\t\t -s <smoothness>: smoothness parameter\n\
\t\t -v: enable verbose (show timing and statistics)\n\
\n\
\t -x: decompress data\n\
\t\t -c <path to data file to be decompressed>\n\
\t\t -d <path to decompressed file>\n\
\t\t -t <s|d>: data type (s: single; d:double)\n\
\t\t -n <ndim>: total number of dimensions\n\
\t\t\t [dim1]: slowest dimention\n\
\t\t\t [dim2]: 2nd slowest dimention\n\
\t\t\t  ...\n\
\t\t\t [dimN]: fastest dimention\n\
\t\t -m <abs|rel>: error bound mode (abs: abolute; rel: relative)\n\
\t\t -e <error>: error bound\n\
\t\t -s <smoothness>: smoothness parameter\n\
\t\t -v: enable verbose (show timing and statistics)\n\
");
  exit(0);
}
template <typename T>
int launch_decompress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                      const char *input_file, const char *output_file,
                      std::vector<mgard_x::SIZE> shape, T tol, T s,
                      enum mgard_x::error_bound_type mode, bool verbose);
template <typename T>
int launch_compress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    const char *input_file, const char *output_file,
                    std::vector<mgard_x::SIZE> shape, T tol, T s,
                    enum mgard_x::error_bound_type mode, bool verbose);

bool has_arg(int argc, char *argv[], std::string option) {
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return true;
    }
  }
  return false;
}

bool require_arg(int argc, char *argv[], std::string option) {
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return true;
    }
  }
  print_usage_message("missing option: " + option + ".");
  return false;
}

std::string get_arg(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    for (int i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        return std::string(argv[i + 1]);
      }
    }
  }
  return std::string("");
}

int get_arg_int(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      int d = std::stoi(arg);
      return d;
    } catch (std::invalid_argument const &e) {
      print_usage_message("illegal argument for option " + option + ".");
      return 0;
    }
  }
  return 0;
}

std::vector<mgard_x::SIZE> get_arg_dims(int argc, char *argv[],
                                        std::string option) {
  std::vector<mgard_x::SIZE> shape;
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int arg_idx = 0, i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
        arg_idx = i + 1;
      }
    }
    try {
      int d = std::stoi(arg);
      for (int i = 0; i < d; i++) {
        shape.push_back(std::stoi(argv[arg_idx + 1 + i]));
      }
      return shape;
    } catch (std::invalid_argument const &e) {
      print_usage_message("illegal argument for option " + option + ".");
      return shape;
    }
  }
  return shape;
}

double get_arg_double(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      double d = std::stod(arg);
      return d;
    } catch (std::invalid_argument const &e) {
      print_usage_message("illegal argument for option " + option + ".");
    }
  }
  return 0;
}

template <typename T> void min_max(size_t n, T *in_buff) {
  T min = std::numeric_limits<T>::infinity();
  T max = 0;
  for (size_t i = 0; i < n; i++) {
    if (min > in_buff[i])
      min = in_buff[i];
    if (max < in_buff[i])
      max = in_buff[i];
  }
  printf("Min: %f, Max: %f\n", min, max);
}

template <typename T> size_t readfile(const char *input_file, T *&in_buff) {
  fprintf(stdout, "Loading file: %s\n", input_file);
  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    fputs("File error", stderr);
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff = (T *)malloc(lSize);
  lSize = fread(in_buff, 1, lSize, pFile);
  fclose(pFile);
  // min_max(lSize/sizeof(T), in_buff);
  return lSize;
}

template <typename T>
void writefile(const char *output_file, size_t num_bytes, T *out_buff) {
  FILE *file = fopen(output_file, "w");
  fwrite(out_buff, 1, num_bytes, file);
  fclose(file);
}

template <typename T>
void print_statistics(double s, enum mgard_x::error_bound_type mode,
                      std::vector<mgard_x::SIZE> shape, T *original_data,
                      T *decompressed_data) {

  mgard_x::SIZE n = 1;
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    n *= shape[d];

  std::cout << std::scientific;
  if (s == std::numeric_limits<T>::infinity()) {
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info << "Absoluate L_inf error: "
                << mgard_x::L_inf_error(n, original_data, decompressed_data,
                                        mode)
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info << "Relative L_inf error: "
                << mgard_x::L_inf_error(n, original_data, decompressed_data,
                                        mode)
                << "\n";
    }
  } else {
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info << "Absoluate L_2 error: "
                << mgard_x::L_2_error(shape, original_data, decompressed_data,
                                      mode)
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info << "Relative L_2 error: "
                << mgard_x::L_2_error(shape, original_data, decompressed_data,
                                      mode)
                << "\n";
    }
  }
  // std::cout << mgard_x::log::log_info << "L_2 error: " <<
  // mgard_x::L_2_error(n, original_data, decompressed_data) << "\n";
  std::cout << mgard_x::log::log_info
            << "MSE: " << mgard_x::MSE(n, original_data, decompressed_data)
            << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_x::log::log_info
            << "PSNR: " << mgard_x::PSNR(n, original_data, decompressed_data)
            << "\n";
}

template <typename T>
int launch_compress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    const char *input_file, const char *output_file,
                    std::vector<mgard_x::SIZE> shape, T tol, T s,
                    enum mgard_x::error_bound_type mode, bool verbose) {
  high_resolution_clock::time_point start, end;
  duration<double> time_span;

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  T *original_data;
  size_t in_size;
  if (std::string(input_file).compare("random") == 0) {
    in_size = original_size * sizeof(T);
    original_data = new T[original_size];
    srand(7117);
    T c = 0;
    for (size_t i = 0; i < original_size; i++) {
      original_data[i] = rand() % 10 + 1;
    }
  } else {
    in_size = readfile(input_file, original_data);
  }
  if (in_size != original_size * sizeof(T)) {
    std::cout << mgard_x::log::log_err << "input file size mismatch!\n";
  }

  T norm;
  if (s == std::numeric_limits<T>::infinity()) {
    norm = mgard_x::L_inf_norm(original_size, original_data);
  } else {
    norm = mgard_x::L_2_norm(shape, original_data);
  }

  void *compressed_data = NULL;
  size_t compressed_size = 0;
  void *decompressed_data = NULL;

  if (mode == mgard_x::error_bound_type::REL)
    tol *= norm;

  if (D == 1) {
    std::array<std::size_t, 1> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    const mgard::TensorMeshHierarchy<1, T> hierarchy(array_shape);
    if (verbose)
      start = high_resolution_clock::now();
    mgard::CompressedDataset<1, T> compressed_dataset =
        mgard::compress(hierarchy, original_data, s, tol);
    if (verbose) {
      end = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(end - start);
      std::cout << mgard_x::log::log_time
                << "Overall compression time: " << time_span.count() << " s ("
                << (double)(original_size * sizeof(T)) / time_span.count() / 1e9
                << " GB/s)\n";
    }
    std::ostringstream buf;
    compressed_dataset.write(buf);
    std::string tmp_str = buf.str();
    compressed_size = tmp_str.length();
    compressed_data = (void *)malloc(compressed_size);
    memcpy(compressed_data, tmp_str.c_str(), compressed_size);
  }
  if (D == 2) {
    std::array<std::size_t, 2> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    const mgard::TensorMeshHierarchy<2, T> hierarchy(array_shape);
    if (verbose)
      start = high_resolution_clock::now();
    mgard::CompressedDataset<2, T> compressed_dataset =
        mgard::compress(hierarchy, original_data, s, tol);
    if (verbose) {
      end = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(end - start);
      std::cout << mgard_x::log::log_time
                << "Overall compression time: " << time_span.count() << " s ("
                << (double)(original_size * sizeof(T)) / time_span.count() / 1e9
                << " GB/s)\n";
    }
    std::ostringstream buf;
    compressed_dataset.write(buf);
    std::string tmp_str = buf.str();
    compressed_size = tmp_str.length();
    compressed_data = (void *)malloc(compressed_size);
    memcpy(compressed_data, tmp_str.c_str(), compressed_size);
  }
  if (D == 3) {
    std::array<std::size_t, 3> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    const mgard::TensorMeshHierarchy<3, T> hierarchy(array_shape);
    if (verbose)
      start = high_resolution_clock::now();
    mgard::CompressedDataset<3, T> compressed_dataset =
        mgard::compress(hierarchy, original_data, s, tol);
    if (verbose) {
      end = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(end - start);
      std::cout << mgard_x::log::log_time
                << "Overall compression time: " << time_span.count() << " s ("
                << (double)(original_size * sizeof(T)) / time_span.count() / 1e9
                << " GB/s)\n";
    }
    std::ostringstream buf;
    compressed_dataset.write(buf);
    std::string tmp_str = buf.str();
    compressed_size = tmp_str.length();
    compressed_data = (void *)malloc(compressed_size);
    memcpy(compressed_data, tmp_str.c_str(), compressed_size);
  }
  if (D == 4) {
    std::array<std::size_t, 4> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    const mgard::TensorMeshHierarchy<4, T> hierarchy(array_shape);
    if (verbose)
      start = high_resolution_clock::now();
    mgard::CompressedDataset<4, T> compressed_dataset =
        mgard::compress(hierarchy, original_data, s, tol);
    if (verbose) {
      end = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(end - start);
      std::cout << mgard_x::log::log_time
                << "Overall compression time: " << time_span.count() << " s ("
                << (double)(original_size * sizeof(T)) / time_span.count() / 1e9
                << " GB/s)\n";
    }
    std::ostringstream buf;
    compressed_dataset.write(buf);
    std::string tmp_str = buf.str();
    compressed_size = tmp_str.length();
    compressed_data = (void *)malloc(compressed_size);
    memcpy(compressed_data, tmp_str.c_str(), compressed_size);
  }
  if (D == 5) {
    std::array<std::size_t, 5> array_shape;
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    const mgard::TensorMeshHierarchy<5, T> hierarchy(array_shape);
    if (verbose)
      start = high_resolution_clock::now();
    mgard::CompressedDataset<5, T> compressed_dataset =
        mgard::compress(hierarchy, original_data, s, tol);
    if (verbose) {
      end = high_resolution_clock::now();
      time_span = duration_cast<duration<double>>(end - start);
      std::cout << mgard_x::log::log_time
                << "Overall compression time: " << time_span.count() << " s ("
                << (double)(original_size * sizeof(T)) / time_span.count() / 1e9
                << " GB/s)\n";
    }
    std::ostringstream buf;
    compressed_dataset.write(buf);
    std::string tmp_str = buf.str();
    compressed_size = tmp_str.length();
    compressed_data = (void *)malloc(compressed_size);
    memcpy(compressed_data, tmp_str.c_str(), compressed_size);
  }

  writefile(output_file, compressed_size, compressed_data);

  printf("In size:  %10ld  Out size: %10ld  Compression ratio: %f \n",
         original_size * sizeof(T), compressed_size,
         (double)original_size * sizeof(T) / compressed_size);

  if (verbose) {
    char temp[] = "decompressed_tmp.dat";
    launch_decompress<T>(D, dtype, output_file, temp, shape, tol, s, mode,
                         true);
    readfile(temp, decompressed_data);
    print_statistics<T>(s, mode, shape, original_data, (T *)decompressed_data);
    // delete[](T *) decompressed_data;
  }

  // delete[](T *) original_data;
  // delete[](mgard_x::SERIALIZED_TYPE *) compressed_data;
  return 0;
}

template <typename T>
int launch_decompress(mgard_x::DIM D, enum mgard_x::data_type dtype,
                      const char *input_file, const char *output_file,
                      std::vector<mgard_x::SIZE> shape, T tol, T s,
                      enum mgard_x::error_bound_type mode, bool verbose) {

  high_resolution_clock::time_point start, end;
  duration<double> time_span;

  mgard_x::SERIALIZED_TYPE *compressed_data;
  size_t compressed_size = readfile(input_file, compressed_data);
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < shape.size(); i++)
    original_size *= shape[i];

  T norm = 1;
  if (mode == mgard_x::error_bound_type::REL)
    tol *= norm;

  if (verbose) {
    start = high_resolution_clock::now();
  }
  void const *const compressed_data_const = compressed_data;

  mgard::MemoryBuffer<const unsigned char> new_data_ =
      mgard::decompress(compressed_data_const, compressed_size);
  const void *decompressed_data = new_data_.data.get();

  if (verbose) {
    end = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(end - start);
    std::cout << mgard_x::log::log_time
              << "Overall decompression time: " << time_span.count() << " s ("
              << (double)(original_size * sizeof(T)) / time_span.count() / 1e9
              << " GB/s)\n";
  }

  int elem_size = 0;
  if (dtype == mgard_x::data_type::Double)
    elem_size = 8;
  else if (dtype == mgard_x::data_type::Float)
    elem_size = 4;
  writefile(output_file, original_size * elem_size, decompressed_data);

  // if (dtype == mgard_x::data_type::Double)
  //   delete[](double *) decompressed_data;
  // else if (dtype == mgard_x::data_type::Float)
  //   delete[](float *) decompressed_data;

  // causing segfault
  // delete [] compressed_data;
  return 0;
}

template int
launch_compress<float>(mgard_x::DIM D, enum mgard_x::data_type dtype,
                       const char *input_file, const char *output_file,
                       std::vector<mgard_x::SIZE> shape, float tol, float s,
                       enum mgard_x::error_bound_type mode, bool verbose);
template int
launch_compress<double>(mgard_x::DIM D, enum mgard_x::data_type dtype,
                        const char *input_file, const char *output_file,
                        std::vector<mgard_x::SIZE> shape, double tol, double s,
                        enum mgard_x::error_bound_type mode, bool verbose);

template int
launch_decompress<float>(mgard_x::DIM D, enum mgard_x::data_type dtype,
                         const char *input_file, const char *output_file,
                         std::vector<mgard_x::SIZE> shape, float tol, float s,
                         enum mgard_x::error_bound_type mode, bool verbose);
template int launch_decompress<double>(
    mgard_x::DIM D, enum mgard_x::data_type dtype, const char *input_file,
    const char *output_file, std::vector<mgard_x::SIZE> shape, double tol,
    double s, enum mgard_x::error_bound_type mode, bool verbose);

bool try_compression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-z"))
    return false;
  std::cout << mgard_x::log::log_info << "mode: compression\n";
  std::string input_file = get_arg(argc, argv, "-i");
  std::string output_file = get_arg(argc, argv, "-c");

  std::cout << mgard_x::log::log_info << "original data: " << input_file
            << "\n";
  std::cout << mgard_x::log::log_info << "compressed data: " << output_file
            << "\n";

  enum mgard_x::data_type dtype;
  std::string dt = get_arg(argc, argv, "-t");
  if (dt.compare("s") == 0) {
    dtype = mgard_x::data_type::Float;
    std::cout << mgard_x::log::log_info << "data type: Single precision\n";
  } else if (dt.compare("d") == 0) {
    dtype = mgard_x::data_type::Double;
    std::cout << mgard_x::log::log_info << "data type: Double precision\n";
  } else
    print_usage_message("wrong data type.");

  mgard_x::DIM D = get_arg_int(argc, argv, "-n");
  std::vector<mgard_x::SIZE> shape = get_arg_dims(argc, argv, "-n");
  std::string shape_string = "shape (";
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    shape_string = shape_string + std::to_string(shape[d]) + " ";
  shape_string = shape_string + ")";

  enum mgard_x::error_bound_type mode; // REL or ABS
  std::string em = get_arg(argc, argv, "-m");
  if (em.compare("rel") == 0) {
    mode = mgard_x::error_bound_type::REL;
    std::cout << mgard_x::log::log_info << "error bound mode: Relative\n";
  } else if (em.compare("abs") == 0) {
    mode = mgard_x::error_bound_type::ABS;
    std::cout << mgard_x::log::log_info << "error bound mode: Absolute\n";
  } else
    print_usage_message("wrong error bound mode.");

  double tol = get_arg_double(argc, argv, "-e");
  double s = get_arg_double(argc, argv, "-s");

  std::cout << mgard_x::log::log_info << "error bound: " << tol << "\n";
  std::cout << mgard_x::log::log_info << "s: " << s << "\n";

  bool verbose = has_arg(argc, argv, "-v");
  if (verbose)
    std::cout << mgard_x::log::log_info << "Verbose: enabled\n";
  if (dtype == mgard_x::data_type::Double)
    launch_compress<double>(D, dtype, input_file.c_str(), output_file.c_str(),
                            shape, (double)tol, (double)s, mode, verbose);
  else if (dtype == mgard_x::data_type::Float)
    launch_compress<float>(D, dtype, input_file.c_str(), output_file.c_str(),
                           shape, (float)tol, (float)s, mode, verbose);
  return true;
}

bool try_decompression(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-x"))
    return false;
  std::cout << mgard_x::log::log_info << "mode: decompression\n";
  std::string input_file = get_arg(argc, argv, "-c");
  std::string output_file = get_arg(argc, argv, "-d");

  std::cout << mgard_x::log::log_info << "compressed data: " << input_file
            << "\n";
  std::cout << mgard_x::log::log_info << "decompressed data: " << output_file
            << "\n";

  enum mgard_x::data_type dtype;
  std::string dt = get_arg(argc, argv, "-t");
  if (dt.compare("s") == 0) {
    dtype = mgard_x::data_type::Float;
    std::cout << mgard_x::log::log_info << "data type: Single precision\n";
  } else if (dt.compare("d") == 0) {
    dtype = mgard_x::data_type::Double;
    std::cout << mgard_x::log::log_info << "data type: Double precision\n";
  } else
    print_usage_message("wrong data type.");

  mgard_x::DIM D = get_arg_int(argc, argv, "-n");
  std::vector<mgard_x::SIZE> shape = get_arg_dims(argc, argv, "-n");
  std::string shape_string = "shape (";
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    shape_string = shape_string + std::to_string(shape[d]) + " ";
  shape_string = shape_string + ")";

  enum mgard_x::error_bound_type mode; // REL or ABS
  std::string em = get_arg(argc, argv, "-m");
  if (em.compare("rel") == 0) {
    mode = mgard_x::error_bound_type::REL;
    std::cout << mgard_x::log::log_info << "error bound mode: Relative\n";
  } else if (em.compare("abs") == 0) {
    mode = mgard_x::error_bound_type::ABS;
    std::cout << mgard_x::log::log_info << "error bound mode: Absolute\n";
  } else
    print_usage_message("wrong error bound mode.");

  double tol = get_arg_double(argc, argv, "-e");
  double s = get_arg_double(argc, argv, "-s");

  std::cout << std::scientific;
  std::cout << mgard_x::log::log_info << "error bound: " << tol << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_x::log::log_info << "s: " << s << "\n";

  bool verbose = has_arg(argc, argv, "-v");
  if (verbose)
    std::cout << mgard_x::log::log_info << "Verbose: enabled\n";
  if (dtype == mgard_x::data_type::Double)
    launch_decompress<double>(D, dtype, input_file.c_str(), output_file.c_str(),
                              shape, (double)tol, (double)s, mode, verbose);
  else if (dtype == mgard_x::data_type::Float)
    launch_decompress<float>(D, dtype, input_file.c_str(), output_file.c_str(),
                             shape, (float)tol, (float)s, mode, verbose);
  return true;
}

int main(int argc, char *argv[]) {

  if (!try_compression(argc, argv) && !try_decompression(argc, argv)) {
    print_usage_message("");
  }
  return 0;
}
