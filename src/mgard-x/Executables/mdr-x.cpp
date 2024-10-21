/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <cstring>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "compress_x.hpp"
#include "mdr_x.hpp"
#include "mgard-x/RuntimeX/Utilities/Log.h"
#include "mgard-x/Utilities/ErrorCalculator.h"

#include "ArgumentParser.h"
using namespace std::chrono;

void print_usage_message(std::string error) {
  if (error.compare("") != 0) {
    std::cout << mgard_x::log::log_err << error << std::endl;
  }
  printf("Options\n\
\t -z / --refactor: refactor data\n\
\t\t -i / --input <path to data file to be refactored>\n\
\t\t -o / --output <path to refactored data dir>\n\
\t\t -dt / --data-type <s/single|d/double>: data type (s: single; d:double)\n\
\t\t -dim / --dimension <ndim>: total number of dimensions\n\
\t\t\t [dim1]: slowest dimention\n\
\t\t\t [dim2]: 2nd slowest dimention\n\
\t\t\t  ...\n\
\t\t\t [dimN]: fastest dimention\n\
\t\t -d / --device <auto|serial|cuda|hip>: device type\n\
\t\t (optional) -v / --verbose <0|1|2|3> 0: error; 1: error+info; 2: error+timing; 3: all\n\
\t\t (optional) -m / --max-memory <max memory usage>  \n\
\t\t (optional) -dd / --domain-decomposition <max-dim|block>\n\
\t\t\t (optional) -dd-size / --domain-decomposition-size <integer> (for block domain decomposition only) \n\
\n\
\t -x / --reconstruct: reconstruct data\n\
\t\t -i / --input <path to refactored data dir>\n\
\t\t -o / --output <path to reconstructed data file>\n\
\t\t (optional)  -g / --orginal <path to original data file for error calculation> (optinal)\n\
\t\t -e / --error-bound <float>: error bound\n\
\t\t -s / --smoothness <float>: smoothness parameter\n\
\t\t -d <auto|serial|cuda|hip>: device type\n\
\t\t (optional) -v / --verbose <0|1|2|3> 0: error; 1: error+info; 2: error+timing; 3: all\n");
  exit(0);
}

template <typename T> void min_max(size_t n, T *in_buff) {
  T min = std::numeric_limits<T>::infinity();
  T max = 0;
  for (size_t i = 0; i < n; i++) {
    if (min > in_buff[i]) {
      min = in_buff[i];
    }
    if (max < in_buff[i]) {
      max = in_buff[i];
    }
  }
  printf("Min: %f, Max: %f\n", min, max);
}

template <typename T> size_t readfile(std::string input_file, T *&in_buff) {
  // std::cout << mgard_x::log::log_info << "Loading file: " << input_file <<
  // "\n";

  FILE *pFile;
  pFile = fopen(input_file.c_str(), "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "file open error!\n";
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
void readfile(std::string input_file, std::vector<T> &in_buff) {
  // std::cout << mgard_x::log::log_info << "Loading file: " << input_file <<
  // "\n";

  FILE *pFile;
  pFile = fopen(input_file.c_str(), "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff.resize(lSize / sizeof(T));
  lSize = fread(in_buff.data(), 1, lSize, pFile);
  fclose(pFile);
}

template <typename T>
std::vector<T *> readcoords(const char *input_file, mgard_x::DIM D,
                            std::vector<mgard_x::SIZE> shape) {
  std::cout << mgard_x::log::log_info
            << "Loading coordinate file: " << input_file << "\n";
  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "coordinate file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  size_t expected_size = 0;
  for (mgard_x::DIM d = 0; d < D; d++) {
    expected_size += sizeof(T) * shape[d];
  }
  if (lSize < expected_size) {
    std::cout << mgard_x::log::log_err << "coordinate file read error!\n";
    exit(-1);
  }
  rewind(pFile);
  std::vector<T *> coords(D);
  for (mgard_x::DIM d = 0; d < D; d++) {
    coords[d] = (T *)malloc(shape[d]);
    lSize = fread(coords[d], sizeof(T), shape[d], pFile);
  }
  fclose(pFile);
  return coords;
}

template <typename T>
void writefile(std::string output_file, T *out_buff, size_t num_bytes) {
  FILE *file = fopen(output_file.c_str(), "w");
  fwrite(out_buff, 1, num_bytes, file);
  fclose(file);
}

template <typename T>
void print_statistics(double s, enum mgard_x::error_bound_type mode,
                      std::vector<mgard_x::SIZE> shape, T *original_data,
                      T *decompressed_data, T tol, bool normalize_coordinates) {
  mgard_x::SIZE n = 1;
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    n *= shape[d];
  T actual_error = 0.0;
  std::cout << std::scientific;
  if (s == std::numeric_limits<T>::infinity()) {
    actual_error =
        mgard_x::L_inf_error(n, original_data, decompressed_data, mode);
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info
                << "Relative L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  } else {
    actual_error = mgard_x::L_2_error(shape, original_data, decompressed_data,
                                      mode, normalize_coordinates);
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info
                << "Relative L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  }

  std::cout << mgard_x::log::log_info
            << "MSE: " << mgard_x::MSE(n, original_data, decompressed_data)
            << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_x::log::log_info
            << "PSNR: " << mgard_x::PSNR(n, original_data, decompressed_data)
            << "\n";

  if (actual_error > tol)
    exit(-1);
}

void create_dir(std::string name) {
  struct stat st = {0};
  if (stat(name.c_str(), &st) == -1) {
    mkdir(name.c_str(), 0700);
  }
}

void write_mdr(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
               mgard_x::MDR::RefactoredData &refactored_data,
               std::string output) {
  size_t size_written = 0;
  create_dir(output);
  std::vector<mgard_x::Byte> serialized_metadata =
      refactored_metadata.Serialize();
  writefile(output + "/header", refactored_metadata.header.data(),
            refactored_metadata.header.size());
  writefile(output + "/metadata", serialized_metadata.data(),
            serialized_metadata.size());
  for (int subdomain_id = 0; subdomain_id < refactored_metadata.metadata.size();
       subdomain_id++) {
    for (int level_idx = 0;
         level_idx <
         refactored_metadata.metadata[subdomain_id].level_sizes.size();
         level_idx++) {
      for (int bitplane_idx = 0;
           bitplane_idx < refactored_metadata.metadata[subdomain_id]
                              .level_sizes[level_idx]
                              .size();
           bitplane_idx++) {
        std::string filename = "component_" + std::to_string(subdomain_id) +
                               "_" + std::to_string(level_idx) + "_" +
                               std::to_string(bitplane_idx);
        writefile(output + "/" + filename,
                  refactored_data.data[subdomain_id][level_idx][bitplane_idx],
                  refactored_metadata.metadata[subdomain_id]
                      .level_sizes[level_idx][bitplane_idx]);
        size_written += refactored_metadata.metadata[subdomain_id]
                            .level_sizes[level_idx][bitplane_idx];
      }
    }
  }
  std::cout << mgard_x::log::log_info << size_written << " bytes written\n";
}

void read_mdr_metadata(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
                       mgard_x::MDR::RefactoredData &refactored_data,
                       std::string input) {

  readfile(input + "/header", refactored_metadata.header);
  std::vector<mgard_x::Byte> serialized_metadata;
  readfile(input + "/metadata", serialized_metadata);
  refactored_metadata.Deserialize(serialized_metadata);
  refactored_metadata.InitializeForReconstruction();
  refactored_data.InitializeForReconstruction(refactored_metadata);
}

void read_mdr(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
              mgard_x::MDR::RefactoredData &refactored_data, std::string input,
              bool initialize_signs) {

  int num_subdomains = refactored_metadata.metadata.size();
  for (int subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
    mgard_x::MDR::MDRMetadata metadata =
        refactored_metadata.metadata[subdomain_id];
    int num_levels = metadata.level_sizes.size();
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      int num_bitplanes = metadata.level_sizes[level_idx].size();
      int loaded_bitplanes = metadata.loaded_level_num_bitplanes[level_idx];
      int reqested_bitplanes =
          metadata.requested_level_num_bitplanes[level_idx];
      for (int bitplane_idx = loaded_bitplanes;
           bitplane_idx < reqested_bitplanes; bitplane_idx++) {
        std::string filename = "component_" + std::to_string(subdomain_id) +
                               "_" + std::to_string(level_idx) + "_" +
                               std::to_string(bitplane_idx);
        mgard_x::SIZE level_size = readfile(
            input + "/" + filename,
            refactored_data.data[subdomain_id][level_idx][bitplane_idx]);
        if (level_size != refactored_metadata.metadata[subdomain_id]
                              .level_sizes[level_idx][bitplane_idx]) {
          std::cout << "mdr component size mismatch.";
          exit(-1);
        }
      }
      if (initialize_signs) {
        // level sign
        refactored_data.level_signs[subdomain_id][level_idx] =
            (bool *)malloc(sizeof(bool) * metadata.level_num_elems[level_idx]);
        memset(refactored_data.level_signs[subdomain_id][level_idx], 0,
               sizeof(bool) * metadata.level_num_elems[level_idx]);
      }
    }
  }
}

int verbose_to_log_level(int verbose) {
  if (verbose == 0) {
    return mgard_x::log::ERR;
  } else if (verbose == 1) {
    return mgard_x::log::ERR | mgard_x::log::INFO;
  } else if (verbose == 2) {
    return mgard_x::log::ERR | mgard_x::log::TIME;
  } else if (verbose == 3) {
    return mgard_x::log::ERR | mgard_x::log::INFO | mgard_x::log::TIME;
  }
}

template <typename T>
int launch_refactor(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    std::string input_file, std::string output_file,
                    std::vector<mgard_x::SIZE> shape,
                    std::string domain_decomposition, mgard_x::SIZE block_size,
                    enum mgard_x::device_type dev_type, int verbose,
                    mgard_x::SIZE max_memory_footprint) {

  mgard_x::Config config;
  config.normalize_coordinates = false;
  config.log_level = verbose_to_log_level(verbose);
  config.decomposition = mgard_x::decomposition_type::MultiDim;
  if (domain_decomposition == "max-dim") {
    config.domain_decomposition = mgard_x::domain_decomposition_type::MaxDim;
  } else if (domain_decomposition == "block") {
    config.domain_decomposition = mgard_x::domain_decomposition_type::Block;
    config.block_size = block_size;
  } else if (domain_decomposition == "variable") {
    config.domain_decomposition = mgard_x::domain_decomposition_type::Block;
  }

  config.dev_type = dev_type;
  config.max_memory_footprint = max_memory_footprint;
  if (dtype == mgard_x::data_type::Float) {
    config.total_num_bitplanes = 32;
  } else if (dtype == mgard_x::data_type::Double) {
    config.total_num_bitplanes = 64;
  }

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  T *original_data = (T *)malloc(original_size * sizeof(T));
  size_t in_size = 0;
  if (std::string(input_file).compare("random") == 0) {
    in_size = original_size * sizeof(T);
    srand(7117);
    T c = 0;
    for (size_t i = 0; i < original_size; i++) {
      original_data[i] = rand() % 10 + 1;
    }
  } else {
    T *file_data;
    in_size = readfile(input_file, file_data);

    size_t loaded_size = 0;
    while (loaded_size < original_size) {
      std::memcpy(original_data + loaded_size, file_data,
                  std::min(in_size / sizeof(T), original_size - loaded_size) *
                      sizeof(T));
      loaded_size += std::min(in_size / sizeof(T), original_size - loaded_size);
    }
    in_size = loaded_size * sizeof(T);
  }
  if (in_size != original_size * sizeof(T)) {
    std::cout << mgard_x::log::log_warn << "input file size mismatch "
              << in_size << " vs. " << original_size * sizeof(T) << "!\n";
  }

  std::cout << mgard_x::log::log_info << "Max output data size: "
            << mgard_x::MDR::MDRMaxOutputDataSize(D, dtype, shape, config)
            << " bytes\n";

  mgard_x::MDR::RefactoredMetadata refactored_metadata;
  mgard_x::MDR::RefactoredData refactored_data;
  mgard_x::pin_memory(original_data, original_size * sizeof(T), config);

  mgard_x::MDR::MDRefactor(D, dtype, shape, original_data, refactored_metadata,
                           refactored_data, config, false);

  write_mdr(refactored_metadata, refactored_data, output_file);

  mgard_x::unpin_memory(original_data, config);
  delete[](T *) original_data;

  return 0;
}

int launch_reconstruct(std::string input_file, std::string output_file,
                       std::string original_file, enum mgard_x::data_type dtype,
                       std::vector<mgard_x::SIZE> shape, double tolerance,
                       double s, enum mgard_x::error_bound_type mode,
                       bool adaptive_resolution,
                       enum mgard_x::device_type dev_type, int verbose) {

  mgard_x::Config config;
  config.normalize_coordinates = false;
  config.log_level = verbose_to_log_level(verbose);
  config.dev_type = dev_type;
  config.mdr_adaptive_resolution = adaptive_resolution;

  mgard_x::Byte *original_data;
  size_t in_size = 0;
  if (original_file.compare("none") != 0 && !config.mdr_adaptive_resolution) {
    if (original_file.compare("random") == 0) {
      size_t original_size = 1;
      for (mgard_x::DIM i = 0; i < shape.size(); i++)
        original_size *= shape[i];
      if (dtype == mgard_x::data_type::Float) {
        in_size = original_size * sizeof(float);
        original_data = (mgard_x::Byte *)new float[original_size];
        srand(7117);
        for (size_t i = 0; i < original_size; i++) {
          ((float *)original_data)[i] = rand() % 10 + 1;
        }
      } else if (dtype == mgard_x::data_type::Double) {
        in_size = original_size * sizeof(double);
        original_data = (mgard_x::Byte *)new double[original_size];
        srand(7117);
        for (size_t i = 0; i < original_size; i++) {
          ((double *)original_data)[i] = rand() % 10 + 1;
        }
      }

    } else {
      in_size = readfile(original_file, original_data);
    }
  }

  mgard_x::MDR::RefactoredMetadata refactored_metadata;
  mgard_x::MDR::RefactoredData refactored_data;
  mgard_x::MDR::ReconstructedData reconstructed_data;
  read_mdr_metadata(refactored_metadata, refactored_data, input_file);
  bool first_reconstruction = true;
  std::vector<double> tols;
  tols.push_back(tolerance);
  // reconstruct to one toleracne for now
  // Progressively reconstruct to additional tolerance can be added by appending
  // to tols
  for (double tol : tols) {
    for (auto &metadata : refactored_metadata.metadata) {
      metadata.requested_tol = tol;
      metadata.requested_s = s;
    }
    mgard_x::MDR::MDRequest(refactored_metadata, config);
    for (auto &metadata : refactored_metadata.metadata) {
      metadata.PrintStatus();
    }
    read_mdr(refactored_metadata, refactored_data, input_file,
             first_reconstruction);

    mgard_x::MDR::MDReconstruct(refactored_metadata, refactored_data,
                                reconstructed_data, config, false);

    first_reconstruction = false;

    if (input_file.compare("none") != 0 && !config.mdr_adaptive_resolution) {
      if (dtype == mgard_x::data_type::Float) {
        print_statistics<float>(s, mode, shape, (float *)original_data,
                                (float *)reconstructed_data.data[0], tol,
                                config.normalize_coordinates);
      } else if (dtype == mgard_x::data_type::Double) {
        print_statistics<double>(s, mode, shape, (double *)original_data,
                                 (double *)reconstructed_data.data[0], tol,
                                 config.normalize_coordinates);
      }
    }
  }
  return 0;
}

bool try_refactoring(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-z", "--refactor"))
    return false;
  mgard_x::log::info("Mode: refactor", true);

  std::string input_file =
      get_arg<std::string>(argc, argv, "Original data", "-i", "--input");
  std::string output_file =
      get_arg<std::string>(argc, argv, "Refactored data", "-o", "--output");
  enum mgard_x::data_type dtype = get_data_type(argc, argv);
  std::vector<mgard_x::SIZE> shape =
      get_args<mgard_x::SIZE>(argc, argv, "Dimensions", "-dim", "--dimension");
  // std::string lossless_level = get_arg<std::string>(argc, argv, "Lossless",
  // "-l", "--lossless");
  enum mgard_x::device_type dev_type = get_device_type(argc, argv);
  int verbose = 0;
  if (has_arg(argc, argv, "-v", "--verbose")) {
    verbose = get_arg<int>(argc, argv, "Verbose", "-v", "--verbose");
  }
  mgard_x::SIZE max_memory_footprint =
      std::numeric_limits<mgard_x::SIZE>::max();
  if (has_arg(argc, argv, "-m", "--max-memory")) {
    max_memory_footprint = (mgard_x::SIZE)get_arg<double>(
        argc, argv, "Max memory", "-m", "--max-memory");
  }
  std::string domain_decomposition = "max-dim";
  mgard_x::SIZE block_size = 0;
  if (has_arg(argc, argv, "-dd", "--domain-decomposition")) {
    domain_decomposition = get_arg<std::string>(
        argc, argv, "Domain decomposition", "-dd", "--domain-decomposition");
    if (domain_decomposition == "block") {
      block_size = get_arg<mgard_x::SIZE>(argc, argv, "Block size", "-dd-size",
                                          "--domain-decomposition-size");
    }
  }
  if (dtype == mgard_x::data_type::Double) {
    launch_refactor<double>(shape.size(), dtype, input_file.c_str(),
                            output_file.c_str(), shape, domain_decomposition,
                            block_size, dev_type, verbose,
                            max_memory_footprint);
  } else if (dtype == mgard_x::data_type::Float) {
    launch_refactor<float>(shape.size(), dtype, input_file.c_str(),
                           output_file.c_str(), shape, domain_decomposition,
                           block_size, dev_type, verbose, max_memory_footprint);
  }
  return true;
}

bool try_reconstruction(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-x", "--reconstruct"))
    return false;
  mgard_x::log::info("mode: reconstruct", true);
  std::string input_file =
      get_arg<std::string>(argc, argv, "Refactored data", "-i", "--input");
  std::string output_file =
      get_arg<std::string>(argc, argv, "Reconstructed data", "-o", "--output");
  // default is none (means original data not provided)
  std::string original_file = "none";
  enum mgard_x::data_type dtype;
  std::vector<mgard_x::SIZE> shape;
  if (has_arg(argc, argv, "-g", "--orignal")) {
    original_file =
        get_arg<std::string>(argc, argv, "Original data", "-g", "--orignal");
    enum mgard_x::data_type dtype = get_data_type(argc, argv);
    std::vector<mgard_x::SIZE> shape = get_args<mgard_x::SIZE>(
        argc, argv, "Dimensions", "-dim", "--dimension");
  }
  // only abs mode is supported now
  enum mgard_x::error_bound_type mode =
      mgard_x::error_bound_type::ABS; // REL or ABS

  double tol =
      get_arg<double>(argc, argv, "Error bound", "-e", "--error-bound");
  double s = get_arg<double>(argc, argv, "Smoothness", "-s", "--smoothness");
  enum mgard_x::device_type dev_type = get_device_type(argc, argv);
  int verbose = 0;
  if (has_arg(argc, argv, "-v", "--verbose")) {
    verbose = get_arg<int>(argc, argv, "Verbose", "-v", "--verbose");
  }
  bool adaptive_resolution = false;
  if (has_arg(argc, argv, "-ar", "--adaptive-resolution")) {
    adaptive_resolution = get_arg<int>(argc, argv, "Adaptive resolution", "-ar",
                                       "--adaptive-resolution");
  }
  if (verbose)
    std::cout << mgard_x::log::log_info << "verbose: enabled.\n";
  launch_reconstruct(input_file, output_file, original_file, dtype, shape, tol,
                     s, mode, adaptive_resolution, dev_type, verbose);
  return true;
}

int main(int argc, char *argv[]) {

  if (!try_refactoring(argc, argv) && !try_reconstruction(argc, argv)) {
    print_usage_message("");
  }
  return 0;
}
