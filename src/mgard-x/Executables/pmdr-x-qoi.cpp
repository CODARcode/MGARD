/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
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

#include <mpi.h>
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
  \n\
 \t -x / --reconstruct: reconstruct data\n\
 \t\t -i / --input <path to refactored data dir>\n\
 \t\t -o / --output <path to reconstructed data file>\n\
 \t\t (optional)  -g / --orginal <path to original data file for error calculation> (optinal)\n\
 \t\t -e / --error-bound <float>: error bound\n\
 \t\t -me / --multi-error-bounds <num of error bounds> <float> <float>..: multiple error bounds\n\
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
size_t readfile_header_metadata(std::string input_file,
                                std::vector<T> &in_buff) {
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
  return lSize;
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

  // if (actual_error > tol)
  // exit(-1);
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
        if (refactored_metadata.metadata[subdomain_id]
                .level_sizes[level_idx][bitplane_idx] > 0) {
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
  }
  std::cout << mgard_x::log::log_info << size_written << " bytes written\n";
}

size_t read_mdr_metadata(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
                         mgard_x::MDR::RefactoredData &refactored_data,
                         std::string input) {

  size_t metadata_size = 0;
  metadata_size +=
      readfile_header_metadata(input + "/header", refactored_metadata.header);
  std::vector<mgard_x::Byte> serialized_metadata;
  metadata_size +=
      readfile_header_metadata(input + "/metadata", serialized_metadata);
  refactored_metadata.Deserialize(serialized_metadata);
  refactored_metadata.InitializeForReconstruction();
  refactored_data.InitializeForReconstruction(refactored_metadata);
  return metadata_size;
}

//  size_t read_mdr(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
//                mgard_x::MDR::RefactoredData &refactored_data, std::string
//                input, bool initialize_signs, mgard_x::Config config) {

//    size_t size_read = 0;
//    int num_subdomains = refactored_metadata.metadata.size();
//    for (int subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++)
//    {
//      mgard_x::MDR::MDRMetadata metadata =
//          refactored_metadata.metadata[subdomain_id];
//      int num_levels = metadata.level_sizes.size();
//      for (int level_idx = 0; level_idx < num_levels; level_idx++) {
//        int num_bitplanes = metadata.level_sizes[level_idx].size();
//        int loaded_bitplanes = metadata.loaded_level_num_bitplanes[level_idx];
//        int reqested_bitplanes =
//            metadata.requested_level_num_bitplanes[level_idx];
//        for (int bitplane_idx = loaded_bitplanes;
//             bitplane_idx < reqested_bitplanes; bitplane_idx++) {
//          std::string filename = "component_" + std::to_string(subdomain_id) +
//                                 "_" + std::to_string(level_idx) + "_" +
//                                 std::to_string(bitplane_idx);
//          mgard_x::SIZE level_size = readfile(
//              input + "/" + filename,
//              refactored_data.data[subdomain_id][level_idx][bitplane_idx]);
//          mgard_x::pin_memory(
//              refactored_data.data[subdomain_id][level_idx][bitplane_idx],
//              level_size, config);
//          if (level_size != refactored_metadata.metadata[subdomain_id]
//                                .level_sizes[level_idx][bitplane_idx]) {
//            std::cout << "mdr component size mismatch.";
//            exit(-1);
//          }
//          size_read += level_size;
//        }
//        if (initialize_signs) {
//          // level sign
//          refactored_data.level_signs[subdomain_id][level_idx] =
//              (bool *)malloc(sizeof(bool) *
//              metadata.level_num_elems[level_idx]);
//          memset(refactored_data.level_signs[subdomain_id][level_idx], 0,
//                 sizeof(bool) * metadata.level_num_elems[level_idx]);
//          mgard_x::pin_memory(
//              refactored_data.level_signs[subdomain_id][level_idx],
//              sizeof(bool) * metadata.level_num_elems[level_idx], config);
//        }
//      }
//    }
//    return size_read;
//  }

size_t read_mdr(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
                mgard_x::MDR::RefactoredData &refactored_data,
                std::string input, bool initialize_signs,
                mgard_x::Config config) {

  size_t size_read = 0;
  int num_subdomains = refactored_metadata.metadata.size();
  for (int subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
    mgard_x::MDR::MDRMetadata metadata =
        refactored_metadata.metadata[subdomain_id];
    int num_levels = metadata.level_sizes.size();
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      int num_bitplanes = metadata.level_sizes[level_idx].size();
      for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {

        if (refactored_metadata.metadata[subdomain_id]
                .level_sizes[level_idx][bitplane_idx] > 0) {
          std::string filename = "component_" + std::to_string(subdomain_id) +
                                 "_" + std::to_string(level_idx) + "_" +
                                 std::to_string(bitplane_idx);
          mgard_x::SIZE level_size = readfile(
              input + "/" + filename,
              refactored_data.data[subdomain_id][level_idx][bitplane_idx]);
          mgard_x::pin_memory(
              refactored_data.data[subdomain_id][level_idx][bitplane_idx],
              level_size, config);
          if (level_size != refactored_metadata.metadata[subdomain_id]
                                .level_sizes[level_idx][bitplane_idx]) {
            std::cout << "mdr component size mismatch.";
            exit(-1);
          }
          size_read += level_size;
        }
      }
      if (initialize_signs) {
        // level sign
        refactored_data.level_signs[subdomain_id][level_idx] =
            (bool *)malloc(sizeof(bool) * metadata.level_num_elems[level_idx]);
        memset(refactored_data.level_signs[subdomain_id][level_idx], 0,
               sizeof(bool) * metadata.level_num_elems[level_idx]);
        mgard_x::pin_memory(
            refactored_data.level_signs[subdomain_id][level_idx],
            sizeof(bool) * metadata.level_num_elems[level_idx], config);
      }
    }
  }
  return size_read;
}

// void posterior_bp_request(mgard_x::MDR::RefactoredMetadata
// &refactored_metadata, double tol, int rank){ 	if (tol == 1e-1) {
// switch (rank) { 			case 0:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 28, 28,
// 24, 24, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 28, 28, 24, 24, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 28, 28, 24, 24, 20, 16, 12, 12, 8}; 				break;
// case 1:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 28, 28,
// 24, 20, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 20, 16, 12, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 28, 28, 24, 20, 20, 16, 12, 12, 8}; 				break;
// case 2:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {28, 28, 24,
// 20, 20, 16, 12, 12, 8, 8};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {28, 28, 24, 20, 20, 16, 12, 12, 8, 8};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 28, 28, 24, 20, 20, 16, 12, 12, 8}; 				break;
// case 3:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 28, 28,
// 24, 24, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 28, 28, 24, 20, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 28, 28, 24, 20, 20, 16, 12, 12, 8}; 				break;
// case 4:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 28, 28,
// 24, 24, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 28, 28, 24, 24, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 20, 20, 16, 12, 12}; 				break;
// case 5:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {28, 28, 24,
// 24, 20, 16, 12, 12, 8, 8};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 20, 16, 12, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 28, 28, 24, 24, 20, 16, 12, 12, 8}; 				break;
// case 6:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 28, 28,
// 24, 24, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 28, 28, 24, 24, 20, 16, 12, 12, 8};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 20, 16, 12, 12}; 				break;
// case 7:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 28,
// 28, 24, 20, 16, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 16, 16, 12, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 28, 28, 24, 24, 20, 16, 12, 8, 8}; 				break;
// default: 				break;
// 		}
// 	}	else if (tol == 1e-2) {
// 		switch (rank) {
// 			case 0:
// 				refactored_metadata.metadata[0].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 24, 20, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 12, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 24, 20, 16, 12, 12}; 				break;
// case 1:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 24, 20, 20, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 24, 20, 20, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 20, 20, 16, 12, 12}; 				break;
// case 2:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 28,
// 28, 24, 20, 16, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 16, 16, 12, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 16, 16, 12, 12}; 				break;
// case 3:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 28,
// 28, 24, 20, 20, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 20, 16, 12, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 20, 16, 12, 12}; 				break;
// case 4:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 28,
// 28, 24, 20, 16, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 20, 20, 16, 12, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 20, 20, 16, 12, 12}; 				break;
// case 5:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 28,
// 28, 24, 20, 16, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 16, 16, 12, 12}; 				break;
// case 6:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 28,
// 28, 24, 20, 20, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 20, 16, 12, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 28, 28, 24, 20, 20, 16, 12, 12}; 				break;
// case 7:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 28, 24, 20, 20, 16, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 20, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 28, 28, 24, 24, 20, 16, 16, 12, 8}; 				break;
// default: 				break;
// 		}
// 	}	else if (tol == 1e-3) {
// 		switch (rank) {
// 			case 0:
// 				refactored_metadata.metadata[0].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 20, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 20, 16, 12}; 				break;
// case 1:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 24, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 20, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 16, 12}; 				break;
// case 2:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 28, 24, 20, 16, 12, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 24, 20, 16, 16, 12}; 				break;
// case 3:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 28, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 16, 12}; 				break;
// case 4:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 24, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 20, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 20, 16, 12}; 				break;
// case 5:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 28, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 16, 12}; 				break;
// case 6:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 28, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 28, 28, 24, 20, 16, 16, 12}; 				break;
// case 7:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 28, 24, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 28, 24, 24, 20, 16, 16, 12};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 24, 20, 20, 16, 12}; 				break;
// default: 				break;
// 		}
// 	}	else if (tol == 1e-4) {
// 		switch (rank) {
// 			case 0:
// 				refactored_metadata.metadata[0].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16}; 				break;
// case 1:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 24, 24, 20, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16}; 				break;
// case 2:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 20, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16}; 				break;
// case 3:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16}; 				break;
// case 4:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16}; 				break;
// case 5:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16}; 				break;
// case 6:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 16, 16}; 				break;
// case 7:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 28, 28, 24, 20, 20, 16};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 28, 24, 20, 20, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 28, 24, 20, 20, 16, 12}; 				break;
// default: 				break;
// 		}
// 	}	else if (tol == 1e-5) {
// 		switch (rank) {
// 			case 0:
// 				refactored_metadata.metadata[0].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 32, 28, 24, 20, 20};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 32, 28, 28, 20, 20};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 32, 28, 24, 20, 20}; 				break;
// case 1:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 32, 32, 28, 24, 20};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20}; 				break;
// case 2:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 32, 32, 28, 24, 20, 20};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 32, 28, 24, 20, 16};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20}; 				break;
// case 3:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20}; 				break;
// case 4:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20}; 				break;
// case 5:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 32, 32, 28, 24, 20};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20}; 				break;
// case 6:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20}; 				break;
// case 7:
// refactored_metadata.metadata[0].requested_level_num_bitplanes = {32, 32, 32,
// 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[1].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 20};
// 				refactored_metadata.metadata[2].requested_level_num_bitplanes
// = {32, 32, 32, 32, 32, 28, 28, 24, 20, 16}; 				break;
// default: 				break;
// 		}
// 	}}

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
                    enum mgard_x::device_type dev_type, int verbose,
                    mgard_x::SIZE max_memory_footprint, int rank) {

  mgard_x::Config config;
  config.normalize_coordinates = false;
  config.log_level = verbose_to_log_level(verbose);
  config.decomposition = mgard_x::decomposition_type::MultiDim;
  // One subdomain per variable: the variables are stacked along dimension 0.
  config.domain_decomposition_mode =
      mgard_x::domain_decomposition_mode_type::Manual;
  config.domain_decomposition_block_size = shape;
  config.domain_decomposition_block_size[0] = shape[0] / 3;

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
  double local_elapsed_time = 0, max_time = 0;
  local_elapsed_time = -MPI_Wtime();
  mgard_x::MDR::MDRefactor(D, dtype, shape, original_data, refactored_metadata,
                           refactored_data, config, false);
  local_elapsed_time += MPI_Wtime();
  MPI_Reduce(&local_elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (!rank)
    std::cout << "max_elapsed_time = " << max_time << std::endl;
  write_mdr(refactored_metadata, refactored_data, output_file);

  mgard_x::unpin_memory(original_data, config);
  delete[](T *) original_data;

  return 0;
}

template <class T>
T compute_max_abs_error(const T *vec_ori, const T *vec_rec, size_t n) {
  T error = fabs(vec_ori[0] - vec_rec[0]);
  T max = error;
  for (int i = 1; i < n; i++) {
    error = fabs(vec_ori[i] - vec_rec[i]);
    if (max < error)
      max = error;
  }
  return max;
}

template <class T> T compute_value_range(const T *vec, size_t n) {
  T min = vec[0];
  T max = vec[0];
  for (int i = 0; i < n; i++) {
    if (vec[i] < min)
      min = vec[i];
    if (vec[i] > max)
      max = vec[i];
  }
  return max - min;
}

template <class T>
void compute_VTOT(const T *Vx, const T *Vy, const T *Vz, size_t n, T *V_TOT_) {
  for (int i = 0; i < n; i++) {
    double V_TOT_2 = Vx[i] * Vx[i] + Vy[i] * Vy[i] + Vz[i] * Vz[i];
    double V_TOT = sqrt(V_TOT_2);
    V_TOT_[i] = V_TOT;
  }
}

int launch_reconstruct(std::string input_file, std::string output_file,
                       std::string original_file, enum mgard_x::data_type dtype,
                       std::vector<mgard_x::SIZE> shape,
                       std::vector<double> tols, double s,
                       enum mgard_x::error_bound_type mode,
                       bool adaptive_resolution,
                       enum mgard_x::device_type dev_type, int verbose,
                       int rank, int decrease_method = 0) {

  double bitrate = 0;
  mgard_x::Config config;
  config.normalize_coordinates = false;
  config.log_level = verbose_to_log_level(verbose);
  config.dev_type = dev_type;
  config.mdr_adaptive_resolution = adaptive_resolution;

  config.mdr_qoi_mode = true;
  config.mdr_qoi_num_variables = shape.size();

  mgard_x::Byte *original_data;
  size_t in_size = 0;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < shape.size(); i++)
    original_size *= shape[i];
  if (original_file.compare("none") != 0 && !config.mdr_adaptive_resolution) {
    if (original_file.compare("random") == 0) {
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
      mgard_x::Byte *file_data;
      in_size = readfile<mgard_x::Byte>(original_file, file_data);

      if (dtype == mgard_x::data_type::Float) {
        original_size *= sizeof(float);
      } else if (dtype == mgard_x::data_type::Double) {
        original_size *= sizeof(double);
      }

      original_data = (mgard_x::Byte *)malloc(original_size);

      size_t loaded_size = 0;
      while (loaded_size < original_size) {

        std::memcpy(original_data + loaded_size, file_data,
                    std::min(in_size, original_size - loaded_size));
        loaded_size += std::min(in_size, original_size - loaded_size);
      }
      in_size = loaded_size;
    }
  }
  mgard_x::Byte *V_TOT_ori;
  std::vector<double> ebs;
  size_t num_elements;
  double tau = 0;
  V_TOT_ori = (mgard_x::Byte *)malloc(in_size / config.mdr_qoi_num_variables);
  mgard_x::Byte *org_Vx_ptr = original_data + original_size / 3 * 0;
  mgard_x::Byte *org_Vy_ptr = original_data + original_size / 3 * 1;
  mgard_x::Byte *org_Vz_ptr = original_data + original_size / 3 * 2;
  if (dtype == mgard_x::data_type::Float) {
    num_elements = (in_size / config.mdr_qoi_num_variables) / sizeof(float);
    compute_VTOT<float>((float *)org_Vx_ptr, (float *)org_Vy_ptr,
                        (float *)org_Vz_ptr, num_elements, (float *)V_TOT_ori);
    //  tau = compute_value_range((float *) V_TOT_ori, num_elements) * tols[0];
    if (decrease_method < 4) {
      ebs.push_back(compute_value_range((float *)org_Vx_ptr, num_elements) *
                    tols[0]);
      ebs.push_back(compute_value_range((float *)org_Vy_ptr, num_elements) *
                    tols[0]);
      ebs.push_back(compute_value_range((float *)org_Vz_ptr, num_elements) *
                    tols[0]);
    } else {
      std::vector<float> var_value_range;
      var_value_range.push_back(
          compute_value_range((float *)org_Vx_ptr, num_elements) * tols[0]);
      var_value_range.push_back(
          compute_value_range((float *)org_Vy_ptr, num_elements) * tols[0]);
      var_value_range.push_back(
          compute_value_range((float *)org_Vz_ptr, num_elements) * tols[0]);
      for (int i = 0; i < config.mdr_qoi_num_variables; i++) {
        ebs.push_back(
            *std::min_element(var_value_range.begin(), var_value_range.end()));
      }
    }
    float local_max = -std::numeric_limits<float>::max();
    float local_min = std::numeric_limits<float>::max();
    float global_max = 0, global_min = 0;
    float *V_TOT = (float *)V_TOT_ori;
    for (int i = 0; i < num_elements; i++) {
      if (V_TOT[i] > local_max)
        local_max = V_TOT[i];
      if (V_TOT[i] < local_min)
        local_min = V_TOT[i];
    }
    //  std::cout << "local_min = " << local_min << ", local_max = " <<
    //  local_max << std::endl;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX,
                  MPI_COMM_WORLD);
    tau = (double)(global_max - global_min) * tols[0];
  } else if (dtype == mgard_x::data_type::Double) {
    num_elements = (in_size / config.mdr_qoi_num_variables) / sizeof(double);
    compute_VTOT<double>((double *)org_Vx_ptr, (double *)org_Vy_ptr,
                         (double *)org_Vz_ptr, num_elements,
                         (double *)V_TOT_ori);
    //  tau = compute_value_range((double *) V_TOT_ori, num_elements) * tols[0];
    if (decrease_method < 4) {
      ebs.push_back(compute_value_range((double *)org_Vx_ptr, num_elements) *
                    tols[0]);
      ebs.push_back(compute_value_range((double *)org_Vy_ptr, num_elements) *
                    tols[0]);
      ebs.push_back(compute_value_range((double *)org_Vz_ptr, num_elements) *
                    tols[0]);
    } else {
      std::vector<double> var_value_range;
      var_value_range.push_back(
          compute_value_range((double *)org_Vx_ptr, num_elements) * tols[0]);
      var_value_range.push_back(
          compute_value_range((double *)org_Vy_ptr, num_elements) * tols[0]);
      var_value_range.push_back(
          compute_value_range((double *)org_Vz_ptr, num_elements) * tols[0]);
      for (int i = 0; i < config.mdr_qoi_num_variables; i++) {
        ebs.push_back(
            *std::min_element(var_value_range.begin(), var_value_range.end()));
      }
    }
    double local_min = -std::numeric_limits<double>::max();
    double local_max = std::numeric_limits<double>::max();
    double global_max = 0, global_min = 0;
    double *V_TOT = (double *)V_TOT_ori;
    for (int i = 0; i < num_elements; i++) {
      if (V_TOT[i] > local_max)
        local_max = V_TOT[i];
      if (V_TOT[i] < local_min)
        local_min = V_TOT[i];
    }
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    tau = (double)(global_max - global_min) * tols[0];
  }

  mgard_x::MDR::RefactoredMetadata refactored_metadata;
  mgard_x::MDR::RefactoredData refactored_data;
  mgard_x::MDR::ReconstructedData reconstructed_data;
  size_t metadata_size =
      read_mdr_metadata(refactored_metadata, refactored_data, input_file);
  refactored_metadata.total_size += metadata_size;

  refactored_metadata.relative_eb = tols[0];
  refactored_metadata.decrease_method = decrease_method;
  refactored_metadata.MPI_enabled = true;
  refactored_metadata.input_path = input_file;
  for (int i = 0; i < config.mdr_qoi_num_variables; i++) {
    refactored_metadata.metadata[i].num_elements = num_elements;
    if (decrease_method == 0) {
      refactored_metadata.metadata[i].requested_tol = ebs[i];
    } else if (decrease_method == 1) {
      refactored_metadata.metadata[i].requested_size = 1;
      refactored_metadata.metadata[i].segmented = true;
    } else if (decrease_method >= 2) {
      refactored_metadata.metadata[i].requested_tol = ebs[i];
      refactored_metadata.metadata[i].corresponding_error_return = true;
    }
    refactored_metadata.metadata[i].tau = tau;
    refactored_metadata.metadata[i].requested_s = s;
  }
  mgard_x::MDR::MDRequest(refactored_metadata, config);

  //  posterior_bp_request(refactored_metadata, tols[0], rank);
  //  double local_IO_time = 0, global_IO_time = 0;
  //  MPI_Barrier(MPI_COMM_WORLD);
  //  local_IO_time = -MPI_Wtime();
  size_t size_read =
      read_mdr(refactored_metadata, refactored_data, input_file, true, config);
  //  MPI_Barrier(MPI_COMM_WORLD);
  //  local_IO_time += MPI_Wtime();
  //  MPI_Reduce(&local_IO_time, &global_IO_time, 1, MPI_DOUBLE, MPI_MAX, 0,
  //  MPI_COMM_WORLD); if(!rank) std::cout << "IO_time = " << global_IO_time <<
  //  std::endl;

  // refactored_metadata.total_size += size_read;
  double local_elapsed_time = 0, max_time = 0;
  local_elapsed_time = -MPI_Wtime();
  mgard_x::MDR::MDReconstruct(refactored_metadata, refactored_data,
                              reconstructed_data, config, false);
  local_elapsed_time += MPI_Wtime();
  MPI_Reduce(&local_elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (!rank)
    std::cout << "max_elapsed_time = " << max_time << std::endl;
  double local_kernel_time = 0, global_kernel_time = 0;
  local_kernel_time = refactored_metadata.kernel_time;
  MPI_Reduce(&local_kernel_time, &global_kernel_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (!rank)
    std::cout << "max_kernel_time = " << global_kernel_time << std::endl;
  for (int i = 0; i < 8; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == i) {
      std::cout << "From Rank " << i << ": "
                << "kernel_time = " << local_kernel_time << std::endl;
    }
  }

  // we can check reconstructed_data.qoi_in_progress here

  //  std::cout << mgard_x::log::log_info << "Additional " << size_read
  //            << " bytes read for reconstruction\n";

  std::vector<mgard_x::Byte *> rec_var_ptrs;
  if (original_file.compare("none") != 0 && !config.mdr_adaptive_resolution) {
    for (int i = 0; i < config.mdr_qoi_num_variables; i++) {
      std::vector<mgard_x::SIZE> var_shape = shape;
      var_shape[0] /= config.mdr_qoi_num_variables;
      mgard_x::Byte *org_var_ptr = original_data + original_size / 3 * i;
      mgard_x::Byte *rec_var_ptr =
          reconstructed_data.data[0] + original_size / 3 * i;
      rec_var_ptrs.push_back(rec_var_ptr);
    }
  }
  mgard_x::Byte *V_TOT_rec;
  V_TOT_rec = (mgard_x::Byte *)malloc(in_size / config.mdr_qoi_num_variables);
  if (dtype == mgard_x::data_type::Float) {
    compute_VTOT<float>((float *)rec_var_ptrs[0], (float *)rec_var_ptrs[1],
                        (float *)rec_var_ptrs[2], num_elements,
                        (float *)V_TOT_rec);
  } else if (dtype == mgard_x::data_type::Double) {
    compute_VTOT<double>((double *)rec_var_ptrs[0], (double *)rec_var_ptrs[1],
                         (double *)rec_var_ptrs[2], num_elements,
                         (double *)V_TOT_rec);
  }
  for (auto &metadata : refactored_metadata.metadata) {
    refactored_metadata.total_size += metadata.GetLoadedBitPlaneSizes();
  }

  // Bitrate
  unsigned long long int local_total_size = refactored_metadata.total_size;
  unsigned long long int global_total_size = 0;
  MPI_Reduce(&local_total_size, &global_total_size, 1, MPI_UNSIGNED_LONG_LONG,
             MPI_SUM, 0, MPI_COMM_WORLD);
  unsigned long long int local_in_size = in_size;
  unsigned long long int global_in_size = 0;
  MPI_Reduce(&local_in_size, &global_in_size, 1, MPI_UNSIGNED_LONG_LONG,
             MPI_SUM, 0, MPI_COMM_WORLD);
  if (dtype == mgard_x::data_type::Float) {
    bitrate = 32 / ((double)global_in_size / global_total_size);
  } else if (dtype == mgard_x::data_type::Double) {
    bitrate = 64 / ((double)global_in_size / global_total_size);
  }
  if (!rank)
    std::cout << "Bitrate = " << bitrate << std::endl; // MPI_REDUCE SUM

  // std::cout << "Original Vx[35345] = " << ((float*) org_Vx_ptr)[35345] << ",
  // Reconstructed Vx[35345] = " << ((float*) rec_var_ptrs[0])[35345] <<
  // std::endl;
  if (!rank)
    std::cout << "Requested_Tau = " << tau << std::endl;

  // Max_est_error
  double local_max_est_error = refactored_metadata.max_est_error;
  double global_max_est_error = 0;
  MPI_Reduce(&local_max_est_error, &global_max_est_error, 1, MPI_DOUBLE,
             MPI_MAX, 0, MPI_COMM_WORLD);
  if (!rank)
    std::cout << "Est_max_error = " << global_max_est_error << std::endl;

  // Max_real_error
  double local_real_max_error = (double)compute_max_abs_error(
      (float *)V_TOT_ori, (float *)V_TOT_rec, num_elements);
  double global_real_max_error = 0;
  MPI_Reduce(&local_real_max_error, &global_real_max_error, 1, MPI_DOUBLE,
             MPI_MAX, 0, MPI_COMM_WORLD);
  if (!rank)
    std::cout << "Real_max_error = " << global_real_max_error << std::endl;
  //  for (int i = 0; i < 8; ++i) {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (rank == i) {
  //       std::cout << "From Rank " << i << ": " << std::endl;
  //       int count = 0;
  //       for (auto &metadata : refactored_metadata.metadata){
  //         if (count == 0) std::cout << "Vx:" << std::endl;
  //         else if (count == 1) std::cout << "Vy:" << std::endl;
  //         else if (count == 2) std::cout << "Vz:" << std::endl;
  //         for (size_t i = 0; i < metadata.loaded_level_num_bitplanes.size();
  //         ++i) {
  //           std::cout << "Level " << i << ": bitplane = "
  //                     <<
  //                     static_cast<int>(metadata.loaded_level_num_bitplanes[i])
  //                     << std::endl;
  //         }
  //         count ++;
  //       }
  //   }
  //  }

  return 0;
}

bool try_refactoring(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-z", "--refactor"))
    return false;
  mgard_x::log::info("Mode: refactor", true);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::ostringstream oss;
  oss << rank;

  std::string input_file =
      get_arg<std::string>(argc, argv, "Original data", "-i", "--input");
  input_file += oss.str() + ".dat";
  //  std::cout << input_file << std::endl;
  std::string output_file =
      get_arg<std::string>(argc, argv, "Refactored data", "-o", "--output");
  output_file += oss.str();
  //  std::cout << output_file << std::endl;
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
  if (dtype == mgard_x::data_type::Double) {
    launch_refactor<double>(shape.size(), dtype, input_file.c_str(),
                            output_file.c_str(), shape, dev_type, verbose,
                            max_memory_footprint, rank);
  } else if (dtype == mgard_x::data_type::Float) {
    launch_refactor<float>(shape.size(), dtype, input_file.c_str(),
                           output_file.c_str(), shape, dev_type, verbose,
                           max_memory_footprint, rank);
  }
  return true;
}

bool try_reconstruction(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-x", "--reconstruct"))
    return false;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::ostringstream oss;
  // oss << "JHTDB_" << rank;
  oss << rank;

  mgard_x::log::info("mode: reconstruct", true);
  std::string input_file =
      get_arg<std::string>(argc, argv, "Refactored data", "-i", "--input");
  // if (!input_file.empty() && input_file.back() == '/') input_file +=
  // oss.str(); else input_file += "/" + oss.str();
  input_file += oss.str();
  //  std::cout << input_file << std::endl;
  std::string output_file =
      get_arg<std::string>(argc, argv, "Reconstructed data", "-o", "--output");
  // default is none (means original data not provided)
  std::string original_file = "none";
  enum mgard_x::data_type dtype;
  std::vector<mgard_x::SIZE> shape;
  if (has_arg(argc, argv, "-g", "--orignal")) {
    original_file =
        get_arg<std::string>(argc, argv, "Original data", "-g", "--orignal");
    // if (!original_file.empty() && original_file.back() == '/') original_file
    // += oss.str() + ".dat"; else original_file += "/" + oss.str() + ".dat";
    original_file += oss.str() + ".dat";
    //  std::cout << original_file << std::endl;
    dtype = get_data_type(argc, argv);
    shape = get_args<mgard_x::SIZE>(argc, argv, "Dimensions", "-dim",
                                    "--dimension");
  }
  // only abs mode is supported now
  enum mgard_x::error_bound_type mode =
      mgard_x::error_bound_type::ABS; // REL or ABS

  std::vector<double> tols;
  if (has_arg(argc, argv, "-e", "--error-bound")) {
    tols.push_back(
        get_arg<double>(argc, argv, "Error bound", "-e", "--error-bound"));
  } else if (has_arg(argc, argv, "-me", "--multi-error-bounds")) {
    tols = get_args<double>(argc, argv, "Multi error bounds", "-me",
                            "--multi-error-bounds");
  } else {
    throw std::runtime_error(
        "Missing option -e/--error-bound or -me/--multi-error-bounds");
  }
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
  int decrease_method;
  if (has_arg(argc, argv, "-dm", "--decrease-method")) {
    decrease_method =
        get_arg<int>(argc, argv, "Decrease method", "-dm", "--decrease-method");
  }
  launch_reconstruct(input_file, output_file, original_file, dtype, shape, tols,
                     s, mode, adaptive_resolution, dev_type, verbose, rank,
                     decrease_method);
  return true;
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  if (!try_refactoring(argc, argv) && !try_reconstruction(argc, argv)) {
    print_usage_message("");
  }

  MPI_Finalize();
  return 0;
}