/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_AUTOTUNER_H
#define MGARD_X_AUTOTUNER_H

#include <fstream>
#include <memory>
#include <regex>

namespace mgard_x {

constexpr int GPK_CONFIG[5][7][3] = {{{1, 1, 16},
                                      {1, 1, 32},
                                      {1, 1, 64},
                                      {1, 1, 128},
                                      {1, 1, 256},
                                      {1, 1, 512},
                                      {1, 1, 1024}},
                                     {{1, 2, 4},
                                      {1, 4, 4},
                                      {1, 4, 8},
                                      {1, 4, 16},
                                      {1, 4, 32},
                                      {1, 2, 64},
                                      {1, 2, 128}},
                                     {{2, 2, 2},
                                      {4, 4, 4},
                                      {4, 4, 8},
                                      {4, 4, 16},
                                      {4, 4, 32},
                                      {2, 2, 64},
                                      {2, 2, 128}},
                                     {{2, 2, 2},
                                      {4, 4, 4},
                                      {4, 4, 8},
                                      {4, 4, 16},
                                      {4, 4, 32},
                                      {2, 2, 64},
                                      {2, 2, 128}},
                                     {{2, 2, 2},
                                      {4, 4, 4},
                                      {4, 4, 8},
                                      {4, 4, 16},
                                      {4, 4, 32},
                                      {2, 2, 64},
                                      {2, 2, 128}}};

constexpr int LPK_CONFIG[5][7][3] = {{{1, 1, 8},
                                      {1, 1, 8},
                                      {1, 1, 8},
                                      {1, 1, 16},
                                      {1, 1, 32},
                                      {1, 1, 64},
                                      {1, 1, 128}},
                                     {{1, 2, 4},
                                      {1, 4, 4},
                                      {1, 8, 8},
                                      {1, 4, 16},
                                      {1, 2, 32},
                                      {1, 2, 64},
                                      {1, 2, 128}},
                                     {{2, 2, 2},
                                      {4, 4, 4},
                                      {8, 8, 8},
                                      {4, 4, 16},
                                      {2, 2, 32},
                                      {2, 2, 64},
                                      {2, 2, 128}},
                                     {{2, 2, 2},
                                      {4, 4, 4},
                                      {8, 8, 8},
                                      {4, 4, 16},
                                      {2, 2, 32},
                                      {2, 2, 64},
                                      {2, 2, 128}},
                                     {{2, 2, 2},
                                      {4, 4, 4},
                                      {8, 8, 8},
                                      {4, 4, 16},
                                      {2, 2, 32},
                                      {2, 2, 64},
                                      {2, 2, 128}}};

constexpr int IPK_CONFIG[5][7][4] = {{{1, 1, 8, 2},
                                      {1, 1, 8, 4},
                                      {1, 1, 8, 4},
                                      {1, 1, 16, 4},
                                      {1, 1, 32, 2},
                                      {1, 1, 64, 2},
                                      {1, 1, 128, 2}},
                                     {{1, 2, 4, 2},
                                      {1, 4, 4, 4},
                                      {1, 8, 8, 4},
                                      {1, 4, 16, 4},
                                      {1, 2, 32, 2},
                                      {1, 2, 64, 2},
                                      {1, 2, 128, 2}},
                                     {{2, 2, 2, 2},
                                      {4, 4, 4, 4},
                                      {8, 8, 8, 4},
                                      {4, 4, 16, 4},
                                      {2, 2, 32, 2},
                                      {2, 2, 64, 2},
                                      {2, 2, 128, 2}},
                                     {{2, 2, 2, 2},
                                      {4, 4, 4, 4},
                                      {8, 8, 8, 4},
                                      {4, 4, 16, 4},
                                      {2, 2, 32, 2},
                                      {2, 2, 64, 2},
                                      {2, 2, 128, 2}},
                                     {{2, 2, 2, 2},
                                      {4, 4, 4, 4},
                                      {8, 8, 8, 4},
                                      {4, 4, 16, 4},
                                      {2, 2, 32, 2},
                                      {2, 2, 64, 2},
                                      {2, 2, 128, 2}}};

constexpr int LWPK_CONFIG[5][7][3] = {{{1, 1, 64},
                                       {1, 1, 64},
                                       {1, 1, 64},
                                       {1, 1, 64},
                                       {1, 1, 128},
                                       {1, 1, 256},
                                       {1, 1, 512}},
                                      {{1, 4, 16},
                                       {1, 4, 16},
                                       {1, 4, 16},
                                       {1, 4, 32},
                                       {1, 4, 64},
                                       {1, 4, 128},
                                       {1, 4, 256}},
                                      {{4, 4, 4},
                                       {4, 4, 8},
                                       {4, 4, 16},
                                       {4, 4, 32},
                                       {4, 4, 64},
                                       {4, 4, 64},
                                       {4, 4, 64}},
                                      {{4, 4, 4},
                                       {4, 4, 8},
                                       {4, 4, 16},
                                       {4, 4, 32},
                                       {4, 4, 64},
                                       {4, 4, 64},
                                       {4, 4, 64}},
                                      {{4, 4, 4},
                                       {4, 4, 8},
                                       {4, 4, 16},
                                       {4, 4, 32},
                                       {4, 4, 64},
                                       {4, 4, 64},
                                       {4, 4, 64}}};

constexpr int LWQK_CONFIG[5][3] = {
    {1, 1, 64}, {1, 4, 32}, {4, 4, 16}, {4, 4, 16}, {4, 4, 16}};

const int tBLK_ENCODE = 256;
const int tBLK_DEFLATE = 128;
const int tBLK_CANONICAL = 128;

template <typename T> MGARDX_CONT int TypeToIdx() {
  if (std::is_same<T, float>::value) {
    return 0;
  } else if (std::is_same<T, double>::value) {
    return 1;
  } else {
    return 0;
  }
}

template <typename... Args>
std::string format(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

template <typename DeviceType>
MGARDX_CONT void FillAutoTunerTable(std::string kernel_name, int precision_idx,
                                    int range_l, int config) {

  std::string device_type_string = "";
  if (std::is_same<DeviceType, SERIAL>::value) {
    device_type_string = "Serial";
  } else if (std::is_same<DeviceType, CUDA>::value) {
    device_type_string = "Cuda";
  } else if (std::is_same<DeviceType, HIP>::value) {
    device_type_string = "Hip";
  } else if (std::is_same<DeviceType, SYCL>::value) {
    device_type_string = "Sycl";
  } else {
    std::cout << log::log_err << "invalid device_type in FillAutoTunerTable.\n";
    exit(-1);
  }

  string curr_file_path = __FILE__;
  string curr_dir_path = curr_file_path.substr(0, curr_file_path.rfind("/"));

  // std::cout << "********************curr_dir_path: " + curr_dir_path << "\n";

  string intput_dir_path =
      curr_dir_path + "/../../../../src/mgard-x/RuntimeX/AutoTuners";

  // std::cout << "********************intput_dir_path: " + intput_dir_path <<
  // "\n";

  std::string extension;
  if (std::is_same<DeviceType, CUDA>::value) {
    extension = ".cu";
  } else {
    extension = ".cpp";
  }

  std::string input_file =
      intput_dir_path + "/AutoTuner" + device_type_string + extension;

  // std::cout << "********************intput_file: " + input_file << "\n";

  std::string regex1_string = "(int AutoTuningTable<.*>::" + kernel_name +
                              ".*\\{)((.*\\{.*\\}.*\n){" +
                              std::to_string(precision_idx) + "})(.*\\{(., ){" +
                              std::to_string(range_l) + "})(.)";

  std::string replace1_string = "$1$2$4 " + std::to_string(config);
  std::string regex2_string = ",  (.)";
  std::string replace2_string = ", $1";
  std::string regex3_string = "\\{ ";
  std::string replace3_string = "{";

  std::ifstream t(input_file);
  std::stringstream buffer;
  buffer << t.rdbuf();
  // std::cout << "********************read file: " << buffer.str() << "\n";

  std::regex e1(regex1_string, std::regex_constants::ECMAScript);
  std::string new_string1 =
      std::regex_replace(buffer.str(), e1, replace1_string);
  // std::cout << "********************new file: " << new_string1 << "\n";

  std::regex e2(regex2_string, std::regex_constants::ECMAScript);
  std::string new_string2 =
      std::regex_replace(new_string1, e2, replace2_string);
  // std::cout << "********************new file2: " << new_string2 << "\n";

  std::regex e3(regex3_string, std::regex_constants::ECMAScript);
  std::string new_string3 =
      std::regex_replace(new_string2, e3, replace3_string);
  // std::cout << "********************new file3: " << new_string3 << "\n";

  std::ofstream ofs(input_file, std::ofstream::trunc);

  ofs << new_string3;

  ofs.close();
}

template <typename DeviceType> class KernelConfigs {
public:
  MGARDX_CONT
  KernelConfigs(){};
};

template <typename DeviceType> class AutoTuningTable {
public:
  MGARDX_CONT
  AutoTuningTable(){};
};

template <typename DeviceType> class AutoTuner {
public:
  MGARDX_CONT
  AutoTuner(){};

  static KernelConfigs<DeviceType> kernelConfigs;
  static AutoTuningTable<DeviceType> autoTuningTable;
  static bool ProfileKenrles;
};

template <typename DeviceType> void BeginAutoTuning() {
  AutoTuner<DeviceType>::ProfileKernels = true;
}

template <typename DeviceType> void EndAutoTuning() {
  AutoTuner<DeviceType>::ProfileKernels = false;
}

} // namespace mgard_x

#include "AutoTunerSerial.h"
#include "AutoTunerOpenmp.h"
#include "AutoTunerCuda.h"
#include "AutoTunerHip.h"
#include "AutoTunerKokkos.h"
#include "AutoTunerSycl.h"

#endif