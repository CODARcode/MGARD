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
#include <string_view>

namespace mgard_x {

const int tBLK_ENCODE = 256;
const int tBLK_DEFLATE = 256;
const int tBLK_CANONICAL = 128;

template <typename T> MGARDX_CONT constexpr int TypeToIdx() {
  if constexpr (std::is_same<T, float>::value) {
    return 0;
  } else if constexpr (std::is_same<T, double>::value) {
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

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void FillAutoTunerTable(std::string kernel_name, int config) {

  int precision_idx = TypeToIdx<T>();
  DIM dim_idx = D - 1;

  std::string device_type_string = "";
  if (std::is_same<DeviceType, SERIAL>::value) {
    device_type_string = "Serial";
  } else if (std::is_same<DeviceType, OPENMP>::value) {
    device_type_string = "Openmp";
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

  string intput_dir_path = curr_dir_path;

  // std::cout << "********************intput_dir_path: " + intput_dir_path <<
  // "\n";

  std::string extension = ".h";

  std::string input_file =
      intput_dir_path + "/AutoTuner" + device_type_string + extension;

  // std::cout << "********************intput_file: " + input_file << "\n";

  std::string regex1_string = "(static constexpr int " + kernel_name +
                              ".*\\{)((.*\\{.*\\}.*\n){" +
                              std::to_string(precision_idx) + "})(.*\\{(., ){" +
                              std::to_string(dim_idx) + "})(.)";

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

template <typename DeviceType> class AutoTuningTable {
public:
  MGARDX_CONT
  AutoTuningTable(){};
};

template <typename DeviceType> class AutoTuner {
public:
  MGARDX_CONT
  AutoTuner(){};

  static AutoTuningTable<DeviceType> autoTuningTable;
  static bool ProfileKenrles;
  static bool WriteToTable;
};

template <typename DeviceType> void BeginAutoTuning() {
  AutoTuner<DeviceType>::ProfileKernels = true;
  AutoTuner<DeviceType>::WriteToTable = true;
}

template <typename DeviceType> void EndAutoTuning() {
  AutoTuner<DeviceType>::ProfileKernels = false;
  AutoTuner<DeviceType>::WriteToTable = false;
}

struct ExecutionConfig {
  constexpr ExecutionConfig(SIZE z, SIZE y, SIZE x) : z(z), y(y), x(x) {}
  SIZE x, y, z;
};

template <DIM D>
MGARDX_CONT constexpr ExecutionConfig GetExecutionConfig(int config_idx) {

  constexpr int CONFIG_CANDIDATE[5][7][3] = {{{1, 1, 16},
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
  ExecutionConfig config(CONFIG_CANDIDATE[D - 1][config_idx][0],
                         CONFIG_CANDIDATE[D - 1][config_idx][1],
                         CONFIG_CANDIDATE[D - 1][config_idx][2]);
  return config;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT constexpr ExecutionConfig
GetExecutionConfig(std::string_view functor_name) {
  int precision_idx = TypeToIdx<T>();
  DIM dim_idx = D - 1;
  int config_idx = 0;
  if (functor_name == "gpk_reo_3d") {
    config_idx =
        AutoTuningTable<DeviceType>::gpk_reo_3d[precision_idx][dim_idx];
  } else if (functor_name == "gpk_rev_3d") {
    config_idx =
        AutoTuningTable<DeviceType>::gpk_rev_3d[precision_idx][dim_idx];
  } else if (functor_name == "gpk_reo_nd") {
    config_idx =
        AutoTuningTable<DeviceType>::gpk_reo_nd[precision_idx][dim_idx];
  } else if (functor_name == "gpk_rev_nd") {
    config_idx =
        AutoTuningTable<DeviceType>::gpk_rev_nd[precision_idx][dim_idx];
  } else if (functor_name == "lpk1_3d") {
    config_idx = AutoTuningTable<DeviceType>::lpk1_3d[precision_idx][dim_idx];
  } else if (functor_name == "lpk2_3d") {
    config_idx = AutoTuningTable<DeviceType>::lpk2_3d[precision_idx][dim_idx];
  } else if (functor_name == "lpk3_3d") {
    config_idx = AutoTuningTable<DeviceType>::lpk3_3d[precision_idx][dim_idx];
  } else if (functor_name == "lpk1_nd") {
    config_idx = AutoTuningTable<DeviceType>::lpk1_nd[precision_idx][dim_idx];
  } else if (functor_name == "lpk2_nd") {
    config_idx = AutoTuningTable<DeviceType>::lpk2_nd[precision_idx][dim_idx];
  } else if (functor_name == "lpk3_nd") {
    config_idx = AutoTuningTable<DeviceType>::lpk3_nd[precision_idx][dim_idx];
  } else if (functor_name == "ipk1_3d") {
    config_idx = AutoTuningTable<DeviceType>::ipk1_3d[precision_idx][dim_idx];
  } else if (functor_name == "ipk2_3d") {
    config_idx = AutoTuningTable<DeviceType>::ipk2_3d[precision_idx][dim_idx];
  } else if (functor_name == "ipk3_3d") {
    config_idx = AutoTuningTable<DeviceType>::ipk3_3d[precision_idx][dim_idx];
  } else if (functor_name == "ipk1_nd") {
    config_idx = AutoTuningTable<DeviceType>::ipk1_nd[precision_idx][dim_idx];
  } else if (functor_name == "ipk2_nd") {
    config_idx = AutoTuningTable<DeviceType>::ipk2_nd[precision_idx][dim_idx];
  } else if (functor_name == "ipk3_nd") {
    config_idx = AutoTuningTable<DeviceType>::ipk3_nd[precision_idx][dim_idx];
  } else if (functor_name == "lwpk") {
    config_idx = AutoTuningTable<DeviceType>::lwpk[precision_idx][dim_idx];
  } else if (functor_name == "lwqzk") {
    config_idx = AutoTuningTable<DeviceType>::lwqzk[precision_idx][dim_idx];
  } else if (functor_name == "lwdqzk") {
    config_idx = AutoTuningTable<DeviceType>::lwdqzk[precision_idx][dim_idx];
  } else if (functor_name == "llk") {
    config_idx = AutoTuningTable<DeviceType>::llk[precision_idx][dim_idx];
  } else if (functor_name == "sdck") {
    config_idx = AutoTuningTable<DeviceType>::sdck[precision_idx][dim_idx];
  } else if (functor_name == "sdmtk") {
    config_idx = AutoTuningTable<DeviceType>::sdmtk[precision_idx][dim_idx];
  } else if (functor_name == "encode") {
    config_idx = AutoTuningTable<DeviceType>::encode[precision_idx][dim_idx];
  } else if (functor_name == "deflate") {
    config_idx = AutoTuningTable<DeviceType>::deflate[precision_idx][dim_idx];
  } else if (functor_name == "decode") {
    config_idx = AutoTuningTable<DeviceType>::decode[precision_idx][dim_idx];
  } else {
    log::err("Wrong functor_name");
    config_idx = 0;
  }
  return GetExecutionConfig<D>(config_idx);
}

} // namespace mgard_x

#include "AutoTunerCuda.h"
#include "AutoTunerHip.h"
#include "AutoTunerOpenmp.h"
#include "AutoTunerSerial.h"
#include "AutoTunerSycl.h"

#endif