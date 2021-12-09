/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_AUTOTUNER_H
#define MGARD_X_AUTOTUNER_H

namespace mgard_x {

constexpr int GPK_CONFIG[5][7][3] = {{{1, 1, 16},{1, 1, 32},{1, 1, 64},{1, 1, 128},{1, 1, 256},{1, 1, 512},{1, 1, 1024}},
                                   {{1, 2, 4},{1, 4, 4},{1, 4, 8},{1, 4, 16},{1, 4, 32},{1, 2, 64},{1, 2, 128}},
                                   {{2, 2, 2},{4, 4, 4},{4, 4, 8},{4, 4, 16},{4, 4, 32},{2, 2, 64},{2, 2, 128}},
                                   {{2, 2, 2},{4, 4, 4},{4, 4, 8},{4, 4, 16},{4, 4, 32},{2, 2, 64},{2, 2, 128}},
                                   {{2, 2, 2},{4, 4, 4},{4, 4, 8},{4, 4, 16},{4, 4, 32},{2, 2, 64},{2, 2, 128}}};

constexpr int LPK_CONFIG[5][7][3] = {{{1, 1, 8},{1, 1, 8},{1, 1, 8},{1, 1, 16},{1, 1, 32},{1, 1, 64},{1, 1, 128}},
                                     {{1, 2, 4},{1, 4, 4},{1, 8, 8},{1, 4, 16},{1, 2, 32},{1, 2, 64},{1, 2, 128}},
                                     {{2, 2, 2},{4, 4, 4},{8, 8, 8},{4, 4, 16},{2, 2, 32},{2, 2, 64},{2, 2, 128}},
                                     {{2, 2, 2},{4, 4, 4},{8, 8, 8},{4, 4, 16},{2, 2, 32},{2, 2, 64},{2, 2, 128}},
                                     {{2, 2, 2},{4, 4, 4},{8, 8, 8},{4, 4, 16},{2, 2, 32},{2, 2, 64},{2, 2, 128}}};

constexpr int IPK_CONFIG[5][7][4] = {{{1, 1, 8, 2},{1, 1, 8, 4},{1, 1, 8, 4},{1, 1, 16, 4},{1, 1, 32, 2},{1, 1, 64, 2},{1, 1, 128, 2}},
                                     {{1, 2, 4, 2},{1, 4, 4, 4},{1, 8, 8, 4},{1, 4, 16, 4},{1, 2, 32, 2},{1, 2, 64, 2},{1, 2, 128, 2}},
                                     {{2, 2, 2, 2},{4, 4, 4, 4},{8, 8, 8, 4},{4, 4, 16, 4},{2, 2, 32, 2},{2, 2, 64, 2},{2, 2, 128, 2}},
                                     {{2, 2, 2, 2},{4, 4, 4, 4},{8, 8, 8, 4},{4, 4, 16, 4},{2, 2, 32, 2},{2, 2, 64, 2},{2, 2, 128, 2}},
                                     {{2, 2, 2, 2},{4, 4, 4, 4},{8, 8, 8, 4},{4, 4, 16, 4},{2, 2, 32, 2},{2, 2, 64, 2},{2, 2, 128, 2}}};

constexpr int LWPK_CONFIG[5][3] = {{1, 1, 8},{1, 4, 4},{4, 4, 4},{4, 4, 4},{4, 4, 4}};


constexpr int LWQK_CONFIG[5][3] = {{1, 1, 64},{1, 4, 32},{4, 4, 16},{4, 4, 16},{4, 4, 16}};


const int tBLK_ENCODE = 256;
const int tBLK_DEFLATE = 128;
const int tBLK_CANONICAL = 128;


template<typename T>
MGARDX_CONT
int TypeToIdx() {
  if (std::is_same<T, float>::value) {
    return 0;
  } else if (std::is_same<T, double>::value) {
    return 1;
  }
}

template <typename DeviceType>
MGARDX_CONT void 
FillAutoTunerTable(std::string kernel_name, int precision_idx, int range_l){

    std::ifstream t("file.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::cout << "********************read file: " << buffer.str() << "\n";

    std::regex e("(int lpk.*\\{\n)((.*\\{.*\\}.*\n){0})(.*\\{(., ){3})(.)", std::regex_constants::ECMAScript);
    std::string new_string = std::regex_replace(buffer.str(), e, "$1$2$4 4");
    std::cout << "********************new file: " << new_string << "\n";

    std::regex e2(",  (.)", std::regex_constants::ECMAScript);
    std::string new_string2 = std::regex_replace(new_string, e2, ", $1");
    std::cout << "********************new file2: " << new_string2 << "\n";


    std::ofstream ofs("file.txt", std::ofstream::trunc);

    ofs << new_string2;

    ofs.close();



  }

template <typename DeviceType>
class KernelConfigs {
public:
  MGARDX_CONT
  KernelConfigs(){};
};

template <typename DeviceType>
class AutoTuningTable {
public:
  MGARDX_CONT
  AutoTuningTable(){};
};


template <typename DeviceType>
class AutoTuner {
public:
  MGARDX_CONT
  AutoTuner(){};

  static KernelConfigs<DeviceType> kernelConfigs;
  static AutoTuningTable<DeviceType> autoTuningTable;
  static bool ProfileKenrles;
};
}

#ifdef MGARDX_COMPILE_SERIAL 
#include "AutoTunerSerial.h"
#endif

#ifdef MGARDX_COMPILE_CUDA
#include "AutoTunerCuda.h"
#endif

#ifdef MGARDX_COMPILE_HIP
#include "AutoTunerHip.h"
#endif

#ifdef MGARDX_COMPILE_KOKKOS
#include "AutoTunerKokkos.h"
#endif

#endif