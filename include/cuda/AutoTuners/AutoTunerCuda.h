/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_AUTOTUNER_CUDA_H
#define MGARD_X_AUTOTUNER_CUDA_H

namespace mgard_x {


  constexpr int GPK_CONFIG[5][7][3] = {{{1, 1, 8},{1, 1, 8},{1, 1, 8},{1, 1, 16},{1, 1, 32},{1, 1, 64},{1, 1, 128}},
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


template <>
class KernelConfigs<CUDA> {
public:
  MGARDm_CONT
  KernelConfigs(){};



};

template <>
class AutoTuningTable<CUDA> {
public:
  MGARDm_CONT
  AutoTuningTable(){
    this->auto_tuning_cc = new int **[num_arch];
    this->auto_tuning_mr1 = new int **[num_arch];
    this->auto_tuning_mr2 = new int **[num_arch];
    this->auto_tuning_mr3 = new int **[num_arch];
    this->auto_tuning_ts1 = new int **[num_arch];
    this->auto_tuning_ts2 = new int **[num_arch];
    this->auto_tuning_ts3 = new int **[num_arch];
    for (int i = 0; i < num_arch; i++) {
      this->auto_tuning_cc[i] = new int *[num_precision];
      this->auto_tuning_mr1[i] = new int *[num_precision];
      this->auto_tuning_mr2[i] = new int *[num_precision];
      this->auto_tuning_mr3[i] = new int *[num_precision];
      this->auto_tuning_ts1[i] = new int *[num_precision];
      this->auto_tuning_ts2[i] = new int *[num_precision];
      this->auto_tuning_ts3[i] = new int *[num_precision];
      for (int j = 0; j < num_precision; j++) {
        this->auto_tuning_cc[i][j] = new int[num_range];
        this->auto_tuning_mr1[i][j] = new int[num_range];
        this->auto_tuning_mr2[i][j] = new int[num_range];
        this->auto_tuning_mr3[i][j] = new int[num_range];
        this->auto_tuning_ts1[i][j] = new int[num_range];
        this->auto_tuning_ts2[i][j] = new int[num_range];
        this->auto_tuning_ts3[i][j] = new int[num_range];
      }
    }

    // Default
    for (int i = 0; i < num_arch; i++) {
      for (int j = 0; j < num_precision; j++) {
        for (int k = 0; k < num_range; k++) {
          this->auto_tuning_cc[i][j][k] = 0;
          this->auto_tuning_mr1[i][j][k] = 0;
          this->auto_tuning_mr2[i][j][k] = 0;
          this->auto_tuning_mr3[i][j][k] = 0;
          this->auto_tuning_ts1[i][j][k] = 0;
          this->auto_tuning_ts2[i][j][k] = 0;
          this->auto_tuning_ts3[i][j][k] = 0;
        }
      }
    }

    // Volta-Single
    this->auto_tuning_cc[1][0][0] = 1;
    this->auto_tuning_cc[1][0][1] = 1;
    this->auto_tuning_cc[1][0][2] = 1;
    this->auto_tuning_cc[1][0][3] = 1;
    this->auto_tuning_cc[1][0][4] = 1;
    this->auto_tuning_cc[1][0][5] = 5;
    this->auto_tuning_cc[1][0][6] = 5;
    this->auto_tuning_cc[1][0][7] = 5;
    this->auto_tuning_cc[1][0][8] = 5;

    this->auto_tuning_mr1[1][0][0] = 1;
    this->auto_tuning_mr2[1][0][0] = 1;
    this->auto_tuning_mr3[1][0][0] = 1;
    this->auto_tuning_mr1[1][0][1] = 1;
    this->auto_tuning_mr2[1][0][1] = 1;
    this->auto_tuning_mr3[1][0][1] = 1;
    this->auto_tuning_mr1[1][0][2] = 1;
    this->auto_tuning_mr2[1][0][2] = 1;
    this->auto_tuning_mr3[1][0][2] = 1;
    this->auto_tuning_mr1[1][0][3] = 3;
    this->auto_tuning_mr2[1][0][3] = 3;
    this->auto_tuning_mr3[1][0][3] = 3;
    this->auto_tuning_mr1[1][0][4] = 4;
    this->auto_tuning_mr2[1][0][4] = 1;
    this->auto_tuning_mr3[1][0][4] = 3;
    this->auto_tuning_mr1[1][0][5] = 5;
    this->auto_tuning_mr2[1][0][5] = 3;
    this->auto_tuning_mr3[1][0][5] = 3;
    this->auto_tuning_mr1[1][0][6] = 5;
    this->auto_tuning_mr2[1][0][6] = 4;
    this->auto_tuning_mr3[1][0][6] = 4;
    this->auto_tuning_mr1[1][0][7] = 3;
    this->auto_tuning_mr2[1][0][7] = 4;
    this->auto_tuning_mr3[1][0][7] = 4;
    this->auto_tuning_mr1[1][0][8] = 3;
    this->auto_tuning_mr2[1][0][8] = 4;
    this->auto_tuning_mr3[1][0][8] = 4;

    this->auto_tuning_ts1[1][0][0] = 1;
    this->auto_tuning_ts2[1][0][0] = 1;
    this->auto_tuning_ts3[1][0][0] = 1;
    this->auto_tuning_ts1[1][0][1] = 1;
    this->auto_tuning_ts2[1][0][1] = 1;
    this->auto_tuning_ts3[1][0][1] = 1;
    this->auto_tuning_ts1[1][0][2] = 2;
    this->auto_tuning_ts2[1][0][2] = 2;
    this->auto_tuning_ts3[1][0][2] = 2;
    this->auto_tuning_ts1[1][0][3] = 3;
    this->auto_tuning_ts2[1][0][3] = 2;
    this->auto_tuning_ts3[1][0][3] = 2;
    this->auto_tuning_ts1[1][0][4] = 3;
    this->auto_tuning_ts2[1][0][4] = 2;
    this->auto_tuning_ts3[1][0][4] = 2;
    this->auto_tuning_ts1[1][0][5] = 3;
    this->auto_tuning_ts2[1][0][5] = 2;
    this->auto_tuning_ts3[1][0][5] = 2;
    this->auto_tuning_ts1[1][0][6] = 5;
    this->auto_tuning_ts2[1][0][6] = 3;
    this->auto_tuning_ts3[1][0][6] = 2;
    this->auto_tuning_ts1[1][0][7] = 5;
    this->auto_tuning_ts2[1][0][7] = 6;
    this->auto_tuning_ts3[1][0][7] = 5;
    this->auto_tuning_ts1[1][0][8] = 5;
    this->auto_tuning_ts2[1][0][8] = 6;
    this->auto_tuning_ts3[1][0][8] = 5;
    // Volta-Double

    this->auto_tuning_cc[1][1][0] = 1;
    this->auto_tuning_cc[1][1][1] = 1;
    this->auto_tuning_cc[1][1][2] = 1;
    this->auto_tuning_cc[1][1][3] = 1;
    this->auto_tuning_cc[1][1][4] = 4;
    this->auto_tuning_cc[1][1][5] = 5;
    this->auto_tuning_cc[1][1][6] = 6;
    this->auto_tuning_cc[1][1][7] = 6;
    this->auto_tuning_cc[1][1][8] = 5;

    this->auto_tuning_mr1[1][1][0] = 1;
    this->auto_tuning_mr2[1][1][0] = 1;
    this->auto_tuning_mr3[1][1][0] = 1;
    this->auto_tuning_mr1[1][1][1] = 1;
    this->auto_tuning_mr2[1][1][1] = 1;
    this->auto_tuning_mr3[1][1][1] = 1;
    this->auto_tuning_mr1[1][1][2] = 1;
    this->auto_tuning_mr2[1][1][2] = 1;
    this->auto_tuning_mr3[1][1][2] = 1;
    this->auto_tuning_mr1[1][1][3] = 1;
    this->auto_tuning_mr2[1][1][3] = 3;
    this->auto_tuning_mr3[1][1][3] = 1;
    this->auto_tuning_mr1[1][1][4] = 4;
    this->auto_tuning_mr2[1][1][4] = 3;
    this->auto_tuning_mr3[1][1][4] = 3;
    this->auto_tuning_mr1[1][1][5] = 5;
    this->auto_tuning_mr2[1][1][5] = 5;
    this->auto_tuning_mr3[1][1][5] = 5;
    this->auto_tuning_mr1[1][1][6] = 4;
    this->auto_tuning_mr2[1][1][6] = 6;
    this->auto_tuning_mr3[1][1][6] = 6;
    this->auto_tuning_mr1[1][1][7] = 6;
    this->auto_tuning_mr2[1][1][7] = 6;
    this->auto_tuning_mr3[1][1][7] = 5;
    this->auto_tuning_mr1[1][1][8] = 6;
    this->auto_tuning_mr2[1][1][8] = 6;
    this->auto_tuning_mr3[1][1][8] = 5;

    this->auto_tuning_ts1[1][1][0] = 1;
    this->auto_tuning_ts2[1][1][0] = 1;
    this->auto_tuning_ts3[1][1][0] = 1;
    this->auto_tuning_ts1[1][1][1] = 1;
    this->auto_tuning_ts2[1][1][1] = 1;
    this->auto_tuning_ts3[1][1][1] = 1;
    this->auto_tuning_ts1[1][1][2] = 2;
    this->auto_tuning_ts2[1][1][2] = 2;
    this->auto_tuning_ts3[1][1][2] = 2;
    this->auto_tuning_ts1[1][1][3] = 3;
    this->auto_tuning_ts2[1][1][3] = 2;
    this->auto_tuning_ts3[1][1][3] = 2;
    this->auto_tuning_ts1[1][1][4] = 3;
    this->auto_tuning_ts2[1][1][4] = 2;
    this->auto_tuning_ts3[1][1][4] = 2;
    this->auto_tuning_ts1[1][1][5] = 4;
    this->auto_tuning_ts2[1][1][5] = 2;
    this->auto_tuning_ts3[1][1][5] = 2;
    this->auto_tuning_ts1[1][1][6] = 5;
    this->auto_tuning_ts2[1][1][6] = 5;
    this->auto_tuning_ts3[1][1][6] = 2;
    this->auto_tuning_ts1[1][1][7] = 5;
    this->auto_tuning_ts2[1][1][7] = 6;
    this->auto_tuning_ts3[1][1][7] = 6;
    this->auto_tuning_ts1[1][1][8] = 5;
    this->auto_tuning_ts2[1][1][8] = 6;
    this->auto_tuning_ts3[1][1][8] = 6;

    // Turing-Single
    this->auto_tuning_cc[2][0][0] = 1;
    this->auto_tuning_cc[2][0][1] = 1;
    this->auto_tuning_cc[2][0][2] = 1;
    this->auto_tuning_cc[2][0][3] = 1;
    this->auto_tuning_cc[2][0][4] = 3;
    this->auto_tuning_cc[2][0][5] = 5;
    this->auto_tuning_cc[2][0][6] = 5;
    this->auto_tuning_cc[2][0][7] = 5;
    this->auto_tuning_cc[2][0][8] = 4;

    this->auto_tuning_mr1[2][0][0] = 1;
    this->auto_tuning_mr2[2][0][0] = 1;
    this->auto_tuning_mr3[2][0][0] = 1;
    this->auto_tuning_mr1[2][0][1] = 1;
    this->auto_tuning_mr2[2][0][1] = 1;
    this->auto_tuning_mr3[2][0][1] = 1;
    this->auto_tuning_mr1[2][0][2] = 1;
    this->auto_tuning_mr2[2][0][2] = 1;
    this->auto_tuning_mr3[2][0][2] = 1;
    this->auto_tuning_mr1[2][0][3] = 1;
    this->auto_tuning_mr2[2][0][3] = 1;
    this->auto_tuning_mr3[2][0][3] = 3;
    this->auto_tuning_mr1[2][0][4] = 4;
    this->auto_tuning_mr2[2][0][4] = 3;
    this->auto_tuning_mr3[2][0][4] = 4;
    this->auto_tuning_mr1[2][0][5] = 4;
    this->auto_tuning_mr2[2][0][5] = 3;
    this->auto_tuning_mr3[2][0][5] = 3;
    this->auto_tuning_mr1[2][0][6] = 6;
    this->auto_tuning_mr2[2][0][6] = 3;
    this->auto_tuning_mr3[2][0][6] = 3;
    this->auto_tuning_mr1[2][0][7] = 5;
    this->auto_tuning_mr2[2][0][7] = 4;
    this->auto_tuning_mr3[2][0][7] = 4;
    this->auto_tuning_mr1[2][0][8] = 5;
    this->auto_tuning_mr2[2][0][8] = 4;
    this->auto_tuning_mr3[2][0][8] = 4;

    this->auto_tuning_ts1[2][0][0] = 1;
    this->auto_tuning_ts2[2][0][0] = 1;
    this->auto_tuning_ts3[2][0][0] = 1;
    this->auto_tuning_ts1[2][0][1] = 1;
    this->auto_tuning_ts2[2][0][1] = 1;
    this->auto_tuning_ts3[2][0][1] = 1;
    this->auto_tuning_ts1[2][0][2] = 2;
    this->auto_tuning_ts2[2][0][2] = 2;
    this->auto_tuning_ts3[2][0][2] = 2;
    this->auto_tuning_ts1[2][0][3] = 3;
    this->auto_tuning_ts2[2][0][3] = 2;
    this->auto_tuning_ts3[2][0][3] = 2;
    this->auto_tuning_ts1[2][0][4] = 3;
    this->auto_tuning_ts2[2][0][4] = 2;
    this->auto_tuning_ts3[2][0][4] = 2;
    this->auto_tuning_ts1[2][0][5] = 3;
    this->auto_tuning_ts2[2][0][5] = 2;
    this->auto_tuning_ts3[2][0][5] = 2;
    this->auto_tuning_ts1[2][0][6] = 5;
    this->auto_tuning_ts2[2][0][6] = 5;
    this->auto_tuning_ts3[2][0][6] = 2;
    this->auto_tuning_ts1[2][0][7] = 5;
    this->auto_tuning_ts2[2][0][7] = 6;
    this->auto_tuning_ts3[2][0][7] = 6;
    this->auto_tuning_ts1[2][0][8] = 5;
    this->auto_tuning_ts2[2][0][8] = 6;
    this->auto_tuning_ts3[2][0][8] = 6;
    // Turing-Double

    this->auto_tuning_cc[2][1][0] = 0;
    this->auto_tuning_cc[2][1][1] = 0;
    this->auto_tuning_cc[2][1][2] = 2;
    this->auto_tuning_cc[2][1][3] = 2;
    this->auto_tuning_cc[2][1][4] = 3;
    this->auto_tuning_cc[2][1][5] = 4;
    this->auto_tuning_cc[2][1][6] = 4;
    this->auto_tuning_cc[2][1][7] = 6;
    this->auto_tuning_cc[2][1][8] = 3;

    this->auto_tuning_mr1[2][1][0] = 1;
    this->auto_tuning_mr2[2][1][0] = 1;
    this->auto_tuning_mr3[2][1][0] = 1;
    this->auto_tuning_mr1[2][1][1] = 1;
    this->auto_tuning_mr2[2][1][1] = 1;
    this->auto_tuning_mr3[2][1][1] = 1;
    this->auto_tuning_mr1[2][1][2] = 1;
    this->auto_tuning_mr2[2][1][2] = 1;
    this->auto_tuning_mr3[2][1][2] = 1;
    this->auto_tuning_mr1[2][1][3] = 1;
    this->auto_tuning_mr2[2][1][3] = 1;
    this->auto_tuning_mr3[2][1][3] = 1;
    this->auto_tuning_mr1[2][1][4] = 4;
    this->auto_tuning_mr2[2][1][4] = 4;
    this->auto_tuning_mr3[2][1][4] = 1;
    this->auto_tuning_mr1[2][1][5] = 1;
    this->auto_tuning_mr2[2][1][5] = 1;
    this->auto_tuning_mr3[2][1][5] = 1;
    this->auto_tuning_mr1[2][1][6] = 1;
    this->auto_tuning_mr2[2][1][6] = 1;
    this->auto_tuning_mr3[2][1][6] = 1;
    this->auto_tuning_mr1[2][1][7] = 1;
    this->auto_tuning_mr2[2][1][7] = 1;
    this->auto_tuning_mr3[2][1][7] = 1;
    this->auto_tuning_mr1[2][1][8] = 1;
    this->auto_tuning_mr2[2][1][8] = 1;
    this->auto_tuning_mr3[2][1][8] = 1;

    this->auto_tuning_ts1[2][1][0] = 1;
    this->auto_tuning_ts2[2][1][0] = 1;
    this->auto_tuning_ts3[2][1][0] = 1;
    this->auto_tuning_ts1[2][1][1] = 1;
    this->auto_tuning_ts2[2][1][1] = 1;
    this->auto_tuning_ts3[2][1][1] = 1;
    this->auto_tuning_ts1[2][1][2] = 2;
    this->auto_tuning_ts2[2][1][2] = 2;
    this->auto_tuning_ts3[2][1][2] = 2;
    this->auto_tuning_ts1[2][1][3] = 3;
    this->auto_tuning_ts2[2][1][3] = 2;
    this->auto_tuning_ts3[2][1][3] = 2;
    this->auto_tuning_ts1[2][1][4] = 2;
    this->auto_tuning_ts2[2][1][4] = 2;
    this->auto_tuning_ts3[2][1][4] = 2;
    this->auto_tuning_ts1[2][1][5] = 2;
    this->auto_tuning_ts2[2][1][5] = 2;
    this->auto_tuning_ts3[2][1][5] = 2;
    this->auto_tuning_ts1[2][1][6] = 3;
    this->auto_tuning_ts2[2][1][6] = 5;
    this->auto_tuning_ts3[2][1][6] = 3;
    this->auto_tuning_ts1[2][1][7] = 3;
    this->auto_tuning_ts2[2][1][7] = 6;
    this->auto_tuning_ts3[2][1][7] = 6;
    this->auto_tuning_ts1[2][1][8] = 3;
    this->auto_tuning_ts2[2][1][8] = 6;
    this->auto_tuning_ts3[2][1][8] = 6;

  }

  MGARDm_CONT
  ~AutoTuningTable() {
    for (int i = 0; i < num_arch; i++) {
      for (int j = 0; j < num_precision; j++) {
        delete[] this->auto_tuning_cc[i][j];
        delete[] this->auto_tuning_mr1[i][j];
        delete[] this->auto_tuning_mr2[i][j];
        delete[] this->auto_tuning_mr3[i][j];
        delete[] this->auto_tuning_ts1[i][j];
        delete[] this->auto_tuning_ts2[i][j];
        delete[] this->auto_tuning_ts3[i][j];
      }
      delete[] this->auto_tuning_cc[i];
      delete[] this->auto_tuning_mr1[i];
      delete[] this->auto_tuning_mr2[i];
      delete[] this->auto_tuning_mr3[i];
      delete[] this->auto_tuning_ts1[i];
      delete[] this->auto_tuning_ts2[i];
      delete[] this->auto_tuning_ts3[i];
    }
    delete[] this->auto_tuning_cc;
    delete[] this->auto_tuning_mr1;
    delete[] this->auto_tuning_mr2;
    delete[] this->auto_tuning_mr3;
    delete[] this->auto_tuning_ts1;
    delete[] this->auto_tuning_ts2;
    delete[] this->auto_tuning_ts3;
  }

  int num_arch = 3;
  int num_precision = 2;
  int num_range = 9;
  int ***auto_tuning_cc;
  int ***auto_tuning_mr1, ***auto_tuning_ts1;
  int ***auto_tuning_mr2, ***auto_tuning_ts2;
  int ***auto_tuning_mr3, ***auto_tuning_ts3;
  int arch, precision;

};

template<typename T>
MGARDm_CONT
int TypeToIdx() {
  if (std::is_same<T, float>::value) {
    return 0;
  } else if (std::is_same<T, double>::value) {
    return 1;
  }
}


template <>
class AutoTuner<CUDA> {
public:
  MGARDm_CONT
  AutoTuner(){};

  static KernelConfigs<CUDA> kernelConfigs;
  static AutoTuningTable<CUDA> autoTuningTable;
  static bool ProfileKernels;
};

}

#endif