// Copyright 2017, Brown University, Providence, RI.
//
//                         All Rights Reserved
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose other than its incorporation into a
// commercial product or service is hereby granted without fee, provided
// that the above copyright notice appear in all copies and that both
// that copyright notice and this permission notice appear in supporting
// documentation, and that the name of Brown University not be used in
// advertising or publicity pertaining to distribution of the software
// without specific, written prior permission.
//
// BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
// INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
// PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
// ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
//
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.1
//
// This file is part of MGARD.
//
// MGARD is distributed under the OSI-approved Apache License, Version 2.0.
// See accompanying file Copyright.txt for details.
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <chrono>

#include "mgard_api_cuda.h"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

using namespace std::chrono;

void print_usage_message(char *argv[], FILE *fp) {
  fprintf(fp,
          "Usage: %s infile nrow ncol nfib tolerance opt (-1: CPU; 0: CUDA, 1: "
          "CUDA-optimized)\n",
          argv[0]);
}

void print_for_more_details_message(char *argv[], FILE *fp) {
  fprintf(fp, "\nFor more details, run: %s --help\n", argv[0]);
}

void print_help_message(char *argv[], FILE *fp) {
  fprintf(fp, "\nThe input file `infile` should contain a "
              "double[`nrow`][`ncol`][`nfib`] array.\n"
              "The array will be compressed so that the error as measured in "
              "the H^`s` norm is\n"
              "no more than `tolerance`. \n");
}

int main(int argc, char *argv[]) {
  size_t result;

  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    print_usage_message(argv, stdout);
    print_help_message(argv, stdout);
    return 0;
  }

  int data_srouce; // 0: generate random data; 1: input file
  char *infile, *outfile;
  int nrow, ncol, nfib, opt, B = 16, num_of_queues = 32;
  std::vector<size_t> shape;
  double tol, s = 0;

  int i = 1;
  data_srouce = atoi(argv[i++]);
  if (data_srouce) {
    infile = argv[i++];
    printf("Input data: %s ", infile);
  } else {
    printf("Input data: random generated ");
  }
  int D = atoi(argv[i++]);
  printf(" shape: %d ( ", D);
  for (int d = 0; d < D; d++) {
    shape.push_back( atoi(argv[i++]));
    printf("%d ", shape[shape.size()-1]);
  }
  printf(")\n");
  tol = atof(argv[i++]);
  printf("Rel. error bound: %.2e ", tol);
  s = atof(argv[i++]);
  printf("S: %.2f\n", s);
  opt = atoi(argv[i++]);
  printf("Optimization: %d\n", opt);


  long lSize;
  long num_doubles;

  num_doubles = 1;
  for (int d = 0; d < shape.size(); d++) { num_doubles *= shape[d]; }
  long num_bytes = sizeof(double) * num_doubles;
  lSize = num_bytes;

  double *in_buff;
  mgard_cuda::cudaMallocHostHelper((void **)&in_buff,
                                   sizeof(double) * num_doubles);
  if (in_buff == NULL) {
    fputs("Memory error", stderr);
    exit(2);
  }

  if (data_srouce == 0) {
    fprintf(stdout,
            "No input file provided. Generating random data for testing\n");
    for (int i = 0; i < num_doubles; i++) {
      in_buff[i] = rand() % 10 + 1;
    }
    // printf("num_doubles %d\n", num_doubles);
    
    fprintf(stdout,
            "Done Generating data.\n");
  } else {
    fprintf(stdout, "Loading file: %s\n", infile);
    FILE *pFile;
    pFile = fopen(infile, "rb");
    if (pFile == NULL) {
      fputs("File error", stderr);
      exit(1);
    }
    fseek(pFile, 0, SEEK_END);
    long lSize = ftell(pFile);

    rewind(pFile);

    lSize = num_bytes;

    if (lSize != num_bytes) {
      fprintf(stderr,
              "%s contains %lu bytes when %lu were expected. Exiting.\n",
              infile, lSize, num_bytes);
      return 1;
    }

    result = fread(in_buff, 1, lSize, pFile);
    if (result != lSize) {
      fputs("Reading error", stderr);
      exit(3);
    }
    fclose(pFile);
  }

  double data_L_inf_norm = 0;
  for (int i = 0; i < num_doubles; ++i) {
    double temp = fabs(in_buff[i]);
    if (temp > data_L_inf_norm)
      data_L_inf_norm = temp;
  }

  size_t out_size;
  unsigned char *mgard_comp_buff;
  double *mgard_out_buff;

  printf("Start compressing and decompressing with GPU\n");
  if (D == 1) {
    mgard_cuda_handle<double, 1> handle(shape);
    mgard_comp_buff = mgard_compress_cuda(handle, in_buff, out_size, tol, s);
    mgard_out_buff = mgard_decompress_cuda(handle, mgard_comp_buff, out_size);
  } else if (D == 2) {
    mgard_cuda_handle<double, 2> handle(shape);
    mgard_comp_buff = mgard_compress_cuda(handle, in_buff, out_size, tol, s);
    mgard_out_buff = mgard_decompress_cuda(handle, mgard_comp_buff, out_size);
  } else if (D == 3) {
    mgard_cuda_handle<double, 3> handle(shape);
    mgard_comp_buff = mgard_compress_cuda(handle, in_buff, out_size, tol, s);
    mgard_out_buff = mgard_decompress_cuda(handle, mgard_comp_buff, out_size);
  } else if (D == 4) {
   mgard_cuda_handle<double, 4> handle(shape);
    mgard_comp_buff = mgard_compress_cuda(handle, in_buff, out_size, tol, s);
    mgard_out_buff = mgard_decompress_cuda(handle, mgard_comp_buff, out_size);
  }


  printf("In size:  %10ld  Out size: %10d  Compression ratio: %10ld \n",
           lSize, out_size, lSize / out_size);

  // FILE *qfile;
  // qfile = fopen ( outfile , "wb" );
  // result = fwrite (mgard_out_buff, 1, lSize, qfile);
  // fclose(qfile);
  // if (result != lSize) {fputs ("Writing error",stderr); exit (4);}
  int error_count = 100;
  double error_L_inf_norm = 0;
  double sum = 0;
  for (int i = 0; i < num_doubles; ++i) {
    double temp = fabs(in_buff[i] - mgard_out_buff[i]);
    if (temp > error_L_inf_norm)
      error_L_inf_norm = temp;
    if (temp / data_L_inf_norm >= tol && error_count) {
      printf("not bounded: buffer[%d]: %f vs. mgard_out_buff[%d]: %f \n", i, in_buff[i], i, mgard_out_buff[i]);
      error_count --;
    }
    sum += temp* temp;
  }

  mgard_cuda::cudaFreeHostHelper(in_buff);

  // printf("sum: %e\n", sum/num_doubles);
  double relative_L_inf_error = error_L_inf_norm / data_L_inf_norm;

  // std::ofstream fout("mgard_out.dat", std::ios::binary);
  // fout.write(reinterpret_cast<const char *>(mgard_comp_buff), out_size);
  // fout.close();

  printf("Rel. L^infty error bound: %10.5E \n", tol);
  printf("Rel. L^infty error: %10.5E \n", relative_L_inf_error);

  if (relative_L_inf_error < tol) {
    printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    return 0;
  } else {
    printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
    return -1;
  }



}
