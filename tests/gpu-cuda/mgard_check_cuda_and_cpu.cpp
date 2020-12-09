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

#include "mgard_api.h"
#include "mgard_api_cuda.h"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

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
  double tol, s = 0;
  if (argc == 6) {
    data_srouce = 0;
    nrow = atoi(argv[1]);
    ncol = atoi(argv[2]);
    nfib = atoi(argv[3]);
    tol = atof(argv[4]);
    opt = atoi(argv[5]);
  } else if (argc == 7) {
    data_srouce = 1;
    infile = argv[1];
    nrow = atoi(argv[2]);
    ncol = atoi(argv[3]);
    nfib = atoi(argv[4]);
    tol = atof(argv[5]);
    opt = atoi(argv[6]);
  } else {
    fprintf(stderr, "%s: Wrong arguments! ", argv[0]);
    print_usage_message(argv, stderr);
    print_for_more_details_message(argv, stderr);
    return 1;
  }

  long lSize;
  double *buffer;
  long num_doubles;

  num_doubles = nrow * ncol * nfib;
  long num_bytes = sizeof(double) * num_doubles;
  lSize = num_bytes;

  buffer = (double *)malloc(sizeof(char) * lSize);
  if (buffer == NULL) {
    fputs("Memory error", stderr);
    exit(2);
  }

  if (data_srouce == 0) {
    fprintf(stdout,
            "No input file provided. Generating random data for testing\n");
    for (int i = 0; i < num_doubles; i++) {
      buffer[i] = rand() % 10 + 1;
    }
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

    result = fread(buffer, 1, lSize, pFile);
    if (result != lSize) {
      fputs("Reading error", stderr);
      exit(3);
    }
    fclose(pFile);
  }

  double data_L_inf_norm = 0;
  for (int i = 0; i < num_doubles; ++i) {
    double temp = fabs(buffer[i]);
    if (temp > data_L_inf_norm)
      data_L_inf_norm = temp;
  }

  double *in_buff;
  mgard_cuda::cudaMallocHostHelper((void **)&in_buff,
                                   sizeof(double) * num_doubles);
  memcpy(in_buff, buffer, sizeof(double) * num_doubles);

  int iflag = 1; // 0 -> float, 1 -> double
  int out_size;

  unsigned char *mgard_comp_buff;
  const double *mgard_out_buff;

  if (opt == -1) {
    fprintf(stdout, "[INFO] Compressing using CPU only\n");
    if (nfib == 1) {
      const std::array<std::size_t, 2> shape = {nrow, ncol};
      const mgard::TensorMeshHierarchy<2, double> hierarchy(shape);
      const size_t ndof = hierarchy.ndof();
      const mgard::CompressedDataset<2, double> compressed =
          mgard::compress(hierarchy, in_buff, 0.0, tol);
      out_size = compressed.size();
      mgard_comp_buff = (unsigned char *)compressed.data();
      mgard::DecompressedDataset<2, double> decompressed =
          mgard::decompress(compressed);
      mgard_out_buff = decompressed.data();
    } else {
      const std::array<std::size_t, 3> shape = {nrow, ncol, nfib};
      const mgard::TensorMeshHierarchy<3, double> hierarchy(shape);
      const size_t ndof = hierarchy.ndof();
      const mgard::CompressedDataset<3, double> compressed =
          mgard::compress(hierarchy, in_buff, 0.0, tol);
      out_size = compressed.size();
      mgard_comp_buff = (unsigned char *)compressed.data();
      mgard::DecompressedDataset<3, double> decompressed =
          mgard::decompress(compressed);
      mgard_out_buff = decompressed.data();
    }
    // mgard_comp_buff = mgard_compress(in_buff, out_size, nrow, ncol, nfib,
    // tol);
  } else {
    fprintf(stdout, "[INFO] Compressing with GPU acceleration\n");
    mgard_cuda_handle<double> handle(nrow, ncol, nfib, B, num_of_queues, opt);
    mgard_comp_buff = mgard_compress_cuda(handle, in_buff, out_size, tol);
    mgard_out_buff = mgard_decompress_cuda(handle, mgard_comp_buff, out_size);
  }
  mgard_cuda::cudaFreeHostHelper(in_buff);

  printf("[INFO] In size:  %10ld  Out size: %10d  Compression ratio: %10ld \n",
         lSize, out_size, lSize / out_size);

  double error_L_inf_norm = 0;
  for (int i = 0; i < num_doubles; ++i) {
    double temp = fabs(buffer[i] - mgard_out_buff[i]);
    if (temp > error_L_inf_norm)
      error_L_inf_norm = temp;
  }
  double relative_L_inf_error = error_L_inf_norm / data_L_inf_norm;

  // Maximum length (plus one for terminating byte) of norm name.
  size_t N = 10;
  char norm_name[N];
  int num_chars_written;
  if (isinf(s)) {
    num_chars_written = snprintf(norm_name, N, "L^infty");
  } else if (s == 0) {
    num_chars_written = snprintf(norm_name, N, "L^2");
  } else {
    num_chars_written = snprintf(norm_name, N, "H^%.1f", s);
  }
  if (num_chars_written <= 0 || num_chars_written >= N) {
    norm_name[0] = '?';
    norm_name[1] = 0;
  }
  printf("[INFO] Rel. %s error tolerance: %10.5E \n", norm_name, tol);
  printf("[INFO] Rel. L^infty error: %10.5E \n", relative_L_inf_error);

  if (relative_L_inf_error < tol) {
    printf(ANSI_GREEN "[INFO] SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    return 0;
  } else {
    printf(ANSI_RED "[INFO] FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
    return 1;
  }
}
