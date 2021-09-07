/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <chrono>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgard_api.h"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

using namespace std::chrono;

void print_usage_message(char *argv[], FILE *fp) {
  fprintf(fp,
          "Usage: %s [input file] [num. of dimensions] [1st dim.] [2nd dim.] "
          "[3rd. dim] ... [tolerance] [s]\n",
          argv[0]);
}

double L_inf_norm(size_t num_double, double *in_buff) {
  double L_inf = 0;
  for (int i = 0; i < num_double; ++i) {
    double temp = fabs(in_buff[i]);
    if (temp > L_inf)
      L_inf = temp;
  }
  return L_inf;
}

double L_2_norm(size_t num_double, double *in_buff) {
  double L_2 = 0;
  for (int i = 0; i < num_double; ++i) {
    double temp = fabs(in_buff[i]);
    L_2 += temp * temp;
  }
  return std::sqrt(L_2);
}

double L_inf_error(size_t num_double, double *in_buff, double *mgard_out_buff) {
  double error_L_inf_norm = 0;
  for (int i = 0; i < num_double; ++i) {
    double temp = fabs(in_buff[i] - mgard_out_buff[i]);
    if (temp > error_L_inf_norm)
      error_L_inf_norm = temp;
  }
  return error_L_inf_norm;
}

double L_2_error(size_t num_double, double *in_buff, double *mgard_out_buff) {
  double error_L_2_norm = 0;
  for (int i = 0; i < num_double; ++i) {
    double temp = fabs(in_buff[i] - mgard_out_buff[i]);
    error_L_2_norm += temp * temp;
  }
  return std::sqrt(error_L_2_norm);
}

int main(int argc, char *argv[]) {
  size_t result;

  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    print_usage_message(argv, stdout);
    return 0;
  }

  int data_srouce; // 0: generate random data; 1: input file
  char *infile;    //, *outfile;
  std::vector<size_t> shape;
  double tol, s = 0;
  int device = 0; // 0 CPU; 1 GPU

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
    shape.push_back(atoi(argv[i++]));
    printf("%lu ", shape[shape.size() - 1]);
  }
  printf(")\n");
  tol = atof(argv[i++]);
  printf("Abs. error bound: %.2e ", tol);
  s = atof(argv[i++]);
  printf("S: %.2f\n", s);
  device = atoi(argv[i++]);
  if (device == 0)
    printf("Use: CPU\n");
  if (device == 1)
    printf("Use: GPU\n");

  size_t lSize;
  long num_double;

  num_double = 1;
  for (size_t d = 0; d < shape.size(); d++) {
    num_double *= shape[d];
  }
  size_t num_bytes = sizeof(double) * num_double;
  lSize = num_bytes;

  double *in_buff;
  mgard_cuda::cudaMallocHostHelper((void **)&in_buff,
                                   sizeof(double) * num_double);
  if (in_buff == NULL) {
    fputs("Memory error", stderr);
    exit(2);
  }

  if (data_srouce == 0) {
    fprintf(stdout,
            "No input file provided. Generating random data for testing\n");
    for (int i = 0; i < num_double; i++) {
      in_buff[i] = rand() % 10 + 1;
    }
    // printf("num_double %d\n", num_double);

    fprintf(stdout, "Done Generating data.\n");
  } else {
    fprintf(stdout, "Loading file: %s\n", infile);
    FILE *pFile;
    pFile = fopen(infile, "rb");
    if (pFile == NULL) {
      fputs("File error", stderr);
      exit(1);
    }
    fseek(pFile, 0, SEEK_END);
    size_t lSize = ftell(pFile);

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

  size_t out_size;
  double *mgard_out_buff = new double[num_double];
  printf("Start compressing and decompressing\n");
  if (D == 1) {
    if (device == 0) {
      const mgard::TensorMeshHierarchy<1, double> hierarchy({shape[0]});
      mgard::CompressedDataset<1, double> compressed_dataset =
          mgard::compress(hierarchy, in_buff, s, tol);
      out_size = compressed_dataset.size();
      mgard::DecompressedDataset<1, double> decompressed_dataset =
          mgard::decompress(compressed_dataset);
      memcpy(mgard_out_buff, decompressed_dataset.data(),
             num_double * sizeof(double));
    } else {
      mgard_cuda::Array<1, double> in_array(shape);
      in_array.loadData(in_buff);
      mgard_cuda::Handle<1, double> handle(shape);
      mgard_cuda::Array<1, unsigned char> compressed_array =
          mgard_cuda::compress(handle, in_array, mgard_cuda::ABS, tol, s);
      out_size = compressed_array.getShape()[0];
      mgard_cuda::Array<1, double> out_array =
          mgard_cuda::decompress(handle, compressed_array);
      memcpy(mgard_out_buff, out_array.getDataHost(),
             num_double * sizeof(double));
    }
  } else if (D == 2) {
    if (device == 0) {
      const mgard::TensorMeshHierarchy<2, double> hierarchy(
          {shape[1], shape[0]});
      mgard::CompressedDataset<2, double> compressed_dataset =
          mgard::compress(hierarchy, in_buff, s, tol);
      out_size = compressed_dataset.size();
      mgard::DecompressedDataset<2, double> decompressed_dataset =
          mgard::decompress(compressed_dataset);
      memcpy(mgard_out_buff, decompressed_dataset.data(),
             num_double * sizeof(double));
    } else {
      mgard_cuda::Array<2, double> in_array(shape);
      in_array.loadData(in_buff);
      mgard_cuda::Handle<2, double> handle(shape);
      mgard_cuda::Array<1, unsigned char> compressed_array =
          mgard_cuda::compress(handle, in_array, mgard_cuda::ABS, tol, s);
      out_size = compressed_array.getShape()[0];
      mgard_cuda::Array<2, double> out_array =
          mgard_cuda::decompress(handle, compressed_array);
      memcpy(mgard_out_buff, out_array.getDataHost(),
             num_double * sizeof(double));
    }
  } else if (D == 3) {
    if (device == 0) {
      const mgard::TensorMeshHierarchy<3, double> hierarchy(
          {shape[2], shape[1], shape[0]});
      mgard::CompressedDataset<3, double> compressed_dataset =
          mgard::compress(hierarchy, in_buff, s, tol);
      out_size = compressed_dataset.size();
      mgard::DecompressedDataset<3, double> decompressed_dataset =
          mgard::decompress(compressed_dataset);
      memcpy(mgard_out_buff, decompressed_dataset.data(),
             num_double * sizeof(double));
    } else {
      mgard_cuda::Array<3, double> in_array(shape);
      in_array.loadData(in_buff);
      mgard_cuda::Handle<3, double> handle(shape);
      mgard_cuda::Array<1, unsigned char> compressed_array =
          mgard_cuda::compress(handle, in_array, mgard_cuda::ABS, tol, s);
      out_size = compressed_array.getShape()[0];
      mgard_cuda::Array<3, double> out_array =
          mgard_cuda::decompress(handle, compressed_array);
      memcpy(mgard_out_buff, out_array.getDataHost(),
             num_double * sizeof(double));
    }
  } else if (D == 4) {
    if (device == 0) {
      const mgard::TensorMeshHierarchy<4, double> hierarchy(
          {shape[3], shape[2], shape[1], shape[0]});
      mgard::CompressedDataset<4, double> compressed_dataset =
          mgard::compress(hierarchy, in_buff, s, tol);
      out_size = compressed_dataset.size();
      mgard::DecompressedDataset<4, double> decompressed_dataset =
          mgard::decompress(compressed_dataset);
      memcpy(mgard_out_buff, decompressed_dataset.data(),
             num_double * sizeof(double));
    } else {
      mgard_cuda::Array<4, double> in_array(shape);
      in_array.loadData(in_buff);
      mgard_cuda::Handle<4, double> handle(shape);
      mgard_cuda::Array<1, unsigned char> compressed_array =
          mgard_cuda::compress(handle, in_array, mgard_cuda::ABS, tol, s);
      out_size = compressed_array.getShape()[0];
      mgard_cuda::Array<4, double> out_array =
          mgard_cuda::decompress(handle, compressed_array);
      memcpy(mgard_out_buff, out_array.getDataHost(),
             num_double * sizeof(double));
    }
  }

  printf("In size:  %10ld  Out size: %10ld  Compression ratio: %10ld \n", lSize,
         out_size, lSize / out_size);

  // FILE *qfile;
  // qfile = fopen ( outfile , "wb" );
  // result = fwrite (mgard_out_buff, 1, lSize, qfile);
  // fclose(qfile);
  // if (result != lSize) {fputs ("Writing error",stderr); exit (4);}

  double max = 0, min = std::numeric_limits<double>::max(), range = 0;
  double error_sum = 0, mse = 0, psnr = 0;
  for (int i = 0; i < num_double; ++i) {
    if (max < in_buff[i])
      max = in_buff[i];
    if (min > in_buff[i])
      min = in_buff[i];
    double err = fabs(in_buff[i] - mgard_out_buff[i]);
    error_sum += err * err;
  }
  range = max - min;
  mse = error_sum / num_double;
  psnr = 20 * log10(range) - 10 * log10(mse);

  // printf("sum: %e\n", sum/num_double);
  double absolute_error = 0.0;

  // if (s == std::numeric_limits<double>::infinity()) {
  absolute_error = L_inf_error(num_double, in_buff, mgard_out_buff);
  printf("Abs. L^infty error bound: %10.5E \n", tol);
  printf("Abs. L^infty error: %10.5E \n", absolute_error);
  // }

  // if (s == 0) {
  //   absolute_error = L_2_error(num_double, in_buff, mgard_out_buff);
  //   printf("Abs. L^2 error bound: %10.5E \n", tol);
  //   printf("Abs. L^2 error: %10.5E \n", absolute_error);
  // }

  printf("MSE: %10.5E\n", mse);
  printf("PSNR: %10.5E\n", psnr);

  if (absolute_error < tol) {
    printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    return 0;
  } else {
    printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
    return -1;
  }
  delete[] mgard_out_buff;
  mgard_cuda::cudaFreeHostHelper(in_buff);
}
