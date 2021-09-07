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

int main(int argc, char *argv[]) {
  size_t result;

  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    print_usage_message(argv, stdout);
    return 0;
  }

  int data_srouce; // 0: generate random data; 1: input file
  char *infile;    //, *outfile;
  std::vector<size_t> shape;
  float tol, s = 0;
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
  printf("Rel. error bound: %.2e ", tol);
  s = atof(argv[i++]);
  printf("S: %.2f\n", s);
  device = atoi(argv[i++]);
  if (device == 0)
    printf("Use: CPU\n");
  if (device == 1)
    printf("Use: GPU\n");

  size_t lSize;
  long num_float;

  num_float = 1;
  for (size_t d = 0; d < shape.size(); d++) {
    num_float *= shape[d];
  }
  size_t num_bytes = sizeof(float) * num_float;
  lSize = num_bytes;

  float *in_buff;
  mgard_cuda::cudaMallocHostHelper((void **)&in_buff,
                                   sizeof(float) * num_float);
  if (in_buff == NULL) {
    fputs("Memory error", stderr);
    exit(2);
  }

  if (data_srouce == 0) {
    fprintf(stdout,
            "No input file provided. Generating random data for testing\n");
    for (int i = 0; i < num_float; i++) {
      in_buff[i] = rand() % 10 + 1;
    }
    // printf("num_float %d\n", num_float);

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

  float data_L_inf_norm = 0;
  for (int i = 0; i < num_float; ++i) {
    float temp = fabs(in_buff[i]);
    if (temp > data_L_inf_norm)
      data_L_inf_norm = temp;
  }

  size_t out_size;
  float *mgard_out_buff = new float[num_float];

  printf("Start compressing and decompressing with GPU\n");
  if (D == 1) {
    if (device == 0) {
      const mgard::TensorMeshHierarchy<1, float> hierarchy({shape[0]});
      mgard::CompressedDataset<1, float> compressed_dataset =
          mgard::compress(hierarchy, in_buff, s, data_L_inf_norm * tol);
      out_size = compressed_dataset.size();
      mgard::DecompressedDataset<1, float> decompressed_dataset =
          mgard::decompress(compressed_dataset);
      memcpy(mgard_out_buff, decompressed_dataset.data(),
             num_float * sizeof(float));
    } else {
      mgard_cuda::Array<1, float> in_array(shape);
      in_array.loadData(in_buff);
      mgard_cuda::Array<1, float> test = in_array;
      mgard_cuda::Handle<1, float> handle(shape);
      mgard_cuda::Array<1, unsigned char> compressed_array =
          mgard_cuda::compress(handle, in_array, mgard_cuda::REL, tol, s);
      out_size = compressed_array.getShape()[0];
      mgard_cuda::Array<1, float> out_array =
          mgard_cuda::decompress(handle, compressed_array);
      memcpy(mgard_out_buff, out_array.getDataHost(),
             num_float * sizeof(float));
    }
  } else if (D == 2) {
    if (device == 0) {
      const mgard::TensorMeshHierarchy<2, float> hierarchy(
          {shape[1], shape[0]});
      mgard::CompressedDataset<2, float> compressed_dataset =
          mgard::compress(hierarchy, in_buff, s, data_L_inf_norm * tol);
      out_size = compressed_dataset.size();
      mgard::DecompressedDataset<2, float> decompressed_dataset =
          mgard::decompress(compressed_dataset);
      memcpy(mgard_out_buff, decompressed_dataset.data(),
             num_float * sizeof(float));
    } else {
      mgard_cuda::Array<2, float> in_array(shape);
      in_array.loadData(in_buff);
      mgard_cuda::Handle<2, float> handle(shape);
      mgard_cuda::Array<1, unsigned char> compressed_array =
          mgard_cuda::compress(handle, in_array, mgard_cuda::REL, tol, s);
      out_size = compressed_array.getShape()[0];
      mgard_cuda::Array<2, float> out_array =
          mgard_cuda::decompress(handle, compressed_array);
      memcpy(mgard_out_buff, out_array.getDataHost(),
             num_float * sizeof(float));
    }
  } else if (D == 3) {
    if (device == 0) {
      const mgard::TensorMeshHierarchy<3, float> hierarchy(
          {shape[2], shape[1], shape[0]});
      mgard::CompressedDataset<3, float> compressed_dataset =
          mgard::compress(hierarchy, in_buff, s, data_L_inf_norm * tol);
      out_size = compressed_dataset.size();
      mgard::DecompressedDataset<3, float> decompressed_dataset =
          mgard::decompress(compressed_dataset);
      memcpy(mgard_out_buff, decompressed_dataset.data(),
             num_float * sizeof(float));
    } else {
      mgard_cuda::Array<3, float> in_array(shape);
      in_array.loadData(in_buff);
      mgard_cuda::Handle<3, float> handle(shape);
      mgard_cuda::Array<1, unsigned char> compressed_array =
          mgard_cuda::compress(handle, in_array, mgard_cuda::REL, tol, s);
      out_size = compressed_array.getShape()[0];
      mgard_cuda::Array<3, float> out_array =
          mgard_cuda::decompress(handle, compressed_array);
      memcpy(mgard_out_buff, out_array.getDataHost(),
             num_float * sizeof(float));
    }
  } else if (D == 4) {
    if (device == 0) {
      const mgard::TensorMeshHierarchy<4, float> hierarchy(
          {shape[3], shape[2], shape[1], shape[0]});
      mgard::CompressedDataset<4, float> compressed_dataset =
          mgard::compress(hierarchy, in_buff, s, data_L_inf_norm * tol);
      out_size = compressed_dataset.size();
      mgard::DecompressedDataset<4, float> decompressed_dataset =
          mgard::decompress(compressed_dataset);
      memcpy(mgard_out_buff, decompressed_dataset.data(),
             num_float * sizeof(float));
    } else {
      mgard_cuda::Array<4, float> in_array(shape);
      in_array.loadData(in_buff);
      mgard_cuda::Handle<4, float> handle(shape);
      mgard_cuda::Array<1, unsigned char> compressed_array =
          mgard_cuda::compress(handle, in_array, mgard_cuda::REL, tol, s);
      out_size = compressed_array.getShape()[0];
      mgard_cuda::Array<4, float> out_array =
          mgard_cuda::decompress(handle, compressed_array);
      memcpy(mgard_out_buff, out_array.getDataHost(),
             num_float * sizeof(float));
    }
  }

  printf("In size:  %10ld  Out size: %10ld  Compression ratio: %10ld \n", lSize,
         out_size, lSize / out_size);

  // FILE *qfile;
  // qfile = fopen ( outfile , "wb" );
  // result = fwrite (mgard_out_buff, 1, lSize, qfile);
  // fclose(qfile);
  // if (result != lSize) {fputs ("Writing error",stderr); exit (4);}
  int error_count = 100;
  float error_L_inf_norm = 0;
  for (int i = 0; i < num_float; ++i) {
    float temp = fabs(in_buff[i] - mgard_out_buff[i]);
    if (temp > error_L_inf_norm)
      error_L_inf_norm = temp;
    if (temp / data_L_inf_norm >= tol && error_count) {
      printf("not bounded: buffer[%d]: %f vs. mgard_out_buff[%d]: %f \n", i,
             in_buff[i], i, mgard_out_buff[i]);
      error_count--;
    }
  }

  float max = 0, min = std::numeric_limits<float>::max(), range = 0;
  float error_sum = 0, mse = 0, psnr = 0;
  for (int i = 0; i < num_float; ++i) {
    if (max < in_buff[i])
      max = in_buff[i];
    if (min > in_buff[i])
      min = in_buff[i];
    float err = fabs(in_buff[i] - mgard_out_buff[i]);
    error_sum += err * err;
  }
  range = max - min;
  mse = error_sum / num_float;
  psnr = 20 * log10(range) - 10 * log10(mse);

  mgard_cuda::cudaFreeHostHelper(in_buff);

  // printf("sum: %e\n", sum/num_float);
  float relative_L_inf_error = error_L_inf_norm / data_L_inf_norm;

  // std::ofstream fout("mgard_out.dat", std::ios::binary);
  // fout.write(reinterpret_cast<const char *>(mgard_comp_buff), out_size);
  // fout.close();

  printf("Rel. L^infty error bound: %10.5E \n", tol);
  printf("Rel. L^infty error: %10.5E \n", relative_L_inf_error);
  printf("MSE: %10.5E\n", mse);
  printf("PSNR: %10.5E\n", psnr);

  if (relative_L_inf_error < tol) {
    printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    return 0;
  } else {
    printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
    return -1;
  }
  delete[] mgard_out_buff;
}
