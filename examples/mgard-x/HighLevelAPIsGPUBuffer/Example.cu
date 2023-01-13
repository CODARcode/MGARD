#include "mgard/compress_x.hpp"

#include <iostream>
#include <vector>

int main() {

  mgard_x::SIZE n1 = 10;
  mgard_x::SIZE n2 = 20;
  mgard_x::SIZE n3 = 30;

  // prepare
  std::cout << "Preparing...";
  void *in_array_cpu = malloc(n1 * n2 * n3 * sizeof(double));
  for (int i = 0; i < n1 * n2 * n3; i++) ((double*)in_array_cpu)[i] = i;
  double *in_array_gpu = nullptr;
  cudaMalloc((void**)&in_array_gpu, n1 * n2 * n3 * sizeof(double));
  cudaMemcpy(in_array_gpu, in_array_cpu, n1 * n2 * n3 * sizeof(double), cudaMemcpyDefault);
  void *compressed_array_cpu = malloc(n1 * n2 * n3 * sizeof(double) + 1e6);
  void *compressed_array_gpu = nullptr;
  cudaMalloc((void**)&compressed_array_gpu, n1 * n2 * n3 * sizeof(double) + 1e6);
  void *decompressed_array_cpu = malloc(n1 * n2 * n3 * sizeof(double));
  void *decompressed_array_gpu = nullptr;
  cudaMalloc((void**)&decompressed_array_gpu, n1 * n2 * n3 * sizeof(double));

  mgard_x::Config config;
  config.lossless = mgard_x::lossless_type::Huffman;
  config.dev_type = mgard_x::device_type::CUDA;
  // config.log_level = mgard_x::log::ERR | mgard_x::log::INFO | mgard_x::log::TIME;

  size_t compressed_size = n1 * n2 * n3 * sizeof(double) + 1e6;
  //... load data into in_array_cpu
  std::vector<mgard_x::SIZE> shape{n1, n2, n3};
  double tol = 0.01, s = 0;
  std::cout << "Done\n";

  std::cout << "Compressing with MGARD-X High level API using CPU buffer input, CPU buffer output...";
  mgard_x::compress(3, mgard_x::data_type::Double, shape, tol, s,
                    mgard_x::error_bound_type::REL, in_array_cpu,
                    compressed_array_cpu, compressed_size, config, true);
  std::cout << "Done\n";
  std::cout
      << "Decompressing with MGARD-X High level API using CPU buffer input, CPU buffer output...";
  mgard_x::decompress(compressed_array_cpu, compressed_size,
                      decompressed_array_cpu, config, true);
  std::cout << "Done\n";

  std::cout << "Compressing with MGARD-X High level API using GPU buffer input, GPU buffer output...";
  mgard_x::compress(3, mgard_x::data_type::Double, shape, tol, s,
                    mgard_x::error_bound_type::REL, in_array_gpu,
                    compressed_array_gpu, compressed_size, config, true);

  std::cout << "Done\n";
  std::cout
      << "Decompressing with MGARD-X High level API using GPU buffer input, GPU buffer output...";
  mgard_x::decompress(compressed_array_gpu, compressed_size,
                      decompressed_array_gpu, config, true);
  std::cout << "Done\n";

  std::cout << "Compressing with MGARD-X High level API using GPU buffer input, CPU buffer output...";
  mgard_x::compress(3, mgard_x::data_type::Double, shape, tol, s,
                    mgard_x::error_bound_type::REL, in_array_gpu,
                    compressed_array_cpu, compressed_size, config, true);

  std::cout << "Done\n";
  std::cout
      << "Decompressing with MGARD-X High level API using CPU buffer input, GPU buffer output...";
  mgard_x::decompress(compressed_array_cpu, compressed_size,
                      decompressed_array_gpu, config, true);
  std::cout << "Done\n";

  std::cout << "Compressing with MGARD-X High level API using CPU buffer input, GPU buffer output...";
  mgard_x::compress(3, mgard_x::data_type::Double, shape, tol, s,
                    mgard_x::error_bound_type::REL, in_array_cpu,
                    compressed_array_gpu, compressed_size, config, true);

  std::cout << "Done\n";
  std::cout
      << "Decompressing with MGARD-X High level API using GPU buffer input, CPU buffer output...";
  mgard_x::decompress(compressed_array_gpu, compressed_size,
                      decompressed_array_cpu, config, true);
  std::cout << "Done\n";

  return 0;
}