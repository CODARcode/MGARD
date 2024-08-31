#include "mgard/compress_x.hpp"

#include <iostream>
// #include <vector>

int main() {

  mgard_x::SIZE n1 = 10;
  mgard_x::SIZE n2 = 20;
  mgard_x::SIZE n3 = 30;

  // prepare
  std::cout << "Preparing...";
  double *in_array_cpu = new double[n1 * n2 * n3];
  void *compressed_array_cpu = NULL;
  size_t compressed_size;
  //... load data into in_array_cpu
  std::vector<mgard_x::SIZE> shape{n1, n2, n3};
  double tol = 0.01, s = 0, norm;
  std::cout << "Done\n";

  std::cout << "Compressing with MGARD-X High level API with CUDA backend...";
  mgard_x::Config config;
  config.lossless = mgard_x::lossless_type::Huffman_Zstd;
  config.dev_type = mgard_x::device_type::CUDA;
  mgard_x::compress(3, mgard_x::data_type::Double, shape, tol, s,
                    mgard_x::error_bound_type::REL, in_array_cpu,
                    compressed_array_cpu, compressed_size, config, false);
  std::cout << "Done\n";

  std::cout
      << "Decompressing with MGARD-X High level API with SERIAL backend...";
  // decompression
  void *decompressed_array_cpu = NULL;
  config.dev_type = mgard_x::device_type::SERIAL;
  mgard_x::decompress(compressed_array_cpu, compressed_size,
                      decompressed_array_cpu, config, false);

  delete[] in_array_cpu;
  std::cout << "Done\n";
}