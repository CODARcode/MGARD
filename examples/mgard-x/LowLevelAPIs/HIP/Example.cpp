#include "mgard/compress_x_lowlevel.hpp"

#include <iostream>
#include <vector>
int main() {

  mgard_x::SIZE n1 = 10;
  mgard_x::SIZE n2 = 20;
  mgard_x::SIZE n3 = 30;

  // prepare
  std::cout << "Preparing data...";
  double *in_array_cpu = new double[n1 * n2 * n3];
  //... load data into in_array_cpu
  std::vector<mgard_x::SIZE> shape{n1, n2, n3};
  mgard_x::Hierarchy<3, double, mgard_x::HIP> hierarchy(shape);
  mgard_x::Array<3, double, mgard_x::HIP> in_array(shape);
  in_array.load(in_array_cpu);
  std::cout << "Done\n";

  std::cout << "Compressing with MGARD-X HIP backend...";
  double tol = 0.01, s = 0, norm;
  mgard_x::Config config;
  config.lossless = mgard_x::lossless_type::Huffman_Zstd;
  mgard_x::Array<1, unsigned char, mgard_x::HIP> compressed_array = mgard_x::compress(
      hierarchy, in_array, mgard_x::error_bound_type::REL, tol, s, norm, config);
  // Get compressed size in number of bytes.
  size_t compressed_size = compressed_array.shape()[0]; 
  unsigned char *compressed_array_cpu = compressed_array.hostCopy();
  std::cout << "Done\n";

  std::cout << "Decompressing with MGARD-X HIP backend...";
  // decompression
  mgard_x::Array<3, double, mgard_x::HIP> decompressed_array =
      mgard_x::decompress(hierarchy, compressed_array, mgard_x::error_bound_type::REL,
                          tol, s, norm, config);
  delete [] in_array_cpu;
  double *decompressed_array_cpu = decompressed_array.hostCopy();
  std::cout << "Done\n";
}