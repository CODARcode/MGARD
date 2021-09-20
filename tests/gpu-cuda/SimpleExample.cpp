#include "mgard/compress_cuda.hpp"
#include <iostream>
#include <vector>
int main() {
  mgard_cuda::SIZE n1 = 10;
  mgard_cuda::SIZE n2 = 20;
  mgard_cuda::SIZE n3 = 30;

  // prepare
  std::cout << "Preparing data...";
  double *in_array_cpu;
  mgard_cuda::cudaMallocHostHelper((void **)&in_array_cpu,
                                   sizeof(double) * n1 * n2 * n3);
  //... load data into in_array_cpu
  std::vector<mgard_cuda::SIZE> shape{n1, n2, n3};
  mgard_cuda::Handle<3, double> handle(shape);
  mgard_cuda::Array<3, double> in_array(shape);
  in_array.loadData(in_array_cpu);
  std::cout << "Done\n";

  std::cout << "Compressing with MGARD-GPU...";
  double tol = 0.01, s = 0;
  mgard_cuda::Array<1, unsigned char> compressed_array =
      mgard_cuda::compress(handle, in_array, mgard_cuda::REL, tol, s);
  size_t compressed_size =
      compressed_array.getShape()[0]; // compressed size in number of bytes.
  unsigned char *compressed_array_cpu = compressed_array.getDataHost();
  std::cout << "Done\n";

  std::cout << "Decompressing with MGARD-GPU...";
  // decompression
  mgard_cuda::Array<3, double> decompressed_array =
      mgard_cuda::decompress(handle, compressed_array);
  mgard_cuda::cudaFreeHostHelper(in_array_cpu);
  double *decompressed_array_cpu = decompressed_array.getDataHost();
  std::cout << "Done\n";
}