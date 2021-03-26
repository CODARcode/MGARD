#include <cmath>
#include <cstddef>

#include <iostream>

#include "mgard/mgard_api.h"

const std::size_t N = 1000;

float f(const float x) { return 5 * std::sin(3 * x) - 2 * std::cos(7 * x); }

int main() {
  float *const u = static_cast<float *>(std::malloc(N * sizeof(*u)));
  std::cout << "generating data ...";
  for (std::size_t i = 0; i < N; ++i) {
    u[i] = static_cast<float>(7 * i) / N;
  }
  std::cout << " done" << std::endl;
  std::cout << "creating mesh hierarchy ...";
  const mgard::TensorMeshHierarchy<1, float> hierarchy({N});
  std::cout << " done" << std::endl;
  const float s = 0;
  const float tolerance = 0.000001;
  std::cout << "compressing ...";
  const mgard::CompressedDataset<1, float> compressed =
      mgard::compress(hierarchy, u, s, tolerance);
  std::cout << " done" << std::endl;
  std::cout << "compression ratio: "
            << static_cast<float>(N * sizeof(*u)) / compressed.size()
            << std::endl;
  std::free(u);
  return 0;
}
