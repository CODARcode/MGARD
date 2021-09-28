#include <cmath>
#include <cstddef>

#include <algorithm>
#include <functional>
#include <iostream>

#include "mgard/compress.hpp"

// Internal headers used in this example to compute the magnitude of the
// compression error.
#include "mgard/TensorNorms.hpp"
#include "mgard/shuffle.hpp"

// This program illustrates the use of MGARD's decompression API. We assume you
// have already gone through the compression example. This time, we will work
// with a function defined on the unit square.

// Number of samples in each dimension (square root of dataset size).
const std::size_t N = 500;

// Function to be sampled.
double f(const double x, const double y) {
  return std::abs(x - 0.7) + 2 * std::abs(y - 0.3);
}

int main() {
  // We again begin by constructing the mesh hierarchy. If no coordinate array
  // is given, MGARD will generate a uniform mesh (points equally spaced in each
  // dimension) on `[0, 1]^{d}`. Let's use the uniform mesh in this example.
  std::cout << "creating mesh hierarchy ...";
  const mgard::TensorMeshHierarchy<2, double> hierarchy({N, N});
  std::cout << " done" << std::endl;

  // Next we generate the dataset. Note that the samples should be stored in C
  // order (last dimension changing fastest).
  double *const u = new double[N * N];
  std::cout << "generating data ...";
  for (std::size_t i = 0; i < N; ++i) {
    const double x = static_cast<double>(i) / N;
    for (std::size_t j = 0; j < N; ++j) {
      const double y = static_cast<double>(j) / N;
      u[N * i + j] = f(x, y);
    }
  }
  std::cout << " done" << std::endl;

  // After compressing and decompressing we will want to measure the magnitude
  // of the compression error (the difference between the original and
  // decompressed datasets). `mgard::compress` modifies `u`, so we need to make
  // a copy before compressing.
  double *const u_copy = new double[N * N];
  std::copy(u, u + N * N, u_copy);

  // Again we set the compression parameters and compress. When `s > 0`, MGARD
  // gives greater weight to the high-frequency components of the data.
  const double s = 1;
  const double tolerance = 0.01;
  std::cout << "compressing ...";
  const mgard::CompressedDataset<2, double> compressed =
      mgard::compress(hierarchy, u, s, tolerance);
  std::cout << " done" << std::endl;
  delete[] u;

  // To decompress, just pass the `mgard::CompressedDataset` to
  // `mgard::decompress`.
  std::cout << "decompressing ...";
  const mgard::DecompressedDataset<2, double> decompressed =
      mgard::decompress(compressed);
  std::cout << " done" << std::endl;

  // Compression error.
  double *const error = new double[N * N];
  // The `data` member function returns a pointer to the decompressed dataset.
  std::transform(u_copy, u_copy + N * N, decompressed.data(), error,
                 std::minus<double>());
  delete[] u_copy;

  // To compute the magnitude of the compression error, we use the internal
  // functions `mgard::norm` and `mgard::shuffle`. Important: these functions
  // are not part of the API, so they should not be considered stable.
  double *const shuffled = new double[N * N];
  mgard::shuffle(hierarchy, error, shuffled);
  delete[] error;
  std::cout << "error tolerance: " << tolerance << std::endl
            << "achieved error: " << mgard::norm(hierarchy, shuffled, s)
            << std::endl;
  delete[] shuffled;
  return 0;
}
