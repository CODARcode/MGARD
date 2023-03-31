#include <cmath>
#include <cstddef>

#include <array>
#include <iostream>
#include <stdexcept>

#include "mgard/TensorQuantityOfInterest.hpp"
#include "mgard/compress.hpp"

class AverageFunctional {
public:
  AverageFunctional(const std::array<std::size_t, 2> lower_left,
                    const std::array<std::size_t, 2> upper_right)
      : lower_left(lower_left), upper_right(upper_right) {
    for (std::size_t i = 0; i < 2; ++i) {
      //      std::cout << upper_right.at(i) << ", " << lower_left.at(i) <<
      //      "\n";
      if (upper_right.at(i) <= lower_left.at(i)) {
        throw std::invalid_argument("invalid region");
      }
    }
  }

  float operator()(const mgard::TensorMeshHierarchy<2, float> &hierarchy,
                   float const *const u) const {
    const std::array<std::size_t, 2> shape = hierarchy.shapes.back();
    const std::size_t n = shape.at(0);
    const std::size_t m = shape.at(1);
    //    std::cout << p << ", " << m << ", " << n << "\n";
    if (upper_right.at(0) > n || upper_right.at(1) > m) {
      throw std::invalid_argument("region isn't contained in domain");
    }
    float total = 0;
    std::size_t count = 0;
    for (std::size_t i = lower_left.at(0); i < upper_right.at(0); ++i) {
      for (std::size_t j = lower_left.at(1); j < upper_right.at(1); ++j) {
        total += u[m * i + j];
        ++count;
      }
    }
    return total / count;
  }

private:
  std::array<std::size_t, 2> lower_left;
  std::array<std::size_t, 2> upper_right;
};

int main(int argc, char **argv) {
  size_t vx = 128, vy = 128;
  std::vector<size_t> w2d = {0, 0, vx, vy};
  float tol = 1.0;
  if (argc > 1) {
    vx = (size_t)std::stoi(argv[1]);
    vy = (size_t)std::stoi(argv[2]);
    w2d[0] = (size_t)std::stoi(argv[3]);
    w2d[1] = (size_t)std::stoi(argv[4]);
    w2d[2] = (size_t)std::stoi(argv[5]);
    w2d[3] = (size_t)std::stoi(argv[6]);
    tol = std::stof(argv[7]);
  }
  const mgard::TensorMeshHierarchy<2, float> hierarchy({vx, vy});

  const AverageFunctional average({w2d[0], w2d[1]}, {w2d[2], w2d[3]});
  const mgard::TensorQuantityOfInterest<2, float> Q(hierarchy, average);
  const float s = 0;
  float Q_norm = Q.norm(s);
  std::cout << "request error bound on QoI (average) = " << tol << "\n";
  std::cout << "Q_norm = " << Q_norm << "\n";

  float *const u =
      static_cast<float *>(std::malloc(hierarchy.ndof() * sizeof(*u)));
  {
    float *p = u;
    for (std::size_t i = 0; i < vx; ++i) {
      const float x = 2.5 + static_cast<float>(i) / 60;
      for (std::size_t m = 0; m < vy; ++m) {
        const float y = 0.75 + static_cast<float>(m) / 15;
        *p++ = 12 + std::sin(2.1 * x - 1.3 * y);
      }
    }
  }
  std::cout << "average using original data: " << average(hierarchy, u)
            << std::endl;
  const float tolerance = tol / Q_norm;
  const mgard::CompressedDataset<2, float> compressed =
      mgard::compress(hierarchy, u, s, tolerance);
  std::cout << "after compression\n";
  const mgard::DecompressedDataset<2, float> decompressed =
      mgard::decompress(compressed);

  std::cout << "average using decompressed data: "
            << average(hierarchy, decompressed.data())
            << ", CR = " << vx * vy * 4 / (compressed.size()) << std::endl;
  float err =
      std::abs(average(hierarchy, decompressed.data()) - average(hierarchy, u));
  std::cout << "real error of QoI (average) = " << err << "\n";
  if (err < tol)
    std::cout << "********** Successful **********\n";
  else
    std::cout << "********** Fail with error preservation **********\n";
  std::free(u);

  return 0;
}
