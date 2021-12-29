#include "catch2/catch_test_macros.hpp"

#include "CompressedDataset.hpp"

TEST_CASE("data buffers and sizes", "[CompressedDataset]") {
  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({6, 3});
    const float s = 0.25;
    const float tolerance = 1.2;
    const std::size_t size = 11;
    void const *const data = new unsigned char[size];
    const mgard::CompressedDataset<2, float> compressed(hierarchy, s, tolerance,
                                                        data, size);
    REQUIRE(compressed.data() == data);
    REQUIRE(compressed.size() == size);
  }
  {
    const mgard::TensorMeshHierarchy<3, double> hierarchy({5, 4, 5});
    const double s = -0.1;
    const double tolerance = 0.025;
    const std::size_t size = 78;
    double const *const u = new double[hierarchy.ndof()];
    void const *const data = new unsigned char[size];
    const mgard::CompressedDataset<3, double> compressed(hierarchy, s,
                                                         tolerance, data, size);
    const mgard::DecompressedDataset<3, double> decompressed(compressed, u);
    REQUIRE(decompressed.data() == u);
  }
}
