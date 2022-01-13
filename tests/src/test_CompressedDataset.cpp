#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <random>
#include <sstream>

#include "CompressedDataset.hpp"
#include "compress.hpp"

#include "testing_random.hpp"

TEST_CASE("data buffers and sizes", "[CompressedDataset]") {
  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({6, 3});
    const float s = 0.25;
    const float tolerance = 1.2;
    const std::size_t size = 11;
    void const *const data = new unsigned char[size];
    const mgard::pb::Header header;
    const mgard::CompressedDataset<2, float> compressed(hierarchy, header, s,
                                                        tolerance, data, size);
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
    const mgard::pb::Header header;
    const mgard::CompressedDataset<3, double> compressed(hierarchy, header, s,
                                                         tolerance, data, size);
    const mgard::DecompressedDataset<3, double> decompressed(compressed, u);
    REQUIRE(decompressed.data() == u);
  }
}

namespace {

template <std::size_t N, typename Real>
void test_serialization(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                        const Real s, const Real tolerance,
                        std::default_random_engine &gen, const Real s_gen) {
  const std::size_t ndof = hierarchy.ndof();
  // We wouldn't need to make two datasets if `decompress` didn't modify the
  // compressed buffer.
  Real *const u = new Real[ndof];
  Real *const v = new Real[ndof];

  generate_reasonable_function(hierarchy, s_gen, gen, u);
  std::copy(u, u + ndof, v);

  const mgard::CompressedDataset<N, Real> u_compressed =
      mgard::compress(hierarchy, u, s, tolerance);
  const mgard::CompressedDataset<N, Real> v_compressed =
      mgard::compress(hierarchy, u, s, tolerance);

  delete[] u;
  delete[] v;

  const mgard::DecompressedDataset<N, Real> u_decompressed =
      mgard::decompress(u_compressed);
  std::ostringstream v_ostream(std::ios_base::binary);
  v_compressed.write(v_ostream);
  const std::string v_serialization = v_ostream.str();
  mgard::MemoryBuffer<const unsigned char> v_decompressed =
      mgard::decompress(v_serialization.c_str(), v_serialization.size());

  Real const *const p = reinterpret_cast<Real const *>(u_decompressed.data());
  Real const *const q =
      reinterpret_cast<Real const *>(v_decompressed.data.get());
  REQUIRE(std::equal(p, p + ndof, q));
}

} // namespace

TEST_CASE("compressed dataset (de)serialization", "[CompressedDataset]") {
  std::default_random_engine gen(534393);
  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({381});
    const float s = -1;
    const float tolerance = 0.01;
    const float s_gen = 0;
    test_serialization(hierarchy, s, tolerance, gen, s_gen);
  }
  {
    const mgard::TensorMeshHierarchy<2, double> hierarchy({72, 72});
    const double s = 0.25;
    const double tolerance = 0.5;
    const double s_gen = 1;
    test_serialization(hierarchy, s, tolerance, gen, s_gen);
  }
  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({19, 31, 5});
    const float s = std::numeric_limits<float>::infinity();
    const float tolerance = 0.05;
    const float s_gen = 0.25;
    test_serialization(hierarchy, s, tolerance, gen, s_gen);
  }
  {
    const mgard::TensorMeshHierarchy<4, double> hierarchy({5, 5, 5, 7});
    const double s = 0;
    const double tolerance = 0.0001;
    const double s_gen = 1.25;
    test_serialization(hierarchy, s, tolerance, gen, s_gen);
  }
}
