#include "catch2/catch.hpp"

#include <array>
#include <numeric>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorMeshHierarchyIteration.hpp"
#include "TensorProlongation.hpp"
#include "shuffle.hpp"
#include "utilities.hpp"

TEST_CASE("constituent prolongations", "[TensorProlongation]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::size_t ndof = 9;
    const std::array<float, ndof> u_ = {-10, 10, -2, 3, 9, -8, 9, -8, 4};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();

    const std::size_t dimension = 0;
    const std::array<std::array<float, ndof>, 3> expecteds = {
        {{-10, 4, -2, 6.5, 9, 1, 9, -1.5, 4},
         {-10, 10, -2.5, 3, 9, -8, 15.5, -8, 4},
         {-10, 10, -2, 3, 6, -8, 9, -8, 4}}};
    for (std::size_t l = 3; l > 0; --l) {
      const std::size_t i = 3 - l;
      const std::array<float, ndof> &expected = expecteds.at(i);
      const mgard::ConstituentProlongationAddition<1, float> PA(hierarchy, l,
                                                                dimension);
      mgard::shuffle(hierarchy, u, v);
      PA({0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += buffer_.at(i) == expected.at(i);
      }
      REQUIRE(tracker);
    }
  }

  SECTION("2D and custom spacing") {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {3, 3}, {{{0, 0.9, 1}, {1, 1.4, 2}}});
    const std::size_t ndof = 3 * 3;
    const std::array<double, ndof> u_ = {4, -1, 4, 7, 3, -3, -4, 1, -7};
    std::array<double, ndof> v_;
    std::array<double, ndof> buffer_;
    double const *const u = u_.data();
    double *const v = v_.data();
    double *const buffer = buffer_.data();
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentProlongationAddition<2, double> PA(hierarchy, l,
                                                                 dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{0, 0}, {0, 1}}};
      const std::array<std::array<double, ndof>, 2> expecteds = {
          {{4, -1, 4, 3.8, 3, -3, -4, 1, -7},
           {4, -1, 4, 7, 3.8, -3, -4, 1, -7}}};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, ndof> &expected = expecteds.at(i);
        mgard::shuffle(hierarchy, u, v);
        PA(multiindex, v);
        mgard::unshuffle(hierarchy, v, buffer);
        TrialTracker tracker;
        for (std::size_t j = 0; j < ndof; ++j) {
          tracker += buffer_.at(j) == Catch::Approx(expected.at(j));
        }
        REQUIRE(tracker);
      }
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 1;
      const mgard::ConstituentProlongationAddition<2, double> PA(hierarchy, l,
                                                                 dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{1, 0}, {2, 0}}};
      const std::array<std::array<double, 9>, 2> expecteds = {
          {{4, -1, 4, 7, 6.0, -3, -4, 1, -7},
           {4, -1, 4, 7, 3, -3, -4, -4.2, -7}}};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        mgard::shuffle(hierarchy, u, v);
        PA(multiindex, v);
        mgard::unshuffle(hierarchy, v, buffer);
        TrialTracker tracker;
        for (std::size_t j = 0; j < 9; ++j) {
          tracker += buffer_.at(j) == Catch::Approx(expected.at(j));
        }
        REQUIRE(tracker);
      }
    }
    {
      bool thrown = false;
      try {
        const mgard::ConstituentProlongationAddition<2, double> PA(hierarchy, 0,
                                                                   0);
      } catch (...) {
        // The constructor will throw a `std::invalid_argument` exception, but
        // (as of this writing) the attempt to initialize `coarse_indices`
        // throws first.
        thrown = true;
      }
      REQUIRE(thrown);
    }
  }

  SECTION("2D and nondyadic") {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({4, 4});
    const std::size_t ndof = 4 * 4;
    const std::array<float, ndof> u_ = {-2, -4, -9, -4, -5,  8,  4, -9,
                                        -4, 2,  -5, 0,  -10, -5, 9, -1};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();
    {
      const std::array<std::size_t, 2> ls = {2, 1};
      const std::array<std::size_t, 2> dimensions = {0, 1};

      const std::array<std::array<std::size_t, 2>, 4> multiindices = {
          {{0, 0}, {0, 1}, {2, 0}, {3, 0}}};
      const std::array<std::array<float, ndof>, 4> expecteds = {
          {{-2, -4, -9, -4, -5, 8, 4, -9, -11.5, 2, -5, 0, -10, -5, 9, -1},
           {-2, -4, -9, -4, -5, 8, 4, -9, -4, 3.5, -5, 0, -10, -5, 9, -1},
           {-2, -4, -9, -4, -5, 8, 4, -9, -4, -2. / 3, -5, 0, -10, -5, 9, -1},
           {-2, -4, -9, -4, -5, 8, 4, -9, -4, 2, -5, 0, -10, -12, 9, -1}}};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::size_t l = ls.at(i);
        const std::size_t dimension = dimensions.at(i);
        const mgard::ConstituentProlongationAddition<2, float> PA(hierarchy, l,
                                                                  dimension);
        for (std::size_t j = 2 * i; j < 2 * (i + 1); ++j) {
          const std::array<std::size_t, 2> &multiindex = multiindices.at(j);
          const std::array<float, ndof> &expected = expecteds.at(j);
          mgard::shuffle(hierarchy, u, v);
          PA(multiindex, v);
          mgard::unshuffle(hierarchy, v, buffer);
          TrialTracker tracker;
          for (std::size_t k = 0; k < ndof; ++k) {
            tracker += buffer_.at(k) == Catch::Approx(expected.at(k));
          }
          REQUIRE(tracker);
        }
      }
    }
  }
}

namespace {

template <std::size_t N, typename Real>
void test_tensor_product_prolongations(std::default_random_engine &generator,
                                       const std::array<std::size_t, N> shape) {
  std::uniform_real_distribution<Real> node_spacing_distribution(0.1, 1);
  std::uniform_real_distribution<Real> polynomial_coefficient_distribution(-2,
                                                                           2);
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, node_spacing_distribution,
                                    shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof);
  Real *const u = u_.data();
  for (std::size_t l = hierarchy.L; l > 0; --l) {
    const MultilinearPolynomial<Real, N> p(generator,
                                           polynomial_coefficient_distribution);
    const mgard::ShuffledTensorNodeRange<N, Real> nodes(hierarchy, l);
    for (const mgard::TensorNode<N> node : nodes) {
      hierarchy.at(u, node.multiindex) =
          hierarchy.date_of_birth(node.multiindex) == l
              ? 0
              : p(coordinates(hierarchy, node));
    }
    const mgard::TensorProlongationAddition<N, Real> PA(hierarchy, l);
    PA(u);
    TrialTracker tracker;
    for (const mgard::TensorNode<N> node : nodes) {
      tracker += hierarchy.at(u, node.multiindex) ==
                 Catch::Approx(p(coordinates(hierarchy, node))).epsilon(0.001);
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("tensor product prolongations", "[TensorProlongation]") {
  std::default_random_engine generator(176067);

  SECTION("dyadic") {
    test_tensor_product_prolongations<1, float>(generator, {129});
    test_tensor_product_prolongations<2, double>(generator, {17, 17});
    // Before increasing the `Catch::Approx` tolerance we got a handful of
    // errors (all quite small) with this one.
    test_tensor_product_prolongations<3, float>(generator, {9, 9, 17});
    test_tensor_product_prolongations<4, double>(generator, {33, 17, 33, 17});
  }

  SECTION("nondyadic") {
    test_tensor_product_prolongations<1, double>(generator, {82});
    test_tensor_product_prolongations<2, float>(generator, {15, 17});
    test_tensor_product_prolongations<3, double>(generator, {6, 10, 8});
    test_tensor_product_prolongations<4, float>(generator, {5, 12, 8, 9});
  }
}

TEST_CASE("prolongations on 'flat' meshes", "[TensorProlongation]") {
  const std::size_t ndof = 12;
  const std::size_t l = 3;
  std::vector<float> u_(ndof);
  std::vector<float> expected_(ndof);
  std::vector<float> obtained_(ndof);
  float *const u = u_.data();
  float *const expected = expected_.data();
  float *const obtained = obtained_.data();
  std::iota(u, u + ndof, 0);
  // We'll finish initializing `u` in the next block.
  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({12});
    for (float &value : hierarchy.on_new_nodes(u, l)) {
      value = 0;
    }
    const mgard::TensorProlongationAddition<1, float> PA(hierarchy, l);
    std::copy(u, u + ndof, expected);
    PA(expected);
  }

  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({1, 12, 1});
    const mgard::TensorProlongationAddition<3, float> PA(hierarchy, l);
    std::copy(u, u + ndof, obtained);
    PA(obtained);

    REQUIRE(obtained_ == expected_);
  }

  {
    const mgard::TensorMeshHierarchy<4, float> hierarchy({12, 1, 1, 1});
    const mgard::TensorProlongationAddition<4, float> PA(hierarchy, l);
    std::copy(u, u + ndof, obtained);
    PA(obtained);

    REQUIRE(obtained_ == expected_);
  }
}
