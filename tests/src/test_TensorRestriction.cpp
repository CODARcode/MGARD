#include "catch2/catch.hpp"

#include <array>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMassMatrix.hpp"
#include "TensorMeshHierarchy.hpp"
#include "TensorProlongation.hpp"
#include "TensorRestriction.hpp"
#include "utilities.hpp"

TEST_CASE("constituent restrictions", "[TensorRestriction]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::array<float, 9> u_ = {9, 2, 4, -4, 7, 5, -2, 5, 6};
    const std::size_t dimension = 0;
    const std::array<std::array<float, 9>, 3> expecteds = {
        {{10, 2, 3, -4, 7.5, 5, 3, 5, 8.5},
         {11, 2, 4, -4, 8, 5, -2, 5, 5},
         {12.5, 2, 4, -4, 7, 5, -2, 5, 9.5}}};
    for (std::size_t l = 3; l > 0; --l) {
      const std::size_t i = 3 - l;
      const std::array<float, 9> &expected = expecteds.at(i);
      const mgard::ConstituentRestriction<1, float> R(hierarchy, l, dimension);
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      R({0}, v);
      TrialTracker tracker;
      for (std::size_t i = 0; i < 9; ++i) {
        tracker += v_.at(i) == expected.at(i);
      }
      REQUIRE(tracker);
    }
  }

  SECTION("1D and custom spacing and nondyadic") {
    const std::vector<double> xs = {0.0, 0.1, 0.9, 1.0};
    const mgard::TensorMeshHierarchy<1, double> hierarchy({4}, {xs});
    const std::array<double, 4> u_ = {5, 2, 2, 4};
    const std::size_t dimension = 0;
    const std::array<std::array<double, 4>, 2> expecteds = {
        {{5, 20. / 9, 2, 52. / 9}, {6.8, 2, 2, 4.2}}};
    for (std::size_t l = 2; l > 0; --l) {
      const mgard::ConstituentRestriction<1, double> R(hierarchy, l, dimension);
      std::array<double, 4> v_ = u_;
      double *const v = v_.data();
      R({0}, v);
      TrialTracker tracker;
      const std::array<double, 4> &expected = expecteds.at(2 - l);
      for (std::size_t i = 0; i < 4; ++i) {
        tracker += v_.at(i) == Approx(expected.at(i));
      }
      REQUIRE(tracker);
    }
  }

  SECTION("2D and custom spacing") {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {3, 3}, {{{0, 0.75, 1}, {0, 0.25, 1}}});
    const std::array<double, 9> u_ = {-9, -5, 1, 9, 3, 2, 9, 5, 6};
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentRestriction<2, double> R(hierarchy, l, dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{0, 0}, {0, 1}}};
      const std::array<std::array<double, 9>, 2> expecteds = {{
          {-6.75, -5, 1, 9, 3, 2, 15.75, 5, 6},
          {-9, -4.25, 1, 9, 3, 2, 9, 7.25, 6},
      }};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        std::array<double, 9> v_ = u_;
        double *const v = v_.data();
        R(multiindex, v);
        TrialTracker tracker;
        for (std::size_t j = 0; j < 9; ++j) {
          tracker += v_.at(j) == Approx(expected.at(j));
        }
        REQUIRE(tracker);
      }
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 1;
      const mgard::ConstituentRestriction<2, double> R(hierarchy, l, dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{1, 0}, {2, 0}}};
      const std::array<std::array<double, 9>, 2> expecteds = {{
          {-9, -5, 1, 11.25, 3, 2.75, 9, 5, 6},
          {-9, -5, 1, 9, 3, 2, 12.75, 5, 7.25},
      }};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        std::array<double, 9> v_ = u_;
        double *const v = v_.data();
        R(multiindex, v);
        TrialTracker tracker;
        for (std::size_t j = 0; j < 9; ++j) {
          tracker += v_.at(j) == Approx(expected.at(j));
        }
        REQUIRE(tracker);
      }
    }
    {
      bool thrown = false;
      try {
        const mgard::ConstituentRestriction<2, double> R(hierarchy, 0, 0);
      } catch (...) {
        // The constructor will throw a `std::invalid_argument` exception, but
        // (as of this writing) the attempt to initialize `coarse_indices`
        // throws first.
        thrown = true;
      }
      REQUIRE(thrown);
    }
  }
}

namespace {

// Prolongate a function on the coarse mesh to the fine mesh, project back down
// to the coarse mesh, and compare.
template <std::size_t N, typename Real>
void test_tensor_projection_identity(std::default_random_engine &generator,
                                     const std::array<std::size_t, N> shape) {
  std::uniform_real_distribution<Real> node_spacing_distribution(1, 1.5);
  std::uniform_real_distribution<Real> polynomial_coefficient_distribution(
      -0.5, 1.25);
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, node_spacing_distribution,
                                    shape);

  std::vector<Real> u_(hierarchy.ndof());
  Real *const u = u_.data();

  for (std::size_t l = hierarchy.L; l > 0; --l) {
    const MultilinearPolynomial<Real, N> p(generator,
                                           polynomial_coefficient_distribution);
    for (const mgard::TensorNode<N, Real> node : hierarchy.nodes(l)) {
      hierarchy.at(u, node.multiindex) = node.l == l ? 0 : p(node.coordinates);
    }

    const mgard::TensorProlongationAddition<N, Real> PA(hierarchy, l);
    const mgard::TensorRestriction<N, Real> R(hierarchy, l);
    const mgard::TensorMassMatrixInverse<N, Real> m_inv(hierarchy, l - 1);
    const mgard::TensorMassMatrix<N, Real> M(hierarchy, l);

    PA(u);
    M(u);
    R(u);
    m_inv(u);

    TrialTracker tracker;
    for (const mgard::TensorNode<N, Real> node : hierarchy.nodes(l - 1)) {
      // Encountered a handful of small errors.
      tracker += hierarchy.at(u, node.multiindex) ==
                 Approx(p(node.coordinates)).epsilon(0.001);
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("tensor product restrictions", "[TensorRestriction]") {
  {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {3, 3}, {{{0, 0.5, 1}, {-1, -0.5, 1}}});
    const std::array<double, 9> u_ = {6, 0, 7, -6, -10, 8, -6, 3, 9};
    const std::size_t l = 1;
    const mgard::TensorRestriction<2, double> R(hierarchy, l);
    const std::array<double, 9> expected = {-0.75, -5,    9.75, -13.5, -10,
                                            5.5,   -10.5, -2,   12.5};
    std::array<double, 9> v_ = u_;
    double *const v = v_.data();
    R(v);
    REQUIRE(v_ == expected);
  }

  std::default_random_engine generator(445624);

  SECTION("dyadic") {
    test_tensor_projection_identity<1, float>(generator, {129});
    test_tensor_projection_identity<2, double>(generator, {65, 65});
    test_tensor_projection_identity<3, float>(generator, {33, 9, 33});
    test_tensor_projection_identity<4, double>(generator, {9, 9, 17, 17});
  }

  SECTION("nondyadic") {
    test_tensor_projection_identity<1, float>(generator, {105});
    test_tensor_projection_identity<2, double>(generator, {45, 65});
    test_tensor_projection_identity<3, float>(generator, {36, 10, 27});
    test_tensor_projection_identity<4, double>(generator, {9, 19, 6, 8});
  }
}
