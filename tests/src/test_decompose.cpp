#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

#include <cstddef>

#include <array>
#include <numeric>
#include <random>
#include <vector>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorProlongation.hpp"
#include "blas.hpp"
#include "decompose.hpp"
#include "shuffle.hpp"

namespace {

template <std::size_t N>
std::array<std::size_t, N> get_dyadic_shape(const std::size_t L) {
  std::array<std::size_t, N> shape;
  shape.fill((1 << L) + 1);
  return shape;
}

} // namespace

namespace {

template <std::size_t N, typename Real>
void test_dyadic_uniform_decomposition(
    const std::vector<Real> &u_,
    const std::vector<std::vector<Real>> &expecteds) {
  Real const *const u = u_.data();
  for (std::size_t L = 0; L < expecteds.size(); ++L) {
    const mgard::TensorMeshHierarchy<N, Real> hierarchy(get_dyadic_shape<N>(L));
    const std::size_t ndof = hierarchy.ndof();
    std::vector<Real> v_(ndof);
    std::vector<Real> buffer_(ndof);
    Real *const v = v_.data();
    Real *const buffer = buffer_.data();

    mgard::shuffle(hierarchy, u, v);
    mgard::decompose(hierarchy, v);
    mgard::unshuffle(hierarchy, v, buffer);

    const std::vector<Real> &expected = expecteds.at(L);
    TrialTracker tracker;
    tracker += expected.size() == ndof;
    for (std::size_t i = 0; i < ndof; ++i) {
      // Got some small errors here when run on Travis.
      tracker += buffer_.at(i) == Catch::Approx(expected.at(i)).epsilon(1e-4);
    }
    REQUIRE(tracker);
  }
}

template <std::size_t N, typename Real>
void test_dyadic_uniform_recomposition(
    const std::vector<Real> &u_,
    const std::vector<std::vector<Real>> &expecteds) {
  Real const *const u = u_.data();
  for (std::size_t L = 0; L < expecteds.size(); ++L) {
    const mgard::TensorMeshHierarchy<N, Real> hierarchy(get_dyadic_shape<N>(L));
    const std::size_t ndof = hierarchy.ndof();
    std::vector<Real> v_(ndof);
    std::vector<Real> buffer_(ndof);
    Real *const v = v_.data();
    Real *const buffer = buffer_.data();

    mgard::shuffle(hierarchy, u, v);
    mgard::recompose(hierarchy, v);
    mgard::unshuffle(hierarchy, v, buffer);

    const std::vector<Real> &expected = expecteds.at(L);
    TrialTracker tracker;
    tracker += expected.size() == ndof;
    for (std::size_t i = 0; i < ndof; ++i) {
      // Got a single very small error in the 3D example.
      tracker += buffer_.at(i) == Catch::Approx(expected.at(i)).epsilon(0.0001);
    }
    REQUIRE(tracker);
  }
}

template <std::size_t N, typename Real>
void test_decomposition_linearity(
    std::default_random_engine &generator,
    std::uniform_real_distribution<Real> &node_spacing_distribution,
    std::uniform_real_distribution<Real> &nodal_coefficient_distribution,
    const std::array<std::size_t, N> shape) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing<N, Real>(generator,
                                             node_spacing_distribution, shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof);
  std::vector<Real> v_(ndof);
  std::vector<Real> w_(ndof);
  Real *const u = u_.data();
  Real *const v = v_.data();
  Real *const w = w_.data();

  // Completely random nodal coefficients was causing some issues with
  // cancellation errors. Generating a slightly nicer function here.
  {
    const Real s = 0.5;
    generate_reasonable_function(hierarchy, s, generator, u);
    generate_reasonable_function(hierarchy, s, generator, v);
  }
  const Real alpha = nodal_coefficient_distribution(generator);
  for (std::size_t i = 0; i < ndof; ++i) {
    w_.at(i) = alpha * u_.at(i) + v_.at(i);
  }

  mgard::decompose(hierarchy, u);
  mgard::decompose(hierarchy, v);
  mgard::decompose(hierarchy, w);

  TrialTracker tracker;
  for (std::size_t i = 0; i < ndof; ++i) {
    // Encountering a few small errors (and more with optimizations turned on).
    tracker += w_.at(i) == Catch::Approx(alpha * u_.at(i) + v_.at(i))
                               .epsilon(0.001)
                               .margin(0.000001);
  }
  REQUIRE(tracker);
}

template <std::size_t N, typename Real>
void test_recomposition_linearity(
    std::default_random_engine &generator,
    std::uniform_real_distribution<Real> &node_spacing_distribution,
    std::uniform_real_distribution<Real> &multilevel_coefficient_distribution,
    const std::array<std::size_t, N> shape) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing<N, Real>(generator,
                                             node_spacing_distribution, shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof);
  std::vector<Real> v_(ndof);
  std::vector<Real> w_(ndof);
  Real *const u = u_.data();
  Real *const v = v_.data();
  Real *const w = w_.data();

  const Real alpha = multilevel_coefficient_distribution(generator);
  for (std::size_t i = 0; i < ndof; ++i) {
    u_.at(i) = multilevel_coefficient_distribution(generator);
    v_.at(i) = multilevel_coefficient_distribution(generator);
    w_.at(i) = alpha * u_.at(i) + v_.at(i);
  }

  mgard::recompose(hierarchy, u);
  mgard::recompose(hierarchy, v);
  mgard::recompose(hierarchy, w);

  TrialTracker tracker;
  for (std::size_t i = 0; i < ndof; ++i) {
    // Encountering a few small errors (and more with optimizations turned on).
    tracker += w_.at(i) == Catch::Approx(alpha * u_.at(i) + v_.at(i))
                               .epsilon(0.001)
                               .margin(0.000001);
  }
  REQUIRE(tracker);
}

template <std::size_t N, typename Real>
void test_decomposition_of_linear_functions(
    std::default_random_engine &generator,
    std::uniform_real_distribution<Real> &node_spacing_distribution,
    std::uniform_real_distribution<Real> &nodal_coefficient_distribution,
    const std::array<std::size_t, N> shape) {
  // If a function is piecewise linear on level `l - 1`, its multilevel
  // coefficients on level `l` should be zero.
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, node_spacing_distribution,
                                    shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof);
  double *const u = u_.data();

  // Not going to bother checking that `hierarchy.L` is nonzero.
  {
    const mgard::PseudoArray<Real> u_on_old =
        hierarchy.on_nodes(u, hierarchy.L - 1);
    std::generate(u_on_old.begin(), u_on_old.end(), [&]() -> Real {
      return nodal_coefficient_distribution(generator);
    });
  }

  const mgard::PseudoArray<Real> u_on_newest =
      hierarchy.on_new_nodes(u, hierarchy.L);
  std::fill(u_on_newest.begin(), u_on_newest.end(), 0);

  {
    const mgard::TensorProlongationAddition PA(hierarchy, hierarchy.L);
    PA(u);
  }
  mgard::decompose(hierarchy, u);

  TrialTracker tracker;
  for (const Real &value : u_on_newest) {
    tracker += std::abs(value) < 1e-6;
  }
  REQUIRE(tracker);
}

template <std::size_t N, typename Real>
void test_recomposition_with_zero_coefficients(
    std::default_random_engine &generator,
    std::uniform_real_distribution<Real> &node_spacing_distribution,
    std::uniform_real_distribution<Real> &multilevel_coefficient_distribution,
    const std::array<std::size_t, N> shape) {
  // If the multilevel coefficients on level `l` are zero, the function
  // should be piecewise linear on level `l - 1`.
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing<N, Real>(generator,
                                             node_spacing_distribution, shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof, 0);
  Real *const u = u_.data();

  // Not going to bother checking that `hierarchy.L` is nonzero.
  const mgard::PseudoArray<Real> u_on_old =
      hierarchy.on_nodes(u, hierarchy.L - 1);
  std::generate(u_on_old.begin(), u_on_old.end(), [&]() -> Real {
    return multilevel_coefficient_distribution(generator);
  });

  mgard::recompose(hierarchy, u);

  std::vector<Real> v_(ndof, 0);
  Real *const v = v_.data();
  // We could just use `v` here.
  const mgard::PseudoArray<Real> v_on_old =
      hierarchy.on_nodes(v, hierarchy.L - 1);
  blas::copy(u_on_old.size, u_on_old.data, v_on_old.data);
  {
    const mgard::TensorProlongationAddition PA(hierarchy, hierarchy.L);
    PA(v);
  }

  TrialTracker tracker;
  for (std::size_t i = 0; i < ndof; ++i) {
    tracker += u_.at(i) == Catch::Approx(v_.at(i));
  }
  REQUIRE(tracker);
}

} // namespace

TEST_CASE("decomposition", "[mgard]") {
  SECTION("1D, dyadic, uniform") {
    const std::vector<float> u_ = {10, 3,   -8,  -6, 3,  0, -5, 0, 0,  -2, -8,
                                   -5, -10, -7,  8,  -2, 3, -1, 0, 9,  -4, -6,
                                   -8, -5,  -10, 1,  3,  7, -8, 1, 10, -2, 8};
    const std::vector<std::vector<float>> expecteds = {
        {10.0, 3.0},
        {11.0, 2.0, -7.0},
        {4.4375, 2.0, -14.5, -3.5, -6.687500000000001},
        {0.4374999999999991, 2.0, -15.678571428571429, -3.5, -4.625000000000002,
         1.0, -5.321428571428571, 2.5, -2.4375000000000004},
        {-0.95703125, 2.0, -15.652199926362297, -3.5, -4.122767857142856, 1.0,
         -4.978599042709867, 2.5, -4.765625000000001, 2.0, -1.173186671575852,
         4.0, -10.689732142857139, -6.0, 9.303985640648008, -7.5,
         -3.1054687499999987},
        {-2.640624999999999,  2.0,  -15.652212055333662, -3.5,
         -4.04917617820324,   1.0,  -4.978756719337627,  2.5,
         -6.024553571428571,  2.0,  -1.1753820153931231, 4.0,
         -9.73304031664212,   -6.0, 9.273408503833881,   -7.5,
         -3.0234374999999996, -2.5, 3.878101069067445,   11.0,
         -1.7446382547864507, 0.0,  -3.6049935368896406, 4.0,
         -8.00669642857143,   4.5,  13.02698941447754,   9.5,
         2.7768547496318097,  0.0,  8.232845339575196,   -11.0,
         0.14062500000000222}};
    test_dyadic_uniform_decomposition<1, float>(u_, expecteds);
  }

  SECTION("2D, dyadic, uniform") {
    const std::vector<double> u_ = {7,  4, 5,  -10, -6, 6,   -8, -5, 6,
                                    -2, 2, -2, 9,   2,  -10, 3,  8,  -8,
                                    -3, 7, -8, -9,  -6, -1,  -4};
    const std::vector<std::vector<double>> expecteds = {
        {7.0, 4.0, 5.0, -10.0},
        {0.9999999999999973, -2.0, 3.9999999999999982, -9.5, -8.5, 0.5,
         -15.000000000000004, -4.0, 3.9999999999999973},
        {3.8007812499999973,
         -2.0,
         -2.9062499999998854,
         -9.5,
         -2.910156250000001,
         1.5,
         -13.75,
         -12.0,
         6.5,
         6.0,
         -1.593749999999881,
         -7.5,
         2.8750000000004396,
         2.5,
         -1.0312499999998854,
         6.0,
         8.75,
         -9.5,
         -0.25,
         14.0,
         -2.5039062500000013,
         -2.0,
         -10.218749999999885,
         4.0,
         -0.6992187500000024}};
    test_dyadic_uniform_decomposition<2, double>(u_, expecteds);
  }

  SECTION("4D, dyadic, uniform") {
    const std::vector<float> u_ = {
        -5, -3, -3, -2, 3,  -9, -2,  1,  1,  2,   4,  5,  -7, -7, 2,  2,  -3,
        -8, -3, -1, -9, -1, -1, 4,   7,  4,  -4,  -8, 1,  7,  -6, -6, 10, 0,
        2,  10, -1, -4, 1,  -4, 5,   3,  2,  1,   3,  -5, 4,  9,  -8, -8, -1,
        -1, 2,  -6, -7, -8, 8,  -10, -4, 5,  -4,  -1, 7,  3,  -4, -5, -3, -2,
        -6, 6,  3,  -9, -9, 10, 2,   2,  -2, -10, -2, -3, -1};
    const std::vector<std::vector<float>> expecteds = {
        {-5.0, -3.0, -3.0, -2.0, 3.0, -9.0, -2.0, 1.0, 1.0, 2.0, 4.0, 5.0, -7.0,
         -7.0, 2.0, 2.0},
        {-2.6406250000000058,
         1.0,
         2.2968749999999902,
         1.5,
         5.25,
         -8.0,
         -2.046875,
         1.5,
         0.7656249999999982,
         6.0,
         9.0,
         11.0,
         -6.25,
         -4.75,
         5.75,
         -0.5,
         -3.5,
         -6.5,
         -6.234375000000018,
         5.0,
         4.078124999999988,
         -3.0,
         1.25,
         10.5,
         0.10937499999999013,
         2.5,
         -1.7031250000000093,
         -2.0,
         2.75,
         4.5,
         -1.5,
         -5.375,
         6.75,
         3.0,
         1.5,
         6.0,
         5.0,
         -0.75,
         1.5,
         -0.875,
         6.5,
         2.875,
         2.25,
         0.75,
         2.25,
         1.0,
         8.75,
         12.5,
         -6.25,
         -5.625,
         2.0,
         -3.5,
         2.0,
         -3.5,
         -10.765625000000007,
         -8.5,
         1.4218749999999971,
         -4.5,
         -5.0,
         -2.5,
         -2.2968750000000018,
         -2.5,
         8.265625,
         11.0,
         -2.5,
         -10.0,
         2.5,
         -1.25,
         -10.0,
         9.0,
         3.0,
         -12.0,
         0.2656249999999858,
         13.5,
         0.3281249999999911,
         7.5,
         0.5,
         -10.5,
         2.4843749999999933,
         -1.5,
         -9.078125000000007}};
    test_dyadic_uniform_decomposition<4, float>(u_, expecteds);
  }

  SECTION("1D, dyadic, nonuniform") {
    const std::array<std::vector<float>, 1> coordinates = {{{10, 28, 34}}};
    const mgard::TensorMeshHierarchy<1, float> hierarchy({3}, coordinates);
    const std::size_t ndof = 3;
    const std::array<float, ndof> u_ = {{7, 2, 5}};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();

    mgard::shuffle(hierarchy, u, v);
    mgard::decompose(hierarchy, v);
    mgard::unshuffle(hierarchy, v, buffer);

    const std::array<float, ndof> expected = {{6.125, -3.5, 2.375}};
    TrialTracker tracker;
    for (std::size_t i = 0; i < ndof; ++i) {
      tracker += buffer_.at(i) == Catch::Approx(expected.at(i));
    }
    REQUIRE(tracker);
  }

  SECTION("linear dependence on input", "[mgard]") {
    std::default_random_engine generator(469957);
    std::uniform_real_distribution<float> node_spacing_distribution(0.2, 1.1);
    std::uniform_real_distribution<float> nodal_coefficient_distribution(0.5,
                                                                         2);

    test_decomposition_linearity<2, float>(generator, node_spacing_distribution,
                                           nodal_coefficient_distribution,
                                           {41, 22});
    test_decomposition_linearity<3, float>(generator, node_spacing_distribution,
                                           nodal_coefficient_distribution,
                                           {17, 17, 33});
    test_decomposition_linearity<4, float>(generator, node_spacing_distribution,
                                           nodal_coefficient_distribution,
                                           {11, 4, 4, 5});
  }

  // Piecewise linear on a coarser grid.
  SECTION("coefficients of linear functions", "[mgard]") {
    std::default_random_engine generator(841397);
    std::uniform_real_distribution<double> node_spacing_distribution(0.1, 0.3);
    std::uniform_real_distribution<double> nodal_coefficient_distribution(-2,
                                                                          2);

    test_decomposition_of_linear_functions<1, double>(
        generator, node_spacing_distribution, nodal_coefficient_distribution,
        {129});
    test_decomposition_of_linear_functions<2, double>(
        generator, node_spacing_distribution, nodal_coefficient_distribution,
        {21, 20});
    test_decomposition_of_linear_functions<3, double>(
        generator, node_spacing_distribution, nodal_coefficient_distribution,
        {14, 10, 17});
  }

  SECTION("on 'flat' meshes", "[mgard]") {
    std::default_random_engine gen(731641);
    // Node spacing distribution.
    std::uniform_real_distribution<double> dis(1, 1.1);
    const mgard::TensorMeshHierarchy<2, double> hierarchy =
        hierarchy_with_random_spacing<2, double>(gen, dis, {17, 21});
    const std::size_t ndof = hierarchy.ndof();

    std::vector<double> u_(ndof);
    std::vector<double> expected_(ndof);
    std::vector<double> obtained_(ndof);
    double *const u = u_.data();
    double *const expected = expected_.data();
    double *const obtained = obtained_.data();

    const double s = 1;
    generate_reasonable_function(hierarchy, s, gen, u);
    std::copy(u, u + ndof, expected);
    mgard::decompose(hierarchy, expected);

    {
      const mgard::TensorMeshHierarchy<3, double> flat_hierarchy =
          make_flat_hierarchy<2, 3, double>(hierarchy, {17, 1, 21});
      std::copy(u, u + ndof, obtained);
      mgard::decompose(flat_hierarchy, obtained);
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += obtained[i] == Catch::Approx(expected[i]);
      }
      REQUIRE(tracker);
    }

    {
      const mgard::TensorMeshHierarchy<5, double> flat_hierarchy =
          make_flat_hierarchy<2, 5, double>(hierarchy, {1, 1, 17, 21, 1});
      std::copy(u, u + ndof, obtained);
      mgard::decompose(flat_hierarchy, obtained);
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += obtained[i] == Catch::Approx(expected[i]);
      }
      REQUIRE(tracker);
    }
  }
}

TEST_CASE("recomposition", "[mgard]") {
  SECTION("1D, dyadic, uniform") {
    const std::vector<double> u_ = {-4, 2,  -4, 2, 7, -10, -4, -9, 9,
                                    6,  -1, 7,  9, 2, -9,  -8, -1};
    const std::vector<std::vector<double>> expecteds = {
        {-4.0, 2.0},
        {-5.0, -3.0, -5.0},
        {-3.0, -0.24999999999999956, -1.4999999999999991, 5.25, 8.0},
        {-6.0625, -3.34375, -4.625, 4.718749999999999, 10.062499999999998,
         -0.28125, 9.375, 1.40625, 11.4375},
        {-10.52568114874816, -8.027337997054492, -9.528994845360826,
         -0.6133100147275421, 4.302374815905742, -6.278350515463919,
         3.14092415316642, -7.764359351988221, -0.6696428571428602,
         3.934002209131073, -3.4623527245949943, 6.43013622974963,
         2.3226251840942544, -0.413475699558175, -7.149576583210604,
         -11.267304860088364, 0.6149668630338745}};
    test_dyadic_uniform_recomposition<1, double>(u_, expecteds);
  }

  SECTION("3D, dyadic, uniform") {
    const std::vector<float> u_ = {
        -6,  0,  -2,  -2, -3,  10, -2, -3, -1, 10,  5,  2,   9,   -10, -5,  5,
        0,   -5, 3,   -4, -1,  0,  1,  10, 8,  -7,  -3, -10, -9,  2,   7,   -10,
        5,   -9, -9,  2,  -7,  -5, 2,  -2, 5,  8,   3,  10,  -9,  -9,  10,  -4,
        9,   -3, 3,   6,  -1,  7,  -4, 6,  3,  -10, 9,  5,   -10, -6,  5,   6,
        -2,  9,  5,   -5, -4,  5,  -5, 1,  7,  1,   -7, -9,  -2,  -4,  6,   7,
        -10, -2, -10, -5, -10, 4,  -6, 4,  4,  -1,  -8, 4,   -3,  8,   6,   -6,
        -7,  -5, -2,  -8, 0,   -3, 3,  3,  8,  4,   2,  7,   -8,  -5,  -10, 3,
        0,   4,  1,   1,  7,   0,  -8, 4,  -5, -4,  -8, -1,  10};
    const std::vector<std::vector<float>> expecteds = {
        {-6.0, 0.0, -2.0, -2.0, -3.0, 10.0, -2.0, -3.0},
        {-8.3125,
         -5.3125,
         -2.3125,
         -6.8125,
         -5.3125,
         10.1875,
         -1.3124999999999996,
         -2.3125,
         2.6875,
         5.8125,
         2.0625,
         0.3125,
         8.5625,
         -10.1875,
         -4.9375,
         8.3125,
         2.5625,
         -3.1875,
         -0.0625,
         -4.5625,
         -1.0625,
         3.9375,
         2.9375,
         9.9375,
         7.9375,
         -2.5625,
         0.9375},
        {-8.442871093750002,  -5.143798828124998,  -1.8447265624999956,
         -1.4206542968749967, 3.003417968750002,   2.817138671875,
         -3.731811523437499,  0.7192382812500013,  2.349975585937501,
         12.980712890625,     -5.922851562499999,  3.6801757812499996,
         9.283203124999998,   -3.8793945312500018, 2.958007812499999,
         -1.7253417968750009, -2.259399414062502,  -2.7934570312500036,
         5.242797851562498,   -1.7209472656250004, -7.527832031250003,
         -6.1989746093750036, -4.870117187500005,  8.364990234374996,
         1.60009765625,       -12.239501953125002, -5.490600585937498,
         -9.741699218749995,  -9.469360351562498,  0.8029785156249989,
         0.03869628906249911, -11.16046142578125,  9.640380859375004,
         -6.847839355468749,  -9.336059570312502,  -6.68310546875,
         -6.830322265625,     4.0224609375,        6.773681640625,
         -1.4750976562500009, -4.1705322265625,    6.339660644531249,
         8.849853515624998,   13.36785888671875,   -8.1141357421875,
         -18.657958984375,    6.509643554687498,   -1.3227539062500027,
         10.962036132812498,  -1.7531738281249996, -2.036132812500001,
         6.162597656250002,   2.361328125000005,   5.48193359375,
         -5.397460937500004,  -0.73974609375,      2.4108886718750013,
         -4.438476562499997,  9.954345703125,      1.3471679687499973,
         -11.443359375,       -7.3408203125,       8.76171875,
         9.4267578125,        -1.9082031250000009, -2.61572265625,
         3.938720703125,      4.4931640625,        0.492919921875,
         4.49267578125,       -11.788085937500002, 0.2182617187499991,
         10.224609375,        6.559082031250001,   0.8935546875000011,
         -5.528076171875002,  2.5228271484375,     1.5737304687500018,
         9.9136962890625,     9.253662109374998,   -13.3111572265625,
         -0.8634643554687504, -4.415771484375,     -0.4915161132812509,
         -6.567260742187502,  -6.09423828125,      -8.249755859375,
         9.5947265625,        9.103271484374998,   3.6118164062499982,
         -14.0179443359375,   3.8834838867187496,  2.784912109375,
         13.24102783203125,   10.6971435546875,    -7.941650390625002,
         -4.983276367187501,  0.9750976562500009,  3.3787841796875,
         -3.217529296875,     8.979980468749998,   5.883056640624998,
         8.786132812499998,   12.345458984375,     9.90478515625,
         4.117431640624999,   4.862182617187498,   12.606933593749998,
         0.0626220703125,     5.518310546875,      -8.7451171875,
         -0.1586914062500009, 2.4277343749999987,  10.779785156249998,
         11.131835937499998,  0.5798339843749987,  7.8282470703125,
         2.0766601562500004,  -2.010864257812501,  13.901611328124998,
         7.904785156249997,   0.815185546875,      1.7255859375000022,
         4.198486328125,      8.671386718749998}};
    test_dyadic_uniform_recomposition<3, float>(u_, expecteds);
  }

  SECTION("4D, dyadic, uniform") {
    const std::vector<float> u_ = {
        10, 9,  3,  1,  -6,  3,  -10, -4,  4,  9,  -8, -8, -6, -2, 8,   -9,  -3,
        -2, -8, 1,  3,  -3,  -8, -7,  -9,  4,  9,  3,  -3, -3, 5,  -10, -10, -5,
        5,  9,  -3, -4, -10, 1,  -2,  -10, -3, -3, 7,  2,  7,  -5, 7,   9,   5,
        6,  9,  -5, -6, -10, 0,  7,   8,   -6, 3,  -2, -7, 10, -9, -5,  -7,  -2,
        0,  -9, 3,  -9, -7,  -6, 8,   8,   3,  2,  -4, 4,  -6};
    const std::vector<std::vector<float>> expecteds = {
        {10.0, 9.0, 3.0, 1.0, -6.0, 3.0, -10.0, -4.0, 4.0, 9.0, -8.0, -8.0,
         -6.0, -2.0, 8.0, -9.0},
        {7.484374999999998,
         17.265625,
         9.046875,
         4.218749999999999,
         -1.59375,
         8.59375,
         -1.046875,
         -3.4531249999999996,
         2.140625000000001,
         8.1875,
         -4.25,
         0.3125,
         -8.890625,
         -0.17187500000000022,
         14.546875,
         -13.96875,
         -3.0937499999999996,
         2.781250000000001,
         -9.109375,
         0.234375,
         7.578125,
         -12.0,
         -8.75,
         0.5,
         -8.890625,
         3.2656250000000004,
         7.421875000000001,
         3.2968749999999982,
         2.703124999999999,
         8.109375,
         5.874999999999999,
         -7.375,
         -5.624999999999999,
         -3.5468750000000004,
         4.546875,
         6.640625000000002,
         -7.656250000000001,
         -1.4062500000000004,
         -0.15625,
         -3.140625,
         -1.9843749999999996,
         -5.828124999999999,
         -6.625,
         -5.562499999999999,
         5.500000000000002,
         -7.609375,
         6.484375,
         3.578125,
         -2.15625,
         6.40625,
         8.96875,
         -2.703125,
         4.328125000000001,
         -5.640624999999998,
         -6.890625000000002,
         -6.859375000000001,
         13.171875,
         5.531249999999998,
         8.84375,
         -2.843749999999999,
         3.953124999999999,
         -3.4531249999999996,
         -6.859374999999998,
         1.4999999999999982,
         -7.562500000000001,
         6.375,
         -12.390625,
         -3.796875,
         1.7968750000000009,
         -11.28125,
         -2.031249999999999,
         -16.78125,
         -10.109375000000002,
         -6.265625000000001,
         9.578125,
         -1.3125,
         -1.4375,
         2.437500000000001,
         -8.515625,
         -4.609375,
         -8.703124999999998},
    };
    test_dyadic_uniform_recomposition<4, float>(u_, expecteds);
  }

  SECTION("linear dependence on input", "[mgard]") {
    std::default_random_engine generator(860343);
    std::uniform_real_distribution<float> node_spacing_distribution(0.1, 0.3);
    std::uniform_real_distribution<float> multilevel_coefficient_distribution(
        -1, 1);

    test_recomposition_linearity<1, float>(generator, node_spacing_distribution,
                                           multilevel_coefficient_distribution,
                                           {201});
    test_recomposition_linearity<2, float>(generator, node_spacing_distribution,
                                           multilevel_coefficient_distribution,
                                           {33, 65});
    test_recomposition_linearity<3, float>(generator, node_spacing_distribution,
                                           multilevel_coefficient_distribution,
                                           {10, 15, 10});
  }

  SECTION("zero coefficients", "[mgard]") {
    std::default_random_engine generator(848733);
    std::uniform_real_distribution<double> node_spacing_distribution(1, 1.1);
    std::uniform_real_distribution<double> multilevel_coefficient_distribution(
        -0.1, 0.1);

    test_recomposition_with_zero_coefficients<1, double>(
        generator, node_spacing_distribution,
        multilevel_coefficient_distribution, {257});
    test_recomposition_with_zero_coefficients<2, double>(
        generator, node_spacing_distribution,
        multilevel_coefficient_distribution, {72, 39});
    test_recomposition_with_zero_coefficients<3, double>(
        generator, node_spacing_distribution,
        multilevel_coefficient_distribution, {17, 15, 9});
  }

  SECTION("on 'flat' meshes", "[mgard]") {
    std::default_random_engine gen(679382);
    // Node spacing distribution.
    std::uniform_real_distribution<float> dis(0.25, 0.35);
    const mgard::TensorMeshHierarchy<3, float> hierarchy =
        hierarchy_with_random_spacing<3, float>(gen, dis, {5, 5, 12});
    const std::size_t ndof = hierarchy.ndof();

    std::vector<float> u_(ndof);
    std::vector<float> expected_(ndof);
    std::vector<float> obtained_(ndof);
    float *const u = u_.data();
    float *const expected = expected_.data();
    float *const obtained = obtained_.data();

    const float s = 0.75;
    generate_reasonable_function(hierarchy, s, gen, expected);
    std::copy(expected, expected + ndof, u);
    mgard::decompose(hierarchy, u);

    {
      const mgard::TensorMeshHierarchy<4, float> flat_hierarchy =
          make_flat_hierarchy<3, 4, float>(hierarchy, {1, 5, 5, 12});
      std::copy(u, u + ndof, obtained);
      mgard::recompose(flat_hierarchy, obtained);
      // Getting slightly higher errors here when precomputing shuffled indices
      // and compiling with `-ffast-math`.
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += obtained[i] == Catch::Approx(expected[i]).margin(1e-7);
      }
      REQUIRE(tracker);
    }

    {
      const mgard::TensorMeshHierarchy<7, float> flat_hierarchy =
          make_flat_hierarchy<3, 7, float>(hierarchy, {1, 5, 1, 5, 1, 12, 1});
      std::copy(u, u + ndof, obtained);
      mgard::recompose(flat_hierarchy, obtained);
      // Getting slightly higher errors here when precomputing shuffled indices
      // and compiling with `-ffast-math`.
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += obtained[i] == Catch::Approx(expected[i]).margin(1e-7);
      }
      REQUIRE(tracker);
    }
  }
}
