#include "catch2/catch.hpp"

#include <cstddef>

#include <random>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "interpolation.hpp"

// Wrapping for convenience.

template <typename Real>
Real interpolate(const std::array<Real, 2> &qs, const std::array<Real, 2> &xs,
                 const std::array<Real, 1> &x) {
  return mgard::interpolate<Real>(qs.at(0), qs.at(1), xs.at(0), xs.at(1),
                                  x.at(0));
}

template <typename Real>
Real interpolate(const std::array<Real, 4> &qs, const std::array<Real, 2> &xs,
                 const std::array<Real, 2> &ys, const std::array<Real, 2> &xy) {
  return mgard::interpolate<Real>(qs.at(0), qs.at(1), qs.at(2), qs.at(3),
                                  xs.at(0), xs.at(1), ys.at(0), ys.at(1),
                                  xy.at(0), xy.at(1));
}

template <typename Real>
Real interpolate(const std::array<Real, 8> &qs, const std::array<Real, 2> &xs,
                 const std::array<Real, 2> &ys, const std::array<Real, 2> &zs,
                 const std::array<Real, 3> &xyz) {
  return mgard::interpolate<Real>(
      qs.at(0), qs.at(1), qs.at(2), qs.at(3), qs.at(4), qs.at(5), qs.at(6),
      qs.at(7), xs.at(0), xs.at(1), ys.at(0), ys.at(1), zs.at(0), zs.at(1),
      xyz.at(0), xyz.at(1), xyz.at(2));
}

TEMPLATE_TEST_CASE("multilinear interpolation", "[interpolation]", float,
                   double) {
  // Mismatches of small magnitude do occasionally occur here, especially in 3D.
  // To see some, set `N` and `M` to 100.
  const std::size_t N = 10;
  const std::size_t M = 10;

  std::default_random_engine generator(902107);
  std::uniform_real_distribution<TestType> distribution(2, 3);

  SECTION("1D") {
    TrialTracker tracker;
    for (std::size_t i = 0; i < N; ++i) {
      MultilinearPolynomial<TestType, 1> p(generator, distribution);
      const std::array<TestType, 2> xs = {distribution(generator),
                                          distribution(generator)};
      const std::array<TestType, 1 << 1> qs = {p({xs.at(0)}), p({xs.at(1)})};
      for (std::size_t j = 0; j < M; ++j) {
        const std::array<TestType, 1> x = {distribution(generator)};
        tracker +=
            p(x) == Approx(interpolate<TestType>(qs, xs, x)).epsilon(0.001);
      }
    }
    REQUIRE(tracker);
  }

  SECTION("2D") {
    TrialTracker tracker;
    for (std::size_t i = 0; i < N; ++i) {
      MultilinearPolynomial<TestType, 2> p(generator, distribution);
      const std::array<TestType, 2> xs = {distribution(generator),
                                          distribution(generator)};
      const std::array<TestType, 2> ys = {distribution(generator),
                                          distribution(generator)};
      const std::array<TestType, 1 << 2> qs = {
          p({xs.at(0), ys.at(0)}), p({xs.at(0), ys.at(1)}),
          p({xs.at(1), ys.at(0)}), p({xs.at(1), ys.at(1)})};
      for (std::size_t j = 0; j < M; ++j) {
        const std::array<TestType, 2> xy = {distribution(generator),
                                            distribution(generator)};
        tracker += p(xy) ==
                   Approx(interpolate<TestType>(qs, xs, ys, xy)).epsilon(0.001);
      }
    }
    REQUIRE(tracker);
  }

  SECTION("3D") {
    TrialTracker tracker;
    for (std::size_t i = 0; i < N; ++i) {
      MultilinearPolynomial<TestType, 3> p(generator, distribution);
      const std::array<TestType, 2> xs = {distribution(generator),
                                          distribution(generator)};
      const std::array<TestType, 2> ys = {distribution(generator),
                                          distribution(generator)};
      const std::array<TestType, 2> zs = {distribution(generator),
                                          distribution(generator)};
      const std::array<TestType, 1 << 3> qs = {
          p({xs.at(0), ys.at(0), zs.at(0)}), p({xs.at(0), ys.at(0), zs.at(1)}),
          p({xs.at(0), ys.at(1), zs.at(0)}), p({xs.at(0), ys.at(1), zs.at(1)}),
          p({xs.at(1), ys.at(0), zs.at(0)}), p({xs.at(1), ys.at(0), zs.at(1)}),
          p({xs.at(1), ys.at(1), zs.at(0)}), p({xs.at(1), ys.at(1), zs.at(1)}),
      };
      for (std::size_t j = 0; j < M; ++j) {
        const std::array<TestType, 3> xyz = {distribution(generator),
                                             distribution(generator),
                                             distribution(generator)};
        tracker +=
            p(xyz) ==
            Approx(interpolate<TestType>(qs, xs, ys, zs, xyz)).epsilon(0.001);
      }
    }
    REQUIRE(tracker);
  }
}
