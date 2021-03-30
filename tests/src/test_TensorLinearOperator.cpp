#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorLinearOperator.hpp"
#include "TensorMeshHierarchy.hpp"
#include "shuffle.hpp"

namespace {

template <std::size_t N>
class DiagonalOperator : public mgard::ConstituentLinearOperator<N, float> {

  using Real = float;
  using CLO = mgard::ConstituentLinearOperator<N, float>;

public:
  //! Constructor.
  //
  //!\param hierarchy Mesh hierarchy on which the element is defined.
  //!\param l Index of the mesh on which the operator is to be applied.
  //!\param dimension Index of the dimension in which the operator is to
  //! be applied.
  //!\param scalar Diagonal of the matrix.
  DiagonalOperator(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                   const std::size_t l, const std::size_t dimension,
                   const std::vector<Real> &diagonal)
      : CLO(hierarchy, l, dimension), diagonal(diagonal) {
    if (diagonal.size() != this->dimension()) {
      throw std::invalid_argument("operator and mesh sizes don't match");
    }
  }

  //! Diagonal of the matrix.
  const std::vector<Real> diagonal;

private:
  void do_operator_parentheses(const std::array<std::size_t, N> multiindex,
                               Real *const v) const override {
    std::array<std::size_t, N> alpha = multiindex;
    std::size_t &variable_index = alpha.at(CLO::dimension_);
    std::vector<Real>::const_iterator p = diagonal.begin();
    for (const std::size_t index : CLO::indices) {
      variable_index = index;
      CLO::hierarchy->at(v, alpha) *= *p++;
    }
  }
};

class ThreeByThreeMatrix : public mgard::ConstituentLinearOperator<2, float> {
public:
  //! Constructor.
  //
  //!\param hierarchy Mesh hierarchy on which the element is defined.
  //!\param l Index of the mesh on which the operator is to be applied.
  //!\param dimension Index of the dimension in which the operator is to
  //! be applied.
  //!\param coefficients Entries of the matrix (by rows).
  ThreeByThreeMatrix(const mgard::TensorMeshHierarchy<2, float> &hierarchy,
                     const std::size_t l, const std::size_t dimension,
                     const std::array<std::array<float, 3>, 3> &coefficients)
      : ConstituentLinearOperator(hierarchy, l, dimension),
        coefficients(coefficients) {
    if (this->dimension() != 3) {
      throw std::invalid_argument("level must have size 3 in given dimension");
    }
  }

  //! Entries of the matrix (by rows).
  const std::array<std::array<float, 3>, 3> &coefficients;

private:
  void do_operator_parentheses(const std::array<std::size_t, 2> multiindex,
                               float *const v) const override {
    // We'll initialize the entries to zeros inside the loop.
    std::array<float, 3> out;
    std::array<std::size_t, 2> alpha = multiindex;
    std::size_t &variable_index = alpha.at(dimension_);
    for (std::size_t i = 0; i < 3; ++i) {
      float &value = out.at(i);
      const std::array<float, 3> &row = coefficients.at(i);
      value = 0;
      std::array<float, 3>::const_iterator p = row.begin();
      for (const std::size_t index : indices) {
        variable_index = index;
        value += *p++ * hierarchy->at(v, alpha);
      }
    }

    std::array<float, 3>::const_iterator q = out.begin();
    for (const std::size_t index : indices) {
      variable_index = index;
      hierarchy->at(v, alpha) = *q++;
    }
  }
};

} // namespace

TEST_CASE("simple constituent operators", "[TensorLinearOperator]") {
  SECTION("diagonal constituent operators") {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({3, 3});
    const std::size_t ndof = 3 * 3;
    const std::array<float, ndof> u_ = {-1, 5, -3, 0, 1, 4, 2, 3, -2};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();
    {
      const std::size_t l = 1;
      const std::size_t dimension = 1;
      const DiagonalOperator A(hierarchy, l, dimension, {1.5, 0.5, 2.5});
      mgard::shuffle(hierarchy, u, v);
      A({0, 0}, v);
      A({2, 0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<float, ndof> expected = {-1.5, 2.5, -7.5, 0,   1,
                                                4,    3.0, 1.5,  -5.0};
      REQUIRE(buffer_ == expected);
    }
    {
      const std::size_t l = 0;
      const std::size_t dimension = 0;
      const DiagonalOperator A(hierarchy, l, dimension, {-2, -1});
      mgard::shuffle(hierarchy, u, v);
      A({0, 0}, v);
      A({0, 2}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<float, ndof> expected = {2, 5, 6, 0, 1, 4, -2, 3, 2};
      REQUIRE(buffer_ == expected);
    }
  }

  SECTION("matrix constituent operators") {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({5, 5});
    const std::size_t ndof = 5 * 5;
    const std::size_t l = 1;
    const std::array<std::array<float, 3>, 3> coefficients = {
        {{4, -8, 0}, {7, -6, 3}, {-2, -10, -10}}};
    std::array<float, ndof> u_ = {4, 0, -4, -5, -7, 3, 9, 3,  -3, 0, -10, 2, -5,
                                  5, 0, 4,  5,  -6, 4, 5, -9, -1, 0, 9,   -6};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();
    {
      const std::size_t dimension = 0;
      const ThreeByThreeMatrix A(hierarchy, l, dimension, coefficients);
      mgard::shuffle(hierarchy, u, v);
      A({0, 2}, v);
      A({0, 4}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<float, ndof> expected = {
          4, 0,   24, -5, -28, 3, 9, 3,  -3, 0,  -10, 2, 2,
          5, -67, 4,  5,  -6,  4, 5, -9, -1, 58, 9,   74};
      REQUIRE(buffer_ == expected);
    }
    {
      const std::size_t dimension = 1;
      const ThreeByThreeMatrix A(hierarchy, l, dimension, coefficients);
      mgard::shuffle(hierarchy, u, v);
      A({0, 0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<float, ndof> expected = {
          48, 0, 31, -5, 102, 3, 9, 3,  -3, 0, -10, 2, -5,
          5,  0, 4,  5,  -6,  4, 5, -9, -1, 0, 9,   -6};
      REQUIRE(buffer_ == expected);
    }
  }
}

TEST_CASE("tensor products of simple constituent operators",
          "[TensorLinearOperator]") {
  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({2, 2, 2});
    const std::size_t ndof = 2 * 2 * 2;
    const std::size_t l = 0;
    const DiagonalOperator A(hierarchy, l, 0, {2, 1});
    const DiagonalOperator B(hierarchy, l, 1, {1, 3});
    const DiagonalOperator C(hierarchy, l, 2, {1, 5});
    const mgard::TensorLinearOperator<3, float> M(hierarchy, l, {&A, &B, &C});

    const std::array<float, ndof> u_ = {0, 1, 9, 7, 3, 0, 1, 10};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();
    {
      mgard::shuffle(hierarchy, u, v);
      M(v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<float, ndof> expected = {0, 10, 54, 210, 3, 0, 3, 150};
      REQUIRE(buffer_ == expected);
    }
  }

  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({3, 3});
    const std::size_t ndof = 3 * 3;
    const std::size_t l = 1;

    const std::array<std::array<float, 3>, 3> A_ = {
        {{1, 1, 2}, {0, 1, 3}, {0, 0, 1}}};
    const ThreeByThreeMatrix A(hierarchy, l, 0, A_);

    const std::array<std::array<float, 3>, 3> B_ = {
        {{1, 4, 2}, {0, 1, 1}, {0, 0, 1}}};
    const ThreeByThreeMatrix B(hierarchy, l, 1, B_);

    const mgard::TensorLinearOperator<2, float> M(hierarchy, l, {&A, &B});

    const std::array<float, ndof> u_ = {1, 0, 2, 0, 0, 0, 4, 0, 1};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();
    {
      mgard::shuffle(hierarchy, u, v);
      M(v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<float, ndof> expected = {17, 4, 4, 18, 3, 3, 6, 1, 1};
      REQUIRE(buffer_ == expected);
    }
  }
}

TEST_CASE("tensor product linear operators on 'flat' meshes",
          "[TensorLinearOperator]") {
  const std::vector<float> A_diagonal = {2, 2, 3, 5};
  const std::vector<float> B_diagonal = {-2, -1, 0, 1, 2};
  const std::size_t L = 2;
  const std::size_t ndof = 20;
  std::vector<float> u_(ndof);
  std::vector<float> expected_(ndof);
  std::vector<float> obtained_(ndof);
  float *const u = u_.data();
  float *const expected = expected_.data();
  float *const obtained = obtained_.data();
  std::iota(u, u + ndof, 0);
  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({4, 5});
    const DiagonalOperator<2> A(hierarchy, L, 0, A_diagonal);
    const DiagonalOperator<2> B(hierarchy, L, 1, B_diagonal);
    const mgard::TensorLinearOperator T(hierarchy, L, {&A, &B});
    std::copy(u, u + ndof, expected);
    T(expected);
  }

  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({4, 5, 1});
    const DiagonalOperator<3> A(hierarchy, L, 0, A_diagonal);
    const DiagonalOperator<3> B(hierarchy, L, 1, B_diagonal);
    const mgard::TensorLinearOperator<3, float> T(hierarchy, L,
                                                  {&A, &B, nullptr});
    std::copy(u, u + ndof, obtained);
    T(obtained);

    REQUIRE(obtained_ == expected_);
  }

  {
    const mgard::TensorMeshHierarchy<4, float> hierarchy({1, 4, 1, 5});
    const DiagonalOperator<4> A(hierarchy, L, 1, A_diagonal);
    const DiagonalOperator<4> B(hierarchy, L, 3, B_diagonal);
    const mgard::TensorLinearOperator<4, float> T(hierarchy, L,
                                                  {nullptr, &A, nullptr, &B});
    std::copy(u, u + ndof, obtained);
    T(obtained);

    REQUIRE(obtained_ == expected_);
  }
}
