#include "catch2/catch.hpp"

#include <array>
#include <stdexcept>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorLinearOperator.hpp"
#include "TensorMeshHierarchy.hpp"

namespace {

class DiagonalOperator : public mgard::ConstituentLinearOperator<2, float> {
public:
  //! Constructor.
  //
  //!\param hierarchy Mesh hierarchy on which the element is defined.
  //!\param l Index of the mesh on which the operator is to be applied.
  //!\param dimension Index of the dimension in which the operator is to
  //! be applied.
  //!\param scalar Diagonal of the matrix.
  DiagonalOperator(const mgard::TensorMeshHierarchy<2, float> &hierarchy,
                   const std::size_t l, const std::size_t dimension,
                   const std::vector<float> diagonal)
      : ConstituentLinearOperator(hierarchy, l, dimension), diagonal(diagonal) {
    if (diagonal.size() != this->dimension()) {
      throw std::invalid_argument("operator and mesh sizes don't match");
    }
  }

  //! Diagonal of the matrix.
  const std::vector<float> diagonal;

private:
  void do_operator_parentheses(const std::array<std::size_t, 2> multiindex,
                               float *const v) const override {
    std::array<std::size_t, 2> alpha = multiindex;
    std::size_t &variable_index = alpha.at(dimension_);
    const std::size_t N = dimension();
    for (std::size_t i = 0; i < N; ++i) {
      variable_index = indices.at(i);
      hierarchy.at(v, alpha) *= diagonal.at(i);
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
      for (std::size_t j = 0; j < 3; ++j) {
        variable_index = indices.at(j);
        value += row.at(j) * hierarchy.at(v, alpha);
      }
    }

    for (std::size_t i = 0; i < 3; ++i) {
      variable_index = indices.at(i);
      hierarchy.at(v, alpha) = out.at(i);
    }
  }
};

} // namespace

TEST_CASE("simple constituent operators", "[TensorLinearOperator]") {
  SECTION("diagonal constituent operators") {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({3, 3});
    const std::array<float, 9> u_ = {-1, 5, -3, 0, 1, 4, 2, 3, -2};
    {
      const DiagonalOperator A(hierarchy, 1, 1, {1.5, 0.5, 2.5});
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      A({0, 0}, v);
      A({2, 0}, v);
      const std::array<float, 9> expected = {-1.5, 2.5, -7.5, 0,   1,
                                             4,    3.0, 1.5,  -5.0};
      REQUIRE(v_ == expected);
    }
    {
      const DiagonalOperator A(hierarchy, 0, 0, {-2, -1});
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      A({0, 0}, v);
      A({0, 2}, v);
      const std::array<float, 9> expected = {2, 5, 6, 0, 1, 4, -2, 3, 2};
      REQUIRE(v_ == expected);
    }
  }

  SECTION("matrix constituent operators") {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({5, 5});
    std::array<float, 25> u_ = {4, 0, -4, -5, -7, 3, 9, 3,  -3, 0, -10, 2, -5,
                                5, 0, 4,  5,  -6, 4, 5, -9, -1, 0, 9,   -6};
    const std::array<std::array<float, 3>, 3> coefficients = {
        {{4, -8, 0}, {7, -6, 3}, {-2, -10, -10}}};
    {
      const ThreeByThreeMatrix A(hierarchy, 1, 0, coefficients);
      std::array<float, 25> v_ = u_;
      float *const v = v_.data();
      A({0, 2}, v);
      A({0, 4}, v);
      const std::array<float, 25> expected = {
          4, 0,   24, -5, -28, 3, 9, 3,  -3, 0,  -10, 2, 2,
          5, -67, 4,  5,  -6,  4, 5, -9, -1, 58, 9,   74};
      REQUIRE(v_ == expected);
    }
    {
      const ThreeByThreeMatrix A(hierarchy, 1, 1, coefficients);
      std::array<float, 25> v_ = u_;
      float *const v = v_.data();
      A({0, 0}, v);
      const std::array<float, 25> expected = {48, 0,   31, -5, 102, 3, 9, 3, -3,
                                              0,  -10, 2,  -5, 5,   0, 4, 5, -6,
                                              4,  5,   -9, -1, 0,   9, -6};
      REQUIRE(v_ == expected);
    }
  }
}
