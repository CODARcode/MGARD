#include "catch2/catch_test_macros.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include "blas.hpp"

#include "unstructured/LinearOperator.hpp"
#include "unstructured/pcg.hpp"

#include "testing_utilities.hpp"

class SimpleDiagonalMatvec : public mgard::LinearOperator {
public:
  explicit SimpleDiagonalMatvec(const std::size_t N)
      : mgard::LinearOperator(N) {}

private:
  virtual void do_operator_parentheses(double const *const x,
                                       double *const y) const override {
    assert(is_square());
    for (std::size_t i = 0; i < range_dimension; ++i) {
      y[i] = (3 + (i % 5)) * x[i];
    }
  }
};

class Identity : public mgard::LinearOperator {
public:
  explicit Identity(const std::size_t N) : mgard::LinearOperator(N) {}

private:
  virtual void do_operator_parentheses(double const *const x,
                                       double *const y) const override {
    assert(is_square());
    blas::copy(domain_dimension, x, y);
  }
};

class FunctionOperator : public mgard::LinearOperator {
public:
  FunctionOperator(const std::size_t N,
                   void (*const f)(double const *const, double *const))
      : mgard::LinearOperator(N), f(f) {}

private:
  void (*f)(double const *const, double *const);

  virtual void do_operator_parentheses(double const *const x,
                                       double *const b) const override {
    assert(is_square());
    assert(domain_dimension == 4);
    return f(x, b);
  }
};

// Mass matrix corresponding to 1D mesh with three elements with volumes of,
// left to right, 3, 4, and 1. Nodes also numbered left to right.
static void mass_matrix_matvec(double const *const x, double *const y) {
  const std::size_t N = 4;
  std::fill(y, y + N, 0);
  // Left element, with volume 3.
  y[0] += 3 * (2 * x[0] + 1 * x[1]);
  y[1] += 3 * (2 * x[1] + 1 * x[0]);
  // Middle element, with volume 4.
  y[1] += 4 * (2 * x[1] + 1 * x[2]);
  y[2] += 4 * (2 * x[2] + 1 * x[1]);
  // Right element, with volume 1.
  y[2] += 1 * (2 * x[2] + 1 * x[3]);
  y[3] += 1 * (2 * x[3] + 1 * x[2]);
  // Scale.
  blas::scal(N, 1.0 / 6.0, y);
}

static void diagonal_scaling(double const *const x, double *const y) {
  y[0] = x[0] / (3);
  y[1] = x[1] / (3 + 4);
  y[2] = x[2] / (4 + 1);
  y[3] = x[3] / (1);
}

TEST_CASE("preconditioned conjugate gradient algorithm", "[pcg]") {
  SECTION("diagonal system") {
    const std::size_t Ns[4] = {1, 11, 111, 1111};
    for (std::size_t N : Ns) {
      const SimpleDiagonalMatvec A(N);
      const Identity P(N);
      double *const b = new double[N];
      double *const x = new double[N];
      double *const buffer = new double[4 * N];

      for (std::size_t i = 0; i < N; ++i) {
        b[i] = 4 + (i % 7);
        x[i] = 0;
      }
      const double b_norm = blas::nrm2(N, b);

      // With `x` initialized to zeroes.
      {
        const mgard::pcg::Diagnostics diagnostics =
            mgard::pcg::pcg(A, b, P, x, buffer);
        REQUIRE(diagnostics.converged);
        REQUIRE(diagnostics.residual_norm <= 1e-6 * b_norm);
      }
      // With `x` already the solution.
      {
        const mgard::pcg::Diagnostics diagnostics =
            mgard::pcg::pcg(A, b, P, x, buffer);
        REQUIRE(diagnostics.converged);
        REQUIRE(diagnostics.residual_norm <= 1e-6 * b_norm);
      }
      // With `x` initialized to something wrong.
      {
        for (std::size_t i = 0; i < N; ++i) {
          x[i] = i % 2 ? 1 : -1;
        }
        const mgard::pcg::Diagnostics diagnostics =
            mgard::pcg::pcg(A, b, P, x, buffer);
        REQUIRE(diagnostics.converged);
        REQUIRE(diagnostics.residual_norm <= 1e-6 * b_norm);
      }
      // Note that I've gotten `nan`s with `x` *un*initialized.

      // Testing approximate elementwise accuracy.
      {
        A(x, buffer);
        require_vector_equality(buffer, b, N, 1e-3);
      }

      // With `b` initialized to zeroes.
      {
        std::fill(b, b + N, 0);
        const mgard::pcg::Diagnostics diagnostics =
            mgard::pcg::pcg(A, b, P, x, buffer);
        REQUIRE(diagnostics.residual_norm == 0);
        TrialTracker tracker;
        for (double *p = x; p != x + N; ++p) {
          tracker += *p == 0;
        }
        REQUIRE(tracker);
      }

      delete[] buffer;
      delete[] x;
      delete[] b;
    }
  }
  SECTION("mass matrix system") {
    const std::size_t N = 4;
    const FunctionOperator A(N, mass_matrix_matvec);
    const FunctionOperator P(N, diagonal_scaling);
    double x[N];
    std::fill(x, x + N, 0);
    double b[N] = {2, -1, -10, 4};
    const double b_norm = blas::nrm2(N, b);
    double buffer[4 * N];
    mgard::pcg::StoppingCriteria criterias[7] = {
        mgard::pcg::StoppingCriteria(1e-1, 0),
        mgard::pcg::StoppingCriteria(1e-4, 0),
        mgard::pcg::StoppingCriteria(1e-7, 0),
        mgard::pcg::StoppingCriteria(0, 1e-2),
        mgard::pcg::StoppingCriteria(0, 1e-4),
        mgard::pcg::StoppingCriteria(0, 1e-6),
        mgard::pcg::StoppingCriteria(1e-2, 1e-2)};
    for (mgard::pcg::StoppingCriteria criteria : criterias) {
      const mgard::pcg::Diagnostics diagnostics =
          mgard::pcg::pcg(A, b, P, x, buffer, criteria);
      // Populate the first bit of `buffer` with the residual.
      blas::copy(N, b, buffer);
      A(x, buffer + N);
      blas::axpy(N, -1.0, buffer + N, buffer);
      const double residual_norm = blas::nrm2(N, buffer);
      REQUIRE(residual_norm <=
              std::max(criteria.absolute, criteria.relative * b_norm));
      if (criteria.absolute <= 0) {
        REQUIRE(diagnostics.residual_norm <= criteria.relative * b_norm);
      }
    }
  }
}
