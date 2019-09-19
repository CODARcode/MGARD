#include "catch2/catch.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include "blaspp/blas.hh"

#include "LinearOperator.hpp"
#include "pcg.hpp"

class SimpleDiagonalMatvec: public helpers::LinearOperator {
    public:
        SimpleDiagonalMatvec(const std::size_t N):
            helpers::LinearOperator(N)
        {
        }

    private:
        virtual void do_operator_parentheses(
            double const * const x, double * const y
        ) const override {
            assert(is_square());
            for (std::size_t i = 0; i < range_dimension; ++i) {
                y[i] = (3 + (i % 5)) * x[i];
            }
        }
};

class Identity: public helpers::LinearOperator {
    public:
        Identity(const std::size_t N):
            helpers::LinearOperator(N)
        {
        }

    private:
        virtual void do_operator_parentheses(
            double const * const x, double * const y
        ) const override {
            assert(is_square());
            blas::copy(domain_dimension, x, 1, y, 1);
        }
};

class FunctionOperator: public helpers::LinearOperator {
    public:
        FunctionOperator(
            const std::size_t N,
            void (*const f)(double const * const, double * const)
        ):
            helpers::LinearOperator(N),
            f(f)
        {
        }

    private:
        void (*f)(double const * const, double * const);

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override {
            assert(is_square());
            assert(domain_dimension == 4);
            return f(x, b);
        }
};

//Mass matrix corresponding to 1D mesh with three elements with volumes of, left
//to right, 3, 4, and 1. Nodes also numbered left to right.
static void mass_matrix_matvec(
    double const * const x, double * const y
) {
    const std::size_t N = 4;
    for (std::size_t i = 0; i < N; ++i) {
        y[i] = 0;
    }
    //Left element, with volume 3.
    y[0] += 3 * (2 * x[0] + 1 * x[1]);
    y[1] += 3 * (2 * x[1] + 1 * x[0]);
    //Middle element, with volume 4.
    y[1] += 4 * (2 * x[1] + 1 * x[2]);
    y[2] += 4 * (2 * x[2] + 1 * x[1]);
    //Right element, with volume 1.
    y[2] += 1 * (2 * x[2] + 1 * x[3]);
    y[3] += 1 * (2 * x[3] + 1 * x[2]);
    //Scale.
    blas::scal(N, 1.0 / 6.0, y, 1);
}

static void diagonal_scaling(
    double const * const x, double * const y
) {
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
            double * const b = (double *) malloc(N * sizeof(*b));
            double * const x = (double *) malloc(N * sizeof(*x));
            double * const buffer = (double *) malloc(4 * N * sizeof(*x));

            for (std::size_t i = 0; i < N; ++i) {
                b[i] = 4 + (i % 7);
                x[i] = 0;
            }

            //With `x` initialized to zeroes.
            {
                double relative_error = helpers::pcg(A, b, P, x, buffer);
                REQUIRE(relative_error < 1e-6);
            }
            //With `x` already the solution.
            {
                double relative_error = helpers::pcg(A, b, P, x, buffer);
                REQUIRE(relative_error < 1e-6);
            }
            //With `x` initialized to something wrong.
            {
                for (std::size_t i = 0; i < N; ++i) {
                    x[i] = i % 2 ? 1 : -1;
                }
                double relative_error = helpers::pcg(A, b, P, x, buffer);
                REQUIRE(relative_error < 1e-6);
            }

            //Testing approximate elementwise accuracy.
            {
                A(x, buffer);
                bool all_close = true;
                for (std::size_t i = 0; i < N; ++i) {
                    all_close = all_close && std::abs(buffer[i] - b[i]) < 1e-3;
                }
                REQUIRE(all_close);
            }

            //With `b` initialized to zeroes.
            {
                for (double *p = b; p != b + N; ++p) {
                    *p = 0;
                }
                double relative_error = helpers::pcg(A, b, P, x, buffer);
                REQUIRE(relative_error == 0);
                bool all_zero = true;
                for (double *p = x; p != x + N; ++p) {
                    all_zero = all_zero && *p == 0;
                }
                REQUIRE(all_zero);
            }

            free(buffer);
            free(x);
            free(b);
        }
    }
    SECTION("mass matrix system") {
        const std::size_t N = 4;
        const FunctionOperator A(N, mass_matrix_matvec);
        const FunctionOperator P(N, diagonal_scaling);
        double x[N];
        double b[N] = {2, -1, -10, 4};
        double buffer[4 * N];
        double rtols[3] = {1e-1, 1e-4, 1e-7};
        for (double rtol : rtols) {
            const double relative_error = helpers::pcg(
                A, b, P, x, buffer, rtol
            );
            const double b_norm = blas::nrm2(N, b, 1);
            //Populate the first bit of `buffer` with the residual.
            blas::copy(N, b, 1, buffer, 1);
            A(x, buffer + N);
            blas::axpy(N, -1, buffer + N, 1, buffer, 1);
            const double residual_norm = blas::nrm2(N, buffer, 1);
            REQUIRE(residual_norm < rtol * b_norm);
            REQUIRE(relative_error <= rtol);
        }
    }
}
