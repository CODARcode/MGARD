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
            const double b_norm = blas::nrm2(N, b, 1);

            //With `x` initialized to zeroes.
            {
                const helpers::PCGDiagnostics diagnostics = helpers::pcg(
                    A, b, P, x, buffer
                );
                REQUIRE(diagnostics.converged);
                REQUIRE(diagnostics.residual_norm <= 1e-6 * b_norm);
            }
            //With `x` already the solution.
            {
                const helpers::PCGDiagnostics diagnostics = helpers::pcg(
                    A, b, P, x, buffer
                );
                REQUIRE(diagnostics.converged);
                REQUIRE(diagnostics.residual_norm <= 1e-6 * b_norm);
            }
            //With `x` initialized to something wrong.
            {
                for (std::size_t i = 0; i < N; ++i) {
                    x[i] = i % 2 ? 1 : -1;
                }
                const helpers::PCGDiagnostics diagnostics = helpers::pcg(
                    A, b, P, x, buffer
                );
                REQUIRE(diagnostics.converged);
                REQUIRE(diagnostics.residual_norm <= 1e-6 * b_norm);
            }
            //Note that I've gotten `nan`s with `x` *un*initialized.

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
                const helpers::PCGDiagnostics diagnostics = helpers::pcg(
                    A, b, P, x, buffer
                );
                REQUIRE(diagnostics.residual_norm == 0);
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
        for (double *p = x; p != x + N; ++p) {
            *p = 0;
        }
        double b[N] = {2, -1, -10, 4};
        const double b_norm = blas::nrm2(N, b, 1);
        double buffer[4 * N];
        helpers::PCGStoppingCriteria criterias[7] = {
            helpers::PCGStoppingCriteria(1e-1,    0),
            helpers::PCGStoppingCriteria(1e-4,    0),
            helpers::PCGStoppingCriteria(1e-7,    0),
            helpers::PCGStoppingCriteria(   0, 1e-2),
            helpers::PCGStoppingCriteria(   0, 1e-4),
            helpers::PCGStoppingCriteria(   0, 1e-6),
            helpers::PCGStoppingCriteria(1e-2, 1e-2)
        };
        for (helpers::PCGStoppingCriteria criteria : criterias) {
            const helpers::PCGDiagnostics diagnostics = helpers::pcg(
                A, b, P, x, buffer, criteria
            );
            //Populate the first bit of `buffer` with the residual.
            blas::copy(N, b, 1, buffer, 1);
            A(x, buffer + N);
            blas::axpy(N, -1, buffer + N, 1, buffer, 1);
            const double residual_norm = blas::nrm2(N, buffer, 1);
            REQUIRE(residual_norm <= std::max(
                criteria.absolute, criteria.relative * b_norm
            ));
            if (criteria.absolute <= 0) {
                REQUIRE(
                    diagnostics.residual_norm <= criteria.relative * b_norm
                );
            }
        }
    }
}
