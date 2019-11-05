#include "measure.hpp"

#include "blas.hpp"

#include <cmath>
#include <cstddef>

//!Subtract one vector from another.
//!
//!\param [in] D Size of `a`, `b`, and `c`.
//!\param [in] a Subtrahend, of size `D`.
//!\param [in] b Minuend, of size `D`.
//!\param [out] c Difference, of size `D`.
static void subtract_into(
    const std::size_t D,
    double const * const a,
    double const * const b,
    double * const c
) {
    blas::copy(D, a, c);
    blas::axpy(D, -1.0, b, c);
}

namespace mgard {

double orient_2d(
    double const * const a, double const * const b, double const * const c
) {
    double buffer[2][2];
    double * const r = buffer[0];
    double * const s = buffer[1];
    subtract_into(2, a, c, r);
    subtract_into(2, b, c, s);
    return r[0] * s[1] - r[1] * s[0];
}

double orient_3d(
    double const * const a,
    double const * const b,
    double const * const c,
    double const * const d
) {
    double buffer[3][3];
    double * const r = buffer[0];
    double * const s = buffer[1];
    double * const q = buffer[2];
    subtract_into(3, a, d, r);
    subtract_into(3, b, d, s);
    subtract_into(3, c, d, q);
    return (
        r[0] * (s[1] * q[2] - s[2] * q[1]) -
        r[1] * (s[0] * q[2] - s[2] * q[0]) +
        r[2] * (s[0] * q[1] - s[1] * q[0])
    );
}

double edge_measure(double const * const p) {
    double buffer[1][3];
    double * const r = buffer[0];
    subtract_into(3, p + 0, p + 3, r);
    return blas::nrm2(3, r);
}

double tri_measure(double const * const p) {
    double const * const A = p + 0;
    double const * const B = p + 3;
    double const * const C = p + 6;
    double buffer[1][3];
    //Cross product of `(A - C)` and `(B - C)`.
    double * const cross_product = buffer[0];
    for (std::size_t i = 0; i < 3; ++i) {
        double inner_buffer[3][2];
        double * const a = inner_buffer[0];
        double * const b = inner_buffer[1];
        double * const c = inner_buffer[2];
        for (std::size_t j = 0; j < 2; ++j) {
            //Skip `i`.
            const std::size_t index = (i + 1 + j) % 3;
            a[j] = A[index];
            b[j] = B[index];
            c[j] = C[index];
        }
        cross_product[i] = orient_2d(a, b, c);
    }
    return blas::nrm2(3, cross_product) / 2;
}

double tet_measure(double const * const p) {
    double const * const A = p + 0;
    double const * const B = p + 3;
    double const * const C = p + 6;
    double const * const D = p + 9;
    return std::abs(orient_3d(A, B, C, D)) / 6;
}

}
