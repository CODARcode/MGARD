#ifndef MEASURE_HPP
#define MEASURE_HPP

#include <cstddef>

namespace helpers {

template <std::size_t D>
double inner_product(double const * const a, double const * const b);

template <std::size_t D>
double norm(double const * const a);

template <std::size_t D>
void subtract_into(
    double const * const a,
    double const * const b,
    double const * const c
);

//See §2 of <https://people.eecs.berkeley.edu/~jrs/meshpapers/robnotes.pdf>.
double orient_2d(
    double const * const a,
    double const * const b,
    double const * const c
);

double orient_3d(
    double const * const a,
    double const * const b,
    double const * const c,
    double const * const d
);

double edge_measure(double const * const p);

double tri_measure(double const * const p);

double tet_measure(double const * const p);

}

#include "measure.tpp"
#endif
