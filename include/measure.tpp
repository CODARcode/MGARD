#include <cmath>

namespace helpers {

template <std::size_t D>
double inner_product(double const * const a, double const * const b) {
    double inner_product = 0;
    for (std::size_t i = 0; i < D; ++i) {
        inner_product += a[i] * b[i];
    }
    return inner_product;
}

template <std::size_t D>
double norm(double const * const a) {
    return std::sqrt(inner_product<D>(a, a));
}

template <std::size_t D>
void subtract_into(
    double const * const a, double const * const b, double * const c
) {
    for (std::size_t i = 0; i < D; ++i) {
        c[i] = a[i] - b[i];
    }
}

}
