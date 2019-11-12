#include "catch2/catch.hpp"

#include <cassert>

template <typename T, typename U, typename SizeType>
void require_vector_equality(
    T p, U q, const SizeType N, const double margin
) {
    bool all_close = true;
    for (SizeType i = 0; i < N; ++i) {
        all_close = all_close && *p++ == Approx(*q++).margin(margin);
    }
    REQUIRE(all_close);
}

template <typename T, typename U>
void require_vector_equality(const T &t, const U &u, const double margin) {
    const typename T::size_type N = t.size();
    const typename U::size_type M = u.size();
    assert(N == M);
    require_vector_equality(t.begin(), u.begin(), N, margin);
}
