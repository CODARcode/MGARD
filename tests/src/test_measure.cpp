#include "catch2/catch.hpp"

#include <cstddef>

#include "measure.hpp"

static void copy(
    double const * const p, const std::size_t N, double * const q
) {
    for (std::size_t i = 0; i < N; ++i) {
        q[i] = p[i];
    }
}

static void scale(double * const p, const std::size_t N, const double factor) {
    for (std::size_t i = 0; i < N; ++i) {
        p[i] *= factor;
    }
}

static void translate(
    double * const p, const std::size_t N, double const * const q
) {
    for (std::size_t i = 0; i < N; ++i) {
        p[i] += q[i];
    }
}

TEST_CASE("inner product", "[measure]") {
    const std::size_t N = 5;
    const double a[N] = {3, -2, 0, 3, 1};
    const double b[N] = {5, 9, -4, 1, -2};
    const double product = helpers::inner_product<N>(a, b);
    REQUIRE(product == -2);

    SECTION("inner product is symmetric") {
        REQUIRE(product == helpers::inner_product<N>(b, a));
    }

    SECTION("inner product is positive definite") {
        REQUIRE(helpers::inner_product<N>(a, a) > 0);
        REQUIRE(helpers::inner_product<N>(b, b) > 0);
    }

    SECTION("inner product is homogeneous") {
        const double factors[3] = {0, 2, -1};
        for (double factor : factors) {
            double c[N];
            copy(a, N, c);
            scale(c, N, factor);
            REQUIRE(helpers::inner_product<N>(c, b) == factor * product);
        }
    }

    SECTION("inner product is additive") {
        double c[N] = {-1, 4, 4, 0, 3};
        const double c_product = helpers::inner_product<N>(a, c);
        translate(c, N, b);
        REQUIRE(c_product + product == helpers::inner_product<N>(a, c));
    }
}

TEST_CASE("norm", "[measure]") {
    const std::size_t N = 4;
    const double a[N] = {8, -3, 4, -2};
    const double b[N] = {2, -3, 7, 3};
    const double a_norm = helpers::norm<N>(a);
    const double b_norm = helpers::norm<N>(b);

    SECTION("norm is positive definite") {
        REQUIRE(a_norm > 0);
        REQUIRE(b_norm > 0);
    }

    SECTION("norm is subadditive") {
        double c[N];
        copy(a, N, c);
        translate(c, N, b);
        REQUIRE(a_norm + b_norm >= helpers::norm<N>(c));
    }

    SECTION("norm is absolute homogeneous") {
        const double factors[3] = {-10, 0, 3};
        for (double factor : factors) {
            double c[N];
            copy(b, N, c);
            scale(c, N, factor);
            REQUIRE(helpers::norm<N>(c) == Approx(std::abs(factor) * b_norm));
        }
    }
}

TEST_CASE("`subtract_into`", "[measure]") {
    const std::size_t N = 4;
    double a[N] = {22, 8, -4, -18};
    double b[N] = {2, -8, 21, 0};
    double c[N] = {20, 16, -25, -18};
    double d[N];
    helpers::subtract_into<N>(a, b, d);
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(d[i] == c[i]);
    }
}

TEST_CASE("`orient_2d`", "[measure]") {
    //Just basic tests. Relying mostly on `test_tri_measure`.
    const std::size_t N = 2;
    const double a[N] = {5, 3};
    const double b[N] = {3, 4};
    const double c[N] = {3, 3};
    REQUIRE(helpers::orient_2d(a, b, c) == 2);
    REQUIRE(helpers::orient_2d(b, a, c) == -2);
    REQUIRE(helpers::orient_2d(a, a, c) == 0);
}

TEST_CASE("`orient_3d`", "[measure]") {
    //Just basic tests. Relying mostly on `test_tet_measure`.
    const std::size_t N = 3;
    const double a[N] = {0, 23, 1};
    const double b[N] = {-8, 3, -2};
    const double c[N] = {8, 8, 0};
    const double d[N] = {-4, 22, -1};
    const double determinant = 428;
    REQUIRE(helpers::orient_3d(a, b, c, d) == determinant);
    REQUIRE(helpers::orient_3d(a, b, d, c) == -determinant);
}

TEST_CASE("edge measure", "[measure]") {
    const std::size_t N = 6;
    const double a[N] = {
        0, 0, 0,
        1,-2, 3
    };
    const double base_length = helpers::edge_measure(a);
    REQUIRE(base_length == Approx(std::sqrt(14)));

    SECTION("edge measure respects translation invariance") {
        double b[N];
        copy(a, N, b);
        const double shift[3] = {2, -10, 5};
        for (std::size_t i = 0; i < N; i += 3) {
            translate(b + i, 3, shift);
        }
        REQUIRE(helpers::edge_measure(b) == Approx(base_length));
    }

    SECTION("edge measure behaves properly under dilation") {
        const double factors[3] = {0.5, 2, 5};
        for (double factor : factors) {
            double b[N];
            copy(a, 6, b);
            //Scale both endpoints at once.
            scale(b, N, factor);
            //Relying on `factor` being nonnegative here.
            const double expected = factor * base_length;
            REQUIRE(helpers::edge_measure(b) == Approx(expected));
        }
    }

    SECTION("edge measure invariant under permutation") {
        double b[N];
        copy(a + 0, 3, b + 3);
        copy(a + 3, 3, b + 0);
        REQUIRE(helpers::edge_measure(b) == Approx(base_length));
    }
}

TEST_CASE("triangle measure", "[measure]") {
    const std::size_t N = 9;
    const double a[N] = {
        3, 1, 1,
        0, 2, 0,
        0, 4, -2
    };
    const double base_area = helpers::tri_measure(a);
    {
        const double expected = 4.242640687119284;
        REQUIRE(base_area == Approx(expected));
    }

    SECTION("triangle measure respects translation invariance") {
        double b[N];
        copy(a, N, b);
        const double shift[3] = {21, 21, 3};
        for (std::size_t i = 0; i < N; i += 3) {
            translate(b + i, 3, shift);
        }
        REQUIRE(helpers::tri_measure(b) == Approx(base_area));
    }

    SECTION("triangle measure behaves properly under dilation") {
        const double factors[3] = {0.01, 121, 920};
        for (double factor : factors) {
            double b[N];
            copy(a, N, b);
            //Scale all vertices at once.
            scale(b, N, factor);
            const double expected = factor * factor * base_area;
            REQUIRE(helpers::tri_measure(b) == Approx(expected));
        }
    }

    SECTION("triangle measure invariant under permutation") {
        double b[N];
        copy(a + 0, 3, b + 3);
        copy(a + 3, 3, b + 6);
        copy(a + 6, 3, b + 0);
        REQUIRE(helpers::tri_measure(b) == Approx(base_area));
    }
}

TEST_CASE("tetrahedron measure", "[measure]") {
    const std::size_t N = 12;
    const double a[N] = {
        0, 0, 0,
        1, 0, 0,
        0, 2, 0,
        0, 0, -4
    };
    const double base_volume = helpers::tet_measure(a);
    {
        const double expected = 8. / 6.;
        REQUIRE(base_volume == Approx(expected));
    }

    SECTION("tetrahedron measure respects translation invariance") {
        double b[N];
        copy(a, N, b);
        const double shift[3] = {-3, 17, 92};
        for (std::size_t i = 0; i < N; i += 3) {
            translate(b + i, 3, shift);
        }
        REQUIRE(helpers::tet_measure(b) == Approx(base_volume));
    }

    SECTION("tetrahedron measure behaves properly under dilation") {
        const double factors[3] = {0.375, 1.8, 12};
        for (double factor : factors) {
            double b[N];
            copy(a, N, b);
            //Scale all vertices at once.
            scale(b, N, factor);
            //Relying on `factor` being nonnegative here.
            const double expected = factor * factor * factor * base_volume;
            REQUIRE(helpers::tet_measure(b) == Approx(expected));
        }
    }

    SECTION("tetrahedron measure invariant under permutation") {
        double b[N];
        copy(a + 0, 3, b + 6);
        copy(a + 3, 3, b + 0);
        copy(a + 6, 3, b + 9);
        copy(a + 9, 3, b + 3);
        REQUIRE(helpers::tet_measure(b) == Approx(base_volume));
    }
}
