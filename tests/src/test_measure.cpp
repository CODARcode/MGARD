#include "test_measure.hpp"

#include <cstddef>

#include <iostream>

#include "tests.hpp"
#include "measure.hpp"

void copy(double const * const p, const std::size_t N, double * const q) {
    for (std::size_t i = 0; i < N; ++i) {
        q[i] = p[i];
    }
}

void scale(double * const p, const std::size_t N, const double factor) {
    for (std::size_t i = 0; i < N; ++i) {
        p[i] *= factor;
    }
}

void translate(double * const p, const std::size_t N, double const * const q) {
    for (std::size_t i = 0; i < N; ++i) {
        p[i] += q[i];
    }
}

void test_inner_product() {
    const std::size_t N = 5;
    const double a[N] = {3, -2, 0, 3, 1};
    const double b[N] = {5, 9, -4, 1, -2};
    const double product = helpers::inner_product<N>(a, b);
    assert_equal(product, -2);

    //Real inner product.
    std::cout << "    testing symmetry ..." << std::endl;
    assert_equal(product, helpers::inner_product<N>(b, a));

    std::cout << "    testing positive definiteness ..." << std::endl;
    assert_true(helpers::inner_product<N>(a, a) > 0);
    assert_true(helpers::inner_product<N>(b, b) > 0);

    std::cout << "    testing homogeneity ..." << std::endl;
    {
        const double factors[3] = {0, 2, -1};
        for (double factor : factors) {
            double c[N];
            copy(a, N, c);
            scale(c, N, factor);
            assert_equal(helpers::inner_product<N>(c, b), factor * product);
        }
    }

    std::cout << "    testing additivity ..." << std::endl;
    {
        double c[N] = {-1, 4, 4, 0, 3};
        const double c_product = helpers::inner_product<N>(a, c);
        translate(c, N, b);
        assert_equal(c_product + product, helpers::inner_product<N>(a, c));
    }
}

void test_norm() {
    const std::size_t N = 4;
    const double a[N] = {8, -3, 4, -2};
    const double b[N] = {2, -3, 7, 3};
    const double a_norm = helpers::norm<N>(a);
    const double b_norm = helpers::norm<N>(b);

    std::cout << "    testing positive definiteness ..." << std::endl;
    assert_true(a_norm > 0);
    assert_true(b_norm > 0);

    std::cout << "    testing subadditivity ..." << std::endl;
    {
        double c[N];
        copy(a, N, c);
        translate(c, N, b);
        assert_true(a_norm + b_norm >= helpers::norm<N>(c));
    }

    std::cout << "    testing absolute homogeneity ..." << std::endl;
    {
        const double factors[3] = {-10, 0, 3};
        for (double factor : factors) {
            double c[N];
            copy(b, N, c);
            scale(c, N, factor);
            assert_true(
                std::abs(std::abs(factor) * b_norm - helpers::norm<N>(c)) < 1e-6
            );
        }
    }
}

void test_subtract_into() {
    const std::size_t N = 4;
    double a[N] = {22, 8, -4, -18};
    double b[N] = {2, -8, 21, 0};
    double c[N] = {20, 16, -25, -18};
    double d[N];
    helpers::subtract_into<N>(a, b, d);
    for (std::size_t i = 0; i < N; ++i) {
        assert_equal(d[i], c[i]);
    }
}

void test_orient_2d() {
    //Just basic tests. Relying mostly on `test_tri_measure`.
    const std::size_t N = 2;
    const double a[N] = {5, 3};
    const double b[N] = {3, 4};
    const double c[N] = {3, 3};
    assert_equal(helpers::orient_2d(a, b, c), 2);
    assert_equal(helpers::orient_2d(b, a, c), -2);
    assert_equal(helpers::orient_2d(a, a, c), 0);
}

void test_orient_3d() {
    //Just basic tests. Relying mostly on `test_tet_measure`.
    const std::size_t N = 3;
    const double a[N] = {0, 23, 1};
    const double b[N] = {-8, 3, -2};
    const double c[N] = {8, 8, 0};
    const double d[N] = {-4, 22, -1};
    const double determinant = 428;
    assert_equal(helpers::orient_3d(a, b, c, d), determinant);
    assert_equal(helpers::orient_3d(a, b, d, c), -determinant);
}

void test_edge_measure() {
    const std::size_t N = 6;
    const double a[N] = {
        0, 0, 0,
        1,-2, 3
    };
    const double base_length = helpers::edge_measure(a);
    assert_true(std::abs(base_length * base_length - 14) < 1e-6);

    std::cout << "    testing translation invariance ..." << std::endl;
    {
        double b[N];
        copy(a, N, b);
        const double shift[3] = {2, -10, 5};
        for (std::size_t i = 0; i < N; i += 3) {
            translate(b + i, 3, shift);
        }
        assert_true(std::abs(base_length - helpers::edge_measure(b)) < 1e-6);
    }

    std::cout << "    testing dilation ..." << std::endl;
    {
        const double factors[3] = {0.5, 2, 5};
        for (double factor : factors) {
            double b[N];
            copy(a, 6, b);
            //Scale both endpoints at once.
            scale(b, N, factor);
            //Relying on `factor` being nonnegative here.
            const double expected = factor * base_length;
            assert_true(
                std::abs(expected - helpers::edge_measure(b)) < 1e-6 * expected
            );
        }
    }

    std::cout << "    testing permutation ..." << std::endl;
    {
        double b[N];
        copy(a + 0, 3, b + 3);
        copy(a + 3, 3, b + 0);
        assert_true(std::abs(base_length - helpers::edge_measure(b)) < 1e-6);
    }
}

void test_tri_measure() {
    const std::size_t N = 9;
    const double a[N] = {
        3, 1, 1,
        0, 2, 0,
        0, 4, -2
    };
    const double base_area = helpers::tri_measure(a);
    {
        const double expected = 4.242640687119284;
        assert_true(std::abs(expected - base_area) < 1e-6);
    }

    std::cout << "    testing translation invariance ..." << std::endl;
    {
        double b[N];
        copy(a, N, b);
        const double shift[3] = {21, 21, 3};
        for (std::size_t i = 0; i < N; i += 3) {
            translate(b + i, 3, shift);
        }
        assert_true(std::abs(base_area - helpers::tri_measure(b)) < 1e-6);
    }

    std::cout << "    testing dilation ..." << std::endl;
    {
        const double factors[3] = {0.01, 121, 920};
        for (double factor : factors) {
            double b[N];
            copy(a, N, b);
            //Scale all vertices at once.
            scale(b, N, factor);
            const double expected = factor * factor * base_area;
            assert_true(
                std::abs(expected - helpers::tri_measure(b)) < 1e-6 * expected
            );
        }
    }

    std::cout << "    testing permutation ..." << std::endl;
    {
        double b[N];
        copy(a + 0, 3, b + 3);
        copy(a + 3, 3, b + 6);
        copy(a + 6, 3, b + 0);
        assert_true(std::abs(base_area - helpers::tri_measure(b)) < 1e-6);
    }
}

void test_tet_measure() {
    const std::size_t N = 12;
    const double a[N] = {
        0, 0, 0,
        1, 0, 0,
        0, 2, 0,
        0, 0, -4
    };
    const double base_volume = helpers::tet_measure(a);
    {
        const double expected = 8 / 6;
        assert_true(std::abs(expected - base_volume) < 1e-6);
    }

    std::cout << "    testing translation invariance ..." << std::endl;
    {
        double b[N];
        copy(a, N, b);
        const double shift[3] = {-3, 17, 92};
        for (std::size_t i = 0; i < N; i += 3) {
            translate(b + i, 3, shift);
        }
        assert_true(std::abs(base_volume - helpers::tet_measure(b)) < 1e-6);
    }

    std::cout << "    testing dilation ..." << std::endl;
    {
        const double factors[3] = {0.375, 1.8, 12};
        for (double factor : factors) {
            double b[N];
            copy(a, N, b);
            //Scale all vertices at once.
            scale(b, N, factor);
            //Relying on `factor` being nonnegative here.
            const double expected = factor * factor * factor * base_volume;
            assert_true(
                std::abs(expected - helpers::tet_measure(b)) < 1e-6 * expected
            );
        }
    }

    std::cout << "    testing permutation ..." << std::endl;
    {
        double b[N];
        copy(a + 0, 3, b + 6);
        copy(a + 3, 3, b + 0);
        copy(a + 6, 3, b + 9);
        copy(a + 9, 3, b + 3);
        assert_true(std::abs(base_volume - helpers::tet_measure(b)) < 1e-6);
    }
}

void test_measure() {
    std::cout << "  testing inner product ..." << std::endl;
    test_inner_product();

    std::cout << "  testing norm ..." << std::endl;
    test_norm();

    std::cout << "  testing `subtract_into` ..." << std::endl;
    test_subtract_into();

    std::cout << "  testing `orient_2d` ..." << std::endl;
    test_orient_2d();

    std::cout << "  testing `orient_3d` ..." << std::endl;
    test_orient_3d();

    std::cout << "  testing edge measure ..." << std::endl;
    test_edge_measure();

    std::cout << "  testing tri measure ..." << std::endl;
    test_tri_measure();

    //std::cout << "  testing tet measure ..." << std::endl;
    //test_tet_measure();
}
