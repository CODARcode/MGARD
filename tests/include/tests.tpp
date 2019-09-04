#include <stdexcept>

template <typename T>
void assert_true(const T b) {
    if (not b) {
        throw std::runtime_error("assert_true test failed.");
    }
};

template <typename T>
void assert_false(const T b) {
    if (b) {
        throw std::runtime_error("assert_false test failed.");
    }
};

template <typename T, typename U>
void assert_equal(const T a, const U b) {
    if (a != b) {
        throw std::runtime_error("assert_equal test failed.");
    }
}

template <typename T, typename U>
void assert_unequal(const T a, const U b) {
    if (a == b) {
        throw std::runtime_error("assert_unequal test failed.");
    }
}
