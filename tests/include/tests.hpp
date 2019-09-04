#ifndef TESTS_HPP
#define TESTS_HPP

template <typename T>
void assert_true(const T);

template <typename T>
void assert_false(const T);

template <typename T, typename U>
void assert_equal(const T, const U);

template <typename T, typename  U>
void assert_unequal(const T, const U);

#include "tests.tpp"
#endif
