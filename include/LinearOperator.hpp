#ifndef LINEAROPERATOR_HPP
#define LINEAROPERATOR_HPP
//!\file
//!\brief Base class for representing linear operators \f$R^{N} \to R^{M}\f$.

#include <cstddef>

#include <utility>

namespace helpers {

//!Linear operator with respect to some fixed bases.
class LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param N Dimension of domain.
        //!\param M Dimension of range.
        LinearOperator(const std::size_t N, const std::size_t M);

        //!Constructor.
        //!
        //!\overload
        //!
        //!\param N Dimension of domain and range.
        LinearOperator(const std::size_t N);

        //!Return the dimensions of the domain and range.
        std::pair<std::size_t, std::size_t> dimensions() const;

        //!Report whether the associated matrix is square.
        bool is_square() const;

        //!Apply the operator to a vector and store the results.
        //!
        //!\param [in] x Vector in the domain.
        //!\param [out] b Vector in the range, obtained by applying the operator
        //!to `x`.
        void operator()(double const * const x, double * const b) const;

    protected:
        //!Dimension of the domain.
        std::size_t domain_dimension;
        //!Dimension of the range.
        std::size_t range_dimension;

    private:
        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const = 0;
};

}

#endif
