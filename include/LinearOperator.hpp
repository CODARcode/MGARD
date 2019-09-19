#ifndef LINEAROPERATOR_HPP
#define LINEAROPERATOR_HPP
//!\file
//!\brief Base class for representing linear operators \f$R^{N} \to R^{M}\f$.

#include <cstddef>

namespace helpers {

class LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param N Dimension of domain.
        //!\param N Dimension of range.
        LinearOperator(const std::size_t N, const std::size_t M);

        //!Constructor.
        //!
        //!\override
        //!
        //!\param N Dimension of domain and range.
        LinearOperator(const std::size_t N);

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
