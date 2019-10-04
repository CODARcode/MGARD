#include "catch2/catch.hpp"

#include "LinearOperator.hpp"

class DoNothingOperator: public helpers::LinearOperator {
    public:
        using helpers::LinearOperator::LinearOperator;

    private:
        virtual void do_operator_parentheses(
            double const * const, double * const
        ) const override {
        }
};

TEST_CASE("basic member functions", "[LinearOperator]") {
    {
        DoNothingOperator A(5, 4);
        REQUIRE(!A.is_square());
        std::pair<std::size_t, std::size_t> dims = A.dimensions();
        REQUIRE((dims.first == 5 && dims.second == 4));
    }

    {
        DoNothingOperator B(10, 10);
        REQUIRE(B.is_square());
        std::pair<std::size_t, std::size_t> dims = B.dimensions();
        REQUIRE((dims.first == 10 && dims.second == 10));
    }

    {
        DoNothingOperator C(1);
        REQUIRE(C.is_square());
        std::pair<std::size_t, std::size_t> dims = C.dimensions();
        REQUIRE((dims.first == 1 && dims.second == 1));
    }
}
