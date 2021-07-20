#ifndef _MDR_MAX_ERROR_COLLECTOR_HPP
#define _MDR_MAX_ERROR_COLLECTOR_HPP

#include "ErrorCollectorInterface.hpp"

namespace MDR {
    // max error collector: computing according to bit-plane definition
    template<class T>
    class MaxErrorCollector : public concepts::ErrorCollectorInterface<T> {
    public:
        MaxErrorCollector(){
            static_assert(std::is_floating_point<T>::value, "MaxErrorCollector: input data must be floating points.");
        }
        std::vector<double> collect_level_error(T const * data, size_t n, int num_bitplanes, T max_level_error) const {
            int level_exp = 0;
            frexp(max_level_error, &level_exp);
            std::vector<double> max_e = std::vector<double>(num_bitplanes + 1, 0);
            max_e[0] = max_level_error;
            double err = ldexp(1.0, level_exp - 1);
            for(int i=1; i<max_e.size(); i++){
                max_e[i] = err;
                err /= 2;
            }
            return max_e;
        }
        void print() const {
            std::cout << "Max error collector." << std::endl;
        }
    };
}
#endif
