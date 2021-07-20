#ifndef _MDR_SQUARED_ERROR_COLLECTOR_HPP
#define _MDR_SQUARED_ERROR_COLLECTOR_HPP

#include "ErrorCollectorInterface.hpp"

namespace MDR {
    union FloatingInt32{
        float f;
        uint32_t i;
    };
    union FloatingInt64{
        double f;
        uint64_t i;
    };
    // s-norm error collector: collecting sum of squared errors
    template<class T>
    class SquaredErrorCollector : public concepts::ErrorCollectorInterface<T> {
    public:
        SquaredErrorCollector(){
            static_assert(std::is_floating_point<T>::value, "SquaredErrorCollector: input data must be floating points.");
            static_assert(!std::is_same<T, long double>::value, "SquaredErrorCollector: long double is not supported.");
        }
        std::vector<double> collect_level_error(T const * data, size_t n, int num_bitplanes, T max_level_error) const {
            int level_exp = 0;
            frexp(max_level_error, &level_exp);
            const int prec = std::is_same<T, double>::value ? 52 : 23;
            using FloatingInt = typename std::conditional<std::is_same<T, double>::value, FloatingInt64, FloatingInt32>::type;
            const int encode_prec = num_bitplanes;
            std::vector<double> squared_error = std::vector<double>(num_bitplanes + 1, 0);
            FloatingInt fi;
            for(int i=0; i<n; i++){
                if(data[i] == 0) continue;
                int data_exp = 0;
                frexp(data[i], &data_exp);
                T val = data[i];
                fi.f = val;
                int exp_diff = level_exp - data_exp + prec - encode_prec;
                // int exp_diff = level_exp - encode_prec - data_exp + prec + 1;
                int index = encode_prec;
                if(exp_diff > 0){
                    // zeroing out unrecorded bitplanes
                    for(int b=0; b<exp_diff; b++){
                        fi.i &= ~(1u << b);            
                    }
                }
                else{
                    // skip padding 0s (no errors)
                    index += exp_diff;
                    exp_diff = 0;
                }
                if(index > 0){
                    for(int b=exp_diff; b<prec; b++){
                        // change b-th bit to 0
                        fi.i &= ~(1u << b);
                        squared_error[index] += (data[i] - fi.f)*(data[i] - fi.f);
                        index --;
                    }
                    while(index >= 0){
                        squared_error[index] += data[i] * data[i];
                        index --;
                    }
                }
            }
            return squared_error;
        }
        void print() const {
            std::cout << "Squared error collector." << std::endl;
        }
    };
}
#endif
