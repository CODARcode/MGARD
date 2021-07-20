#ifndef _MDR_SQUARED_ERROR_ESTIMATOR_HPP
#define _MDR_SQUARED_ERROR_ESTIMATOR_HPP

#include "ErrorEstimatorInterface.hpp"

namespace MDR {
    template<class T>
    class SquaredErrorEstimator : public concepts::ErrorEstimatorInterface<T>{};

    // L2 error estimator for hierarchical basis
    template<class T>
    class L2ErrorEstimator_HB : public SquaredErrorEstimator<T> {
    public:
        L2ErrorEstimator_HB(int num_dims, int target_level){
            s_table = std::vector<T>(target_level + 1);
            for(int i=0; i<=target_level; i++){
                // vol(P_l) where vol(P_l) = 2^(d(L-l))
                int l = i;
                s_table[i] = pow(2, num_dims*(target_level - l));
            }
        }
        L2ErrorEstimator_HB() : L2ErrorEstimator_HB(1, 0) {}
        // need to multiply 2 as (e1 + e2)^2 <= 2(e1^2 + e2^2)
        inline T estimate_error(T error, int level) const {
            return s_table[level] * error * 2;
        }
        inline T estimate_error(T data, T reconstructed_data, int level) const {
            return s_table[level] * (data - reconstructed_data) * 2;
        }
        inline T estimate_error_gain(T base, T current_level_err, T next_level_err, int level) const {
            return s_table[level] * (current_level_err - next_level_err);
        }
        void print() const {
            std::cout << "L2-norm error estimator for hierarchical basis" << std::endl;
        }
    private:
        std::vector<T> s_table;
    };

    // S-norm error estimator for orthogonal basis
    template<class T>
    class SNormErrorEstimator : public SquaredErrorEstimator<T> {
    public:
        SNormErrorEstimator(int num_dims, int target_level, T s) : s(s) {
            s_table = std::vector<T>(target_level + 1);
            for(int i=0; i<=target_level; i++){
                // 2^(sl) * vol(P_l) where vol(P_l) = 2^(d(L-l))
                int l = i;
                s_table[i] = pow(2, 2*s*l + num_dims*(target_level - l));
            }
        }
        SNormErrorEstimator() : SNormErrorEstimator(1, 0, 0) {}
        inline T estimate_error(T error, int level) const {
            return s_table[level] * error;
        }
        inline T estimate_error(T data, T reconstructed_data, int level) const {
            return s_table[level] * (data - reconstructed_data);
        }
        inline T estimate_error_gain(T base, T current_level_err, T next_level_err, int level) const {
            return s_table[level] * (current_level_err - next_level_err);
        }
        void print() const {
            std::cout << "S-norm error estimator (s = " << s << ")." << std::endl;
        }
    private:
        T s = 0;
        std::vector<T> s_table;
    };
}
#endif
