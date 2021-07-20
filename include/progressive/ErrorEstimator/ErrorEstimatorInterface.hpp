#ifndef _MDR_ERROR_ESTIMATOR_INTERFACE_HPP
#define _MDR_ERROR_ESTIMATOR_INTERFACE_HPP

namespace MDR {
    namespace concepts {

        // Error estimator: estimate impact of data error on result error
        template<class T>
        class ErrorEstimatorInterface {
        public:

            ErrorEstimatorInterface() = default;

            virtual ~ErrorEstimatorInterface() = default;

            virtual inline T estimate_error(T error, int level) const = 0;

            virtual inline T estimate_error(T data, T reconstructed_data, int level) const = 0;

            virtual inline T estimate_error_gain(T base, T current_level_err, T next_level_err, int level) const = 0;

            virtual void print() const = 0;
        };
    }
}
#endif
