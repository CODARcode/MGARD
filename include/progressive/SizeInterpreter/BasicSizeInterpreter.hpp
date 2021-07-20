#ifndef _MDR_BASIC_SIZE_INTERPRETER_HPP
#define _MDR_BASIC_SIZE_INTERPRETER_HPP

#include "SizeInterpreterInterface.hpp"

// inorder and round-robin size interpreter

namespace MDR {
    // direct in-order bit-plane retrieval
    template<class ErrorEstimator>
    class InorderSizeInterpreter : public concepts::SizeInterpreterInterface {
    public:
        InorderSizeInterpreter(const ErrorEstimator& e){
            error_estimator = e;
        }
        std::vector<uint32_t> interpret_retrieve_size(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<std::vector<double>>& level_errors, double tolerance, std::vector<uint8_t>& index) const {
            const int num_levels = level_sizes.size();
            std::vector<uint32_t> retrieve_sizes(num_levels, 0);
            double accumulated_error = 0;
            for(int i=0; i<num_levels; i++){
                accumulated_error += error_estimator.estimate_error(level_errors[i][index[i]], i);
            }
            bool tolerance_met = false;
            for(int i=0; i<num_levels; i++){
                for(int j=index[i]; j<level_sizes[i].size(); j++){
                    retrieve_sizes[i] += level_sizes[i][j];
                    index[i] ++;
                    accumulated_error -= error_estimator.estimate_error(level_errors[i][j], i);
                    accumulated_error += error_estimator.estimate_error(level_errors[i][j + 1], i);
                    if(accumulated_error < tolerance){
                        tolerance_met = true;
                        break;
                    }
                }
                if(tolerance_met) break;
            }
            std::cout << "Requested tolerance = " << tolerance << ", estimated error = " << accumulated_error << std::endl;
            return retrieve_sizes;
        }
        void print() const {
            std::cout << "In-order size interpreter." << std::endl;
        }
    private:
        ErrorEstimator error_estimator;
    };
    // Round-robin bit-plane retrieval
    template<class ErrorEstimator>
    class RoundRobinSizeInterpreter : public concepts::SizeInterpreterInterface {
    public:
        RoundRobinSizeInterpreter(const ErrorEstimator& e){
            error_estimator = e;
        }
        std::vector<uint32_t> interpret_retrieve_size(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<std::vector<double>>& level_errors, double tolerance, std::vector<uint8_t>& index) const {
            const int num_levels = level_sizes.size();
            std::vector<uint32_t> retrieve_sizes(num_levels, 0);
            double accumulated_error = 0;
            for(int i=0; i<num_levels; i++){
                accumulated_error += error_estimator.estimate_error(level_errors[i][index[i]], i);
            }
 
            int max_level_size = 0;
            for(int i=0; i<num_levels; i++){
                if(level_sizes[i].size() > max_level_size){
                    max_level_size = level_sizes[i].size();
                }
            }
            bool tolerance_met = false;
            int starting_level = 0;
            for(int i=1; i<num_levels; i++){
                if(index[i] < index[i - 1]){
                    starting_level = i;
                    break;
                }
            }
            for(int i=starting_level; i<num_levels; i++){
                int j = index[i];
                if(j >= level_sizes[i].size()) continue;
                retrieve_sizes[i] += level_sizes[i][j];
                index[i] ++;
                accumulated_error -= error_estimator.estimate_error(level_errors[i][j], i);
                accumulated_error += error_estimator.estimate_error(level_errors[i][j + 1], i);
                if(accumulated_error < tolerance){
                    tolerance_met = true;
                    break;
                }                
            }
            if(!tolerance_met){
                for(int j=0; j<max_level_size; j++){
                    for(int i=0; i<num_levels; i++){
                        j = index[i];
                        if(j >= level_sizes[i].size()) continue;
                        retrieve_sizes[i] += level_sizes[i][j];
                        index[i] ++;
                        accumulated_error -= error_estimator.estimate_error(level_errors[i][j], i);
                        accumulated_error += error_estimator.estimate_error(level_errors[i][j + 1], i);
                        if(accumulated_error < tolerance){
                            tolerance_met = true;
                            break;
                        }
                    }
                    if(tolerance_met) break;
                }                
            }
            std::cout << "Requested tolerance = " << tolerance << ", estimated error = " << accumulated_error << std::endl;
            return retrieve_sizes;
        }
        void print() const {
            std::cout << "Round-robin reorganizer." << std::endl;
        }
    private:
        ErrorEstimator error_estimator;
    };
}
#endif