#ifndef _MDR_GREEDY_BASED_SIZE_INTERPRETER_HPP
#define _MDR_GREEDY_BASED_SIZE_INTERPRETER_HPP

#include "SizeInterpreterInterface.hpp"
#include <queue>
#include "RefactorUtils.hpp"

// inorder and round-robin size interpreter

namespace MDR {
    struct UnitErrorGain{
        double unit_error_gain;
        int level;
        UnitErrorGain(double u, int l) : unit_error_gain(u), level(l) {}
    };
    struct CompareUnitErrorGain{
        bool operator()(const UnitErrorGain& u1, const UnitErrorGain& u2){
            return u1.unit_error_gain < u2.unit_error_gain;
        }
    };
    // greedy bit-plane retrieval
    template<class ErrorEstimator>
    class GreedyBasedSizeInterpreter : public concepts::SizeInterpreterInterface {
    public:
        GreedyBasedSizeInterpreter(const ErrorEstimator& e){
            error_estimator = e;
        }
        std::vector<uint32_t> interpret_retrieve_size(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<std::vector<double>>& level_errors, double tolerance, std::vector<uint8_t>& index) const {
            const int num_levels = level_sizes.size();
            std::vector<uint32_t> retrieve_sizes(num_levels, 0);

            double accumulated_error = 0;
            for(int i=0; i<num_levels; i++){
                accumulated_error += error_estimator.estimate_error(level_errors[i][index[i]], i);
            }
            std::priority_queue<UnitErrorGain, std::vector<UnitErrorGain>, CompareUnitErrorGain> heap;
            for(int i=0; i<num_levels; i++){
                double error_gain = error_estimator.estimate_error_gain(accumulated_error, level_errors[i][index[i]], level_errors[i][index[i] + 1], i);
                heap.push(UnitErrorGain(error_gain / level_sizes[i][index[i]], i));
            }            

            bool tolerance_met = false;
            while((!tolerance_met) && (!heap.empty())){
                auto unit_error_gain = heap.top();
                heap.pop();
                int i = unit_error_gain.level;
                int j = index[i];
                retrieve_sizes[i] += level_sizes[i][j];
                accumulated_error -= error_estimator.estimate_error(level_errors[i][j], i);
                accumulated_error += error_estimator.estimate_error(level_errors[i][j + 1], i);
                if(accumulated_error < tolerance){
                    tolerance_met = true;
                }
                index[i] ++;
                if(index[i] != level_sizes[i].size()){
                    double error_gain = error_estimator.estimate_error_gain(accumulated_error, level_errors[i][index[i]], level_errors[i][index[i] + 1], i);
                    heap.push(UnitErrorGain(error_gain / level_sizes[i][index[i]], i));
                }
                std::cout << i;
            }
            std::cout << std::endl;
            std::cout << "Requested tolerance = " << tolerance << ", estimated error = " << accumulated_error << std::endl;
            return retrieve_sizes;
        }
        void print() const {
            std::cout << "Greedy based size interpreter." << std::endl;
        }
    private:
        ErrorEstimator error_estimator;
    };
    // greedy bit-plane retrieval with sign exculsion (excluding the first component)
    template<class ErrorEstimator>
    class SignExcludeGreedyBasedSizeInterpreter : public concepts::SizeInterpreterInterface {
    public:
        SignExcludeGreedyBasedSizeInterpreter(const ErrorEstimator& e){
            error_estimator = e;
        }
        std::vector<uint32_t> interpret_retrieve_size(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<std::vector<double>>& level_errors, double tolerance, std::vector<uint8_t>& index) const {
            for(int i=0; i<level_errors.size(); i++){
                for(int j=0; j<level_errors[i].size(); j++){
                    std::cout << level_errors[i][j] << " ";
                }
                std::cout << std::endl;
            }
            int num_levels = level_sizes.size();
            std::vector<uint32_t> retrieve_sizes(num_levels, 0);
            double accumulated_error = 0;
            for(int i=0; i<num_levels; i++){
                accumulated_error += error_estimator.estimate_error(level_errors[i][index[i]], i);
            }
            std::priority_queue<UnitErrorGain, std::vector<UnitErrorGain>, CompareUnitErrorGain> heap;
            // identify minimal level
            double min_error = accumulated_error;
            for(int i=0; i<num_levels; i++){
                min_error -= error_estimator.estimate_error(level_errors[i][index[i]], i);
                min_error += error_estimator.estimate_error(level_errors[i].back(), i);
                // fetch the first component if index is 0
                if(index[i] == 0){
                    retrieve_sizes[i] += level_sizes[i][index[i]];
                    accumulated_error -= error_estimator.estimate_error(level_errors[i][index[i]], i);
                    accumulated_error += error_estimator.estimate_error(level_errors[i][index[i] + 1], i);
                    index[i] ++;
                    std::cout << i;
                }
                // push the next one
                if(index[i] != level_sizes[i].size()){
                    double error_gain = error_estimator.estimate_error_gain(accumulated_error, level_errors[i][index[i]], level_errors[i][index[i] + 1], i);
                    heap.push(UnitErrorGain(error_gain / level_sizes[i][index[i]], i));
                }
                if(min_error < tolerance){
                    // the min error of first 0~i levels meets the tolerance
                    num_levels = i + 1;
                    break;
                }
            }

            bool tolerance_met = accumulated_error < tolerance;
            while((!tolerance_met) && (!heap.empty())){
                auto unit_error_gain = heap.top();
                heap.pop();
                int i = unit_error_gain.level;
                int j = index[i];
                retrieve_sizes[i] += level_sizes[i][j];
                accumulated_error -= error_estimator.estimate_error(level_errors[i][j], i);
                accumulated_error += error_estimator.estimate_error(level_errors[i][j + 1], i);
                if(accumulated_error < tolerance){
                    tolerance_met = true;
                }
                index[i] ++;
                if(index[i] != level_sizes[i].size()){
                    double error_gain = error_estimator.estimate_error_gain(accumulated_error, level_errors[i][index[i]], level_errors[i][index[i] + 1], i);
                    heap.push(UnitErrorGain(error_gain / level_sizes[i][index[i]], i));
                }
                std::cout << i;
            }
            std::cout << std::endl;
            std::cout << "Requested tolerance = " << tolerance << ", estimated error = " << accumulated_error << std::endl;
            return retrieve_sizes;
        }
        void print() const {
            std::cout << "Greedy based size interpreter." << std::endl;
        }
    private:
        ErrorEstimator error_estimator;
    };

    struct ConsecutiveUnitErrorGain{
        double unit_error_gain;
        int level;
        int consecutive_num;
        ConsecutiveUnitErrorGain(double u, int l, int n) : unit_error_gain(u), level(l), consecutive_num(n) {}
    };
    struct CompareConsecutiveUnitErrorGain{
        bool operator()(const ConsecutiveUnitErrorGain& u1, const ConsecutiveUnitErrorGain& u2){
            return u1.unit_error_gain < u2.unit_error_gain;
        }
    };
    // greedy bit-plane retrieval for negabinary encoding: allowing for consecutive bitplane that can increase the efficiency
    template<class ErrorEstimator>
    class NegaBinaryGreedyBasedSizeInterpreter : public concepts::SizeInterpreterInterface {
    public:
        NegaBinaryGreedyBasedSizeInterpreter(const ErrorEstimator& e){
            error_estimator = e;
        }
        std::vector<uint32_t> interpret_retrieve_size(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<std::vector<double>>& level_errors, double tolerance, std::vector<uint8_t>& index) const {
            int num_levels = level_sizes.size();
            std::vector<uint32_t> retrieve_sizes(num_levels, 0);
            double accumulated_error = 0;
            for(int i=0; i<num_levels; i++){
                accumulated_error += error_estimator.estimate_error(level_errors[i][index[i]], i);
            }
            std::priority_queue<ConsecutiveUnitErrorGain, std::vector<ConsecutiveUnitErrorGain>, CompareConsecutiveUnitErrorGain> heap;
            // identify minimal level
            double min_error = accumulated_error;
            for(int i=0; i<num_levels; i++){
                min_error -= error_estimator.estimate_error(level_errors[i][index[i]], i);
                min_error += error_estimator.estimate_error(level_errors[i].back(), i);
                // fetch the first component if index is 0
                // if(index[i] == 0){
                //     retrieve_sizes[i] += level_sizes[i][index[i]];
                //     accumulated_error -= error_estimator.estimate_error(level_errors[i][index[i]], i);
                //     accumulated_error += error_estimator.estimate_error(level_errors[i][index[i] + 1], i);
                //     index[i] ++;
                //     std::cout << i;
                // }
                // push the next one
                if(index[i] != level_sizes[i].size()){
                    heap.push(estimated_efficiency(accumulated_error, index[i], i, level_errors[i], level_sizes[i]));
                }
                // if(min_error < tolerance){
                //     // the min error of first 0~i levels meets the tolerance
                //     num_levels = i + 1;
                //     break;
                // }
            }

            bool tolerance_met = accumulated_error < tolerance;
            while((!tolerance_met) && (!heap.empty())){
                auto unit_error_gain = heap.top();
                heap.pop();
                int i = unit_error_gain.level;
                int j = index[i];
                int num = unit_error_gain.consecutive_num;
                for(int k=0; k<num; k++){
                    retrieve_sizes[i] += level_sizes[i][j + k];                
                }
                accumulated_error -= error_estimator.estimate_error(level_errors[i][j], i);
                accumulated_error += error_estimator.estimate_error(level_errors[i][j + num], i);
                if(accumulated_error < tolerance){
                    tolerance_met = true;
                }
                index[i] += num;
                if(index[i] != level_sizes[i].size()){
                    heap.push(estimated_efficiency(accumulated_error, index[i], i, level_errors[i], level_sizes[i]));
                }
                for(int k=0; k<num; k++) std::cout << i;
            }
            std::cout << std::endl;
            std::cout << "Requested tolerance = " << tolerance << ", estimated error = " << accumulated_error << std::endl;
            return retrieve_sizes;
        }
        void print() const {
            std::cout << "Greedy based size interpreter for negabinary encoding." << std::endl;
        }
    private:
        inline ConsecutiveUnitErrorGain estimated_efficiency(double accumulated_error, int index, int level, const std::vector<double>& bitplane_errors, const std::vector<uint32_t>& bitplane_sizes) const {
            double current_error_gain = error_estimator.estimate_error_gain(accumulated_error, bitplane_errors[index], bitplane_errors[index + 1], level);
            uint32_t current_size = bitplane_sizes[index];
            double current_efficiency = current_error_gain / current_size;
            int consecutive_num = 1;
            for(int i=2; i<bitplane_sizes.size() - index; i++){
                double next_error_gain = error_estimator.estimate_error_gain(accumulated_error, bitplane_errors[index], bitplane_errors[index + i], level);             
                uint32_t next_size = current_size + bitplane_sizes[index + i - 1];
                double next_efficiency = next_error_gain / next_size;
                if((current_efficiency > 0) && (current_efficiency > next_efficiency)){
                    break;
                }
                else{
                    current_error_gain = next_error_gain;
                    current_efficiency = next_efficiency;
                    current_size = next_size;
                    consecutive_num = i;
                }
            }
            return ConsecutiveUnitErrorGain(current_efficiency, level, consecutive_num);
        }
        ErrorEstimator error_estimator;
    };

}
#endif
