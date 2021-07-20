#ifndef _MDR_BASIC_REORGANIZER_HPP
#define _MDR_BASIC_REORGANIZER_HPP

#include "ReorganizerInterface.hpp"

namespace MDR {
    // direct in-order bit-plane placement
    class InOrderReorganizer : public concepts::ReorganizerInterface {
    public:
        InOrderReorganizer(){}
        uint8_t * reorganize(const std::vector<std::vector<uint8_t*>>& level_components, const std::vector<std::vector<uint32_t>>& level_sizes, std::vector<uint8_t>& order, uint32_t& total_size) const {
            const int num_levels = level_sizes.size();
            total_size = 0;
            for(int i=0; i<num_levels; i++){
                for(int j=0; j<level_sizes[i].size(); j++){
                    total_size += level_sizes[i][j];
                }
            }
            uint8_t * reorganized_data = (uint8_t *) malloc(total_size);
            uint8_t * reorganized_data_pos = reorganized_data;
            for(int i=0; i<num_levels; i++){
                for(int j=0; j<level_sizes[i].size(); j++){
                    order.push_back(i);
                    memcpy(reorganized_data_pos, level_components[i][j], level_sizes[i][j]);
                    reorganized_data_pos += level_sizes[i][j];
                }
            }
            return reorganized_data;
        }
        void print() const {
            std::cout << "In-order reorganizer." << std::endl;
        }
    };
    // Round-robin bit-plane placement
    class RoundRobinReorganizer : public concepts::ReorganizerInterface {
    public:
        RoundRobinReorganizer(){}
        uint8_t * reorganize(const std::vector<std::vector<uint8_t*>>& level_components, const std::vector<std::vector<uint32_t>>& level_sizes, std::vector<uint8_t>& order, uint32_t& total_size) const {
            const int num_levels = level_sizes.size();
            total_size = 0;
            for(int i=0; i<num_levels; i++){
                for(int j=0; j<level_sizes[i].size(); j++){
                    total_size += level_sizes[i][j];
                }
            }
            uint8_t * reorganized_data = (uint8_t *) malloc(total_size);
            uint8_t * reorganized_data_pos = reorganized_data;
            int max_level_size = 0;
            for(int i=0; i<num_levels; i++){
                if(level_sizes[i].size() > max_level_size){
                    max_level_size = level_sizes[i].size();
                }
            }
            for(int j=0; j<max_level_size - 1; j++){
                for(int i=0; i<num_levels; i++){
                    if(j >= level_sizes[i].size() - 1) continue;
                    order.push_back(i);
                    memcpy(reorganized_data_pos, level_components[i][j], level_sizes[i][j]);
                    reorganized_data_pos += level_sizes[i][j];
                }
            }
            return reorganized_data;
        }
        void print() const {
            std::cout << "Round-robin reorganizer." << std::endl;
        }
    };
}
#endif