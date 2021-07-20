#ifndef _MDR_HPSS_FILE_RETRIEVER_HPP
#define _MDR_HOSS_FILE_RETRIEVER_HPP

#include "RetrieverInterface.hpp"
#include <cstdio>

namespace MDR {
    // Data retriever for files
    class ConcatLevelFileRetriever : public concepts::RetrieverInterface {
    public:
        ConcatLevelFileRetriever(const std::string& metadata_file, const std::vector<std::string>& level_files) : metadata_file(metadata_file), level_files(level_files) {
            offsets = std::vector<uint32_t>(level_files.size(), 0);
        }

        std::vector<std::vector<const uint8_t*>> retrieve_level_components(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<uint32_t>& retrieve_sizes, const std::vector<uint8_t>& prev_level_num_bitplanes, const std::vector<uint8_t>& level_num_bitplanes){
            assert(offsets.size() == retrieve_sizes.size());
            release();
            uint32_t total_retrieve_size = 0;
            for(int i=0; i<level_files.size(); i++){
                std::cout << "Retrieve " << +level_num_bitplanes[i] << " (" << +(level_num_bitplanes[i] - prev_level_num_bitplanes[i]) << " more) bitplanes from level " << i << std::endl;
                FILE * file = fopen(level_files[i].c_str(), "r");
                if(fseek(file, offsets[i], SEEK_SET)){
                    std::cerr << "Errors in fseek while retrieving from file" << std::endl;
                }
                uint8_t * buffer = (uint8_t *) malloc(retrieve_sizes[i]);
                fread(buffer, sizeof(uint8_t), retrieve_sizes[i], file);
                concated_level_components.push_back(buffer);
                fclose(file);
                offsets[i] += retrieve_sizes[i];
                total_retrieve_size += offsets[i];
            }
            std::cout << "Total retrieve size = " << total_retrieve_size << std::endl;
            return interleave_level_components(level_sizes, prev_level_num_bitplanes, level_num_bitplanes);
        }

        uint8_t * load_metadata() const {
            FILE * file = fopen(metadata_file.c_str(), "r");
            fseek(file, 0, SEEK_END);
            uint32_t num_bytes = ftell(file);
            rewind(file);
            uint8_t * metadata = (uint8_t *) malloc(num_bytes);
            fread(metadata, 1, num_bytes, file);
            fclose(file);
            return metadata;
        }

        void release(){
            for(int i=0; i<concated_level_components.size(); i++){
                free(concated_level_components[i]);
            }
            concated_level_components.clear();
        }

        ~ConcatLevelFileRetriever(){}

        void print() const {
            std::cout << "File retriever." << std::endl;
        }
    private:
        std::vector<std::vector<const uint8_t*>> interleave_level_components(const std::vector<std::vector<uint32_t>>& level_sizes, const std::vector<uint8_t>& prev_level_num_bitplanes, const std::vector<uint8_t>& level_num_bitplanes){
            std::vector<std::vector<const uint8_t*>> level_components;
            for(int i=0; i<level_num_bitplanes.size(); i++){
                const uint8_t * pos = concated_level_components[i];
                std::vector<const uint8_t*> interleaved_level;
                for(int j=prev_level_num_bitplanes[i]; j<level_num_bitplanes[i]; j++){
                    interleaved_level.push_back(pos);
                    pos += level_sizes[i][j];
                }
                level_components.push_back(interleaved_level);
            }
            return level_components;
        }

        std::vector<std::string> level_files;
        std::string metadata_file;
        std::vector<uint32_t> offsets;
        std::vector<uint8_t*> concated_level_components;
    };
}
#endif
