#ifndef _MDR_HPSS_WRITER_HPP
#define _MDR_HPSS_WRITER_HPP

#include "WriterInterface.hpp"
#include <cstdio>

namespace MDR {
    // A writer that writes the concatenated level components
    // Merge multiple components if size is small
    class HPSSFileWriter : public concepts::WriterInterface {
    public:
        HPSSFileWriter(const std::string& metadata_file, const std::vector<std::string>& level_files, int num_process, int min_HPSS_size) : metadata_file(metadata_file), level_files(level_files), min_size((min_HPSS_size - 1)/num_process + 1) {}

        std::vector<uint32_t> write_level_components(const std::vector<std::vector<uint8_t*>>& level_components, const std::vector<std::vector<uint32_t>>& level_sizes) const {
            std::vector<uint32_t> level_num;
            for(int i=0; i<level_components.size(); i++){
                uint32_t concated_level_size = 0;
                uint32_t prev_index = 0;
                uint32_t count = 0;
                for(int j=0; j<level_components[i].size(); j++){
                    concated_level_size += level_sizes[i][j];
                    if((concated_level_size >= min_size) || (j == level_components[i].size() - 1)){
                        // TODO: deal with the last file that may not be larger than min_size
                        uint8_t * concated_level_data = (uint8_t *) malloc(concated_level_size);
                        uint8_t * concated_level_data_pos = concated_level_data;
                        std::cout << +prev_index << " " << j << " " << concated_level_size << std::endl;
                        for(int k=prev_index + 1; k<=j; k++){
                            memcpy(concated_level_data_pos, level_components[i][k], level_sizes[i][k]);
                            concated_level_data_pos += level_sizes[i][k];
                        }
                        FILE * file = fopen((level_files[i] + "_" + std::to_string(count)).c_str(), "w");
                        fwrite(concated_level_data, 1, concated_level_size, file);
                        fclose(file);
                        free(concated_level_data);
                        count ++;
                        concated_level_size = 0;
                        prev_index = j;
                    }
                }
            }
            return level_num;
        }

        void write_metadata(uint8_t const * metadata, uint32_t size) const {
            FILE * file = fopen(metadata_file.c_str(), "w");
            fwrite(metadata, 1, size, file);
            fclose(file);
        }

        ~HPSSFileWriter(){}

        void print() const {
            std::cout << "HPSS file writer." << std::endl;
        }
    private:
        uint32_t min_size = 0;
        std::vector<std::string> level_files;
        std::string metadata_file;
    };
}
#endif
